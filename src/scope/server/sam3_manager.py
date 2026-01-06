import base64
import contextlib
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .models_config import get_assets_dir

logger = logging.getLogger(__name__)
SAM3_DEBUG = os.getenv("BURN_DEBUG_SAM3") == "1"
SAM3_MASK_DILATE = 0
SAM3_MASK_DILATE_ITERS = 0
SAM3_MASK_BLUR = 0
SAM3_MASK_INTENSITY = 1.0
SAM3_PROMPT_STRIDE = 1
SAM3_PERSON_PROMPT = "person"
SAM3_MAX_FPS = 15.0

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass
class Sam3MaskSession:
    session_id: str
    mask_dir: Path
    video_path: Path
    height: int
    width: int
    frame_count: int
    prompt: str
    input_fps: float | None
    sam3_fps: float | None
    applied_mask_indices: list[int] = field(default_factory=list)


class Sam3MaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictor = None
        self._predictor_dtype: torch.dtype | None = None
        self._sessions: dict[str, Sam3MaskSession] = {}
        self._applied_lock = threading.Lock()

        assets_dir = get_assets_dir()
        self._masks_dir = assets_dir / "sam3_masks"
        self._masks_dir.mkdir(parents=True, exist_ok=True)

    def _force_predictor_float32(self, predictor) -> None:
        def _coerce_module(module: torch.nn.Module) -> None:
            for name, param in module.named_parameters(recurse=False):
                if param is not None and param.data.dtype != torch.float32:
                    param.data = param.data.float()
            for name, buf in module.named_buffers(recurse=False):
                if buf is not None and buf.data.dtype != torch.float32:
                    module._buffers[name] = buf.float()
            for child in module.children():
                _coerce_module(child)
            if hasattr(module, "dtype") and isinstance(getattr(module, "dtype"), torch.dtype):
                setattr(module, "dtype", torch.float32)

        for attr in ("model", "sam_model", "sam3_model", "module"):
            obj = getattr(predictor, attr, None)
            if isinstance(obj, torch.nn.Module):
                _coerce_module(obj)
            elif obj is not None and hasattr(obj, "float"):
                obj.float()
                if hasattr(obj, "dtype") and isinstance(getattr(obj, "dtype"), torch.dtype):
                    setattr(obj, "dtype", torch.float32)

        if isinstance(predictor, torch.nn.Module):
            _coerce_module(predictor)
        elif hasattr(predictor, "float"):
            predictor.float()
            if hasattr(predictor, "dtype") and isinstance(getattr(predictor, "dtype"), torch.dtype):
                setattr(predictor, "dtype", torch.float32)

    def _disable_sam3_autocast(self) -> None:
        patched = []
        def _sam3_no_autocast(*args, **kwargs):  # type: ignore[no-untyped-def]
            return contextlib.nullcontext()
        for module_name in (
            "sam3.model.sam3_video_inference",
            "sam3.model.sam3_video_base",
            "sam3.model.sam3_video_predictor",
        ):
            try:
                module = __import__(module_name, fromlist=["*"])
            except Exception:
                continue
            if hasattr(module, "autocast"):
                try:
                    setattr(module, "autocast", _sam3_no_autocast)
                    patched.append(module_name)
                except Exception:
                    logger.exception("Failed to patch SAM3 autocast in %s", module_name)
        if patched:
            logger.info("SAM3 autocast disabled in modules: %s", ", ".join(patched))

    @contextlib.contextmanager
    def _disable_autocast(self):
        orig_autocast = getattr(torch, "autocast", None)
        orig_cuda_autocast = None
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
            orig_cuda_autocast = getattr(torch.cuda.amp, "autocast", None)

        def _no_autocast(*args, **kwargs):  # type: ignore[no-untyped-def]
            return contextlib.nullcontext()

        if orig_autocast is not None:
            torch.autocast = _no_autocast  # type: ignore[assignment]
        if orig_cuda_autocast is not None:
            torch.cuda.amp.autocast = _no_autocast  # type: ignore[assignment]
        try:
            yield
        finally:
            if orig_autocast is not None:
                torch.autocast = orig_autocast  # type: ignore[assignment]
            if orig_cuda_autocast is not None:
                torch.cuda.amp.autocast = orig_cuda_autocast  # type: ignore[assignment]

    def _get_predictor(self, desired_dtype: torch.dtype | None = None):
        if self._predictor is not None:
            if desired_dtype is not None and self._predictor_dtype != desired_dtype:
                self._predictor = None
                self._predictor_dtype = None
            else:
                return self._predictor

        if self._predictor is not None:
            return self._predictor

        with self._lock:
            if self._predictor is not None:
                if desired_dtype is not None and self._predictor_dtype != desired_dtype:
                    self._predictor = None
                    self._predictor_dtype = None
                else:
                    return self._predictor
            if self._predictor is not None:
                return self._predictor

            try:
                from sam3.model_builder import build_sam3_video_predictor
            except Exception as exc:  # pragma: no cover - dependency error
                raise RuntimeError(
                    f"SAM3 import failed: {exc}. Ensure sam3 and its dependencies are installed."
                ) from exc

            prev_dtype = torch.get_default_dtype()
            if desired_dtype is not None:
                torch.set_default_dtype(desired_dtype)
            self._predictor = build_sam3_video_predictor()
            self._predictor_dtype = desired_dtype or prev_dtype
            torch.set_default_dtype(prev_dtype)
            try:
                self._disable_sam3_autocast()
            except Exception:  # pragma: no cover - best effort
                logger.exception("SAM3 autocast patch failed")
            return self._predictor

    def _mask_path(self, session_dir: Path, frame_index: int) -> Path:
        return session_dir / f"{frame_index:06d}.png"

    def _resample_video(
        self, input_path: Path, output_path: Path, target_fps: float | None
    ) -> tuple[Path, float | None]:
        if cv2 is None:
            return input_path, None
        if not target_fps or target_fps <= 0:
            return input_path, None

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            cap.release()
            return input_path, None

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            return input_path, None

        if src_fps > 0 and abs(src_fps - target_fps) < 0.01:
            cap.release()
            return input_path, src_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            writer.release()
            return input_path, None

        frame_index = 0
        next_time = 0.0
        src_fps = src_fps if src_fps > 0 else target_fps
        src_frame_time = 1.0 / src_fps if src_fps > 0 else 0.0
        target_frame_time = 1.0 / target_fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_index * src_frame_time
            if current_time + 1e-6 >= next_time:
                writer.write(frame)
                next_time += target_frame_time
            frame_index += 1

        cap.release()
        writer.release()
        return output_path, target_fps

    def generate_masks(
        self,
        video_base64: str,
        prompt: str,
        box: list[int] | None = None,
        input_fps: float | None = None,
    ) -> Sam3MaskSession:
        prompt = SAM3_PERSON_PROMPT
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        predictor = self._get_predictor(desired_dtype=torch.float32)
        try:
            self._force_predictor_float32(predictor)
        except Exception:  # pragma: no cover - best effort
            logger.exception("SAM3 float32 cast failed")
        session_id = str(uuid.uuid4())
        session_dir = self._masks_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        video_path = session_dir / "source.mp4"
        video_path.write_bytes(base64.b64decode(video_base64))

        resampled_path = session_dir / "source_resampled.mp4"
        source_fps = None
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

        effective_fps = None
        if source_fps and source_fps > 0:
            effective_fps = min(source_fps, SAM3_MAX_FPS)
        elif input_fps and input_fps > 0:
            effective_fps = input_fps

        video_path, forced_fps = self._resample_video(
            video_path, resampled_path, effective_fps
        )

        response = None
        with self._disable_autocast():
            response = predictor.handle_request(
                {"type": "start_session", "resource_path": str(video_path)}
            )
        if not isinstance(response, dict) or "session_id" not in response:
            logger.exception("SAM3 start_session returned invalid response: %s", response)
            raise RuntimeError(
                f"SAM3 start_session failed: {response}"
            )
        sam3_session_id = response["session_id"]

        frame_count_guess = None
        sam3_fps = forced_fps
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS)) if sam3_fps is None else sam3_fps
                if count > 0:
                    frame_count_guess = count
                if fps > 0:
                    sam3_fps = fps
            cap.release()

        def send_prompt(frame_index: int) -> None:
            prompt_payload = {
                "type": "add_prompt",
                "session_id": sam3_session_id,
                "frame_index": frame_index,
            }
            prompt_payload["text"] = prompt
            predictor.handle_request(prompt_payload)

        if frame_count_guess:
            for frame_index in range(0, frame_count_guess, SAM3_PROMPT_STRIDE):
                send_prompt(frame_index)
        else:
            send_prompt(0)

        frame_count = 0
        height = 0
        width = 0

        try:
            with self._disable_autocast():
                for result in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": sam3_session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": None,
                    }
                ):
                    frame_idx = result["frame_index"]
                    masks = result["outputs"].get("out_binary_masks")
                    if masks is None:
                        continue
                    if isinstance(masks, np.ndarray):
                        if masks.size == 0:
                            continue
                        merged = np.any(masks, axis=0).astype(np.uint8) * 255
                    else:
                        if masks.numel() == 0:
                            continue
                        merged = masks.any(dim=0).cpu().numpy().astype(np.uint8) * 255
                    if cv2 is not None:
                        mask = merged
                        if SAM3_MASK_DILATE > 0:
                            kernel = np.ones(
                                (SAM3_MASK_DILATE, SAM3_MASK_DILATE), dtype=np.uint8
                            )
                            mask = cv2.dilate(mask, kernel, iterations=SAM3_MASK_DILATE_ITERS)
                        if SAM3_MASK_BLUR > 0:
                            blur_size = SAM3_MASK_BLUR
                            if blur_size % 2 == 0:
                                blur_size += 1
                            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                        if SAM3_MASK_INTENSITY != 1.0:
                            mask = np.clip(mask.astype(np.float32) * SAM3_MASK_INTENSITY, 0, 255)
                            mask = mask.astype(np.uint8)
                        merged = mask
                    height, width = merged.shape
                    frame_count = max(frame_count, frame_idx + 1)
                    Image.fromarray(merged, mode="L").save(self._mask_path(session_dir, frame_idx))
        except Exception:
            logger.exception("SAM3 propagate_in_video failed")
            raise
        finally:
            torch.set_default_dtype(prev_dtype)

        with self._disable_autocast():
            predictor.handle_request({"type": "close_session", "session_id": sam3_session_id})

        if frame_count == 0:
            raise RuntimeError("SAM3 returned no masks for the provided prompt.")

        effective_input_fps = effective_fps or sam3_fps
        session = Sam3MaskSession(
            session_id=session_id,
            mask_dir=session_dir,
            video_path=video_path,
            height=height,
            width=width,
            frame_count=frame_count,
            prompt=prompt,
            input_fps=effective_input_fps,
            sam3_fps=sam3_fps,
        )
        self._sessions[session_id] = session
        return session

    def get_masks_for_frames(
        self, session_id: str, frame_indices: Iterable[int]
    ) -> list[torch.Tensor]:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Mask session {session_id} not found")

        frames = []
        hits = 0
        misses = 0
        frame_indices_list = list(frame_indices)
        for frame_idx in frame_indices_list:
            if session.frame_count > 0:
                input_fps = session.input_fps or 15.0
                sam3_fps = session.sam3_fps or input_fps
                if (
                    input_fps > 0
                    and sam3_fps > 0
                    and abs(input_fps - sam3_fps) < 0.01
                ):
                    mask_frame_idx = frame_idx % session.frame_count
                else:
                    if input_fps <= 0:
                        input_fps = sam3_fps or 15.0
                    time_sec = frame_idx / input_fps
                    mask_frame_idx = int(round(time_sec * sam3_fps)) % session.frame_count
            else:
                mask_frame_idx = frame_idx

            mask_path = self._mask_path(session.mask_dir, mask_frame_idx)
            if not mask_path.exists():
                blank = np.zeros((session.height, session.width), dtype=np.uint8)
                mask_array = blank
                misses += 1
            else:
                mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                hits += 1

            tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(-1)
            frames.append(tensor)
        if SAM3_DEBUG:
            first_idx = frame_indices_list[0] if frame_indices_list else None
            last_idx = frame_indices_list[-1] if frame_indices_list else None
            logger.info(
                "SAM3 mask fetch: session=%s frames=%d hits=%d misses=%d first=%s last=%s",
                session_id,
                len(frames),
                hits,
                misses,
                first_idx,
                last_idx,
            )
        return frames

    def get_masks_for_indices(
        self, session_id: str, frame_indices: Iterable[int]
    ) -> list[torch.Tensor]:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Mask session {session_id} not found")

        frames = []
        hits = 0
        misses = 0
        for frame_idx in frame_indices:
            mask_frame_idx = (
                frame_idx % session.frame_count if session.frame_count > 0 else frame_idx
            )
            mask_path = self._mask_path(session.mask_dir, mask_frame_idx)
            if not mask_path.exists():
                blank = np.zeros((session.height, session.width), dtype=np.uint8)
                mask_array = blank
                misses += 1
            else:
                mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                hits += 1

            tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(-1)
            frames.append(tensor)

        if SAM3_DEBUG:
            logger.info(
                "SAM3 mask fetch (index): session=%s frames=%d hits=%d misses=%d",
                session_id,
                len(frames),
                hits,
                misses,
            )
        return frames

    def get_session(self, session_id: str) -> Sam3MaskSession | None:
        return self._sessions.get(session_id)

    def reset_applied_indices(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        with self._applied_lock:
            session.applied_mask_indices.clear()

    def append_applied_indices(self, session_id: str, indices: list[int]) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        with self._applied_lock:
            session.applied_mask_indices.extend(indices)

    def get_applied_indices(self, session_id: str) -> list[int]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        with self._applied_lock:
            return list(session.applied_mask_indices)

    def get_mask_indices(
        self,
        session_id: str,
        frame_indices: list[int],
        frame_times: list[float | None] | None,
        use_server_video: bool,
        use_time_mapping: bool,
    ) -> list[int]:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Mask session {session_id} not found")

        if use_server_video:
            if session.frame_count > 0:
                return [idx % session.frame_count for idx in frame_indices]
            return list(frame_indices)

        input_fps = session.input_fps or 15.0
        sam3_fps = session.sam3_fps or input_fps
        if sam3_fps <= 0:
            sam3_fps = input_fps if input_fps > 0 else 15.0

        if use_time_mapping and frame_times is not None and any(
            time_val is not None for time_val in frame_times
        ):
            time_origin = None
            for time_val in frame_times:
                if time_val is not None:
                    time_origin = time_val
                    break
            indices: list[int] = []
            for idx, time_val in enumerate(frame_times):
                fallback_idx = frame_indices[idx] if idx < len(frame_indices) else idx
                if time_val is None:
                    mask_idx = fallback_idx
                else:
                    normalized_time = time_val - (time_origin or 0.0)
                    if normalized_time < 0:
                        normalized_time = 0.0
                    mask_idx = int(round(normalized_time * sam3_fps))
                if session.frame_count > 0:
                    mask_idx = mask_idx % session.frame_count
                indices.append(mask_idx)
            return indices

        if input_fps > 0 and sam3_fps > 0 and abs(input_fps - sam3_fps) < 0.01:
            if session.frame_count > 0:
                return [idx % session.frame_count for idx in frame_indices]
            return list(frame_indices)

        indices = []
        for idx in frame_indices:
            time_sec = idx / input_fps if input_fps > 0 else 0.0
            mask_idx = int(round(time_sec * sam3_fps))
            if session.frame_count > 0:
                mask_idx = mask_idx % session.frame_count
            indices.append(mask_idx)
        return indices

    def get_masks_for_times(
        self,
        session_id: str,
        frame_times: Iterable[float | None],
        fallback_indices: Iterable[int],
    ) -> list[torch.Tensor]:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Mask session {session_id} not found")

        input_fps = session.input_fps or 15.0
        sam3_fps = session.sam3_fps or input_fps
        if sam3_fps <= 0:
            sam3_fps = input_fps if input_fps > 0 else 15.0

        frames = []
        hits = 0
        misses = 0
        frame_times_list = list(frame_times)
        fallback_indices_list = list(fallback_indices)

        time_origin = None
        for time_sec in frame_times_list:
            if time_sec is not None:
                time_origin = time_sec
                break

        for idx, time_sec in enumerate(frame_times_list):
            fallback_idx = (
                fallback_indices_list[idx]
                if idx < len(fallback_indices_list)
                else idx
            )
            if time_sec is None:
                if session.frame_count > 0:
                    mask_frame_idx = fallback_idx % session.frame_count
                else:
                    mask_frame_idx = fallback_idx
            else:
                normalized_time = time_sec - (time_origin or 0.0)
                if normalized_time < 0:
                    normalized_time = 0.0
                mask_frame_idx = int(round(normalized_time * sam3_fps))
                if session.frame_count > 0:
                    mask_frame_idx = mask_frame_idx % session.frame_count

            mask_path = self._mask_path(session.mask_dir, mask_frame_idx)
            if not mask_path.exists():
                blank = np.zeros((session.height, session.width), dtype=np.uint8)
                mask_array = blank
                misses += 1
            else:
                mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                hits += 1

            tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(-1)
            frames.append(tensor)

        first_time = frame_times_list[0] if frame_times_list else None
        last_time = frame_times_list[-1] if frame_times_list else None
        if SAM3_DEBUG:
            logger.info(
                "SAM3 mask fetch (time): session=%s frames=%d hits=%d misses=%d sam3_fps=%s origin=%s first_time=%s last_time=%s",
                session_id,
                len(frames),
                hits,
                misses,
                sam3_fps,
                time_origin,
                first_time,
                last_time,
            )

        return frames


sam3_mask_manager = Sam3MaskManager()
