import base64
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .models_config import get_assets_dir

logger = logging.getLogger(__name__)
SAM3_DEBUG = os.getenv("BURN_DEBUG_SAM3") == "1"
SAM3_MASK_DILATE = int(os.getenv("BURN_SAM3_MASK_DILATE", "7"))
SAM3_MASK_DILATE_ITERS = int(os.getenv("BURN_SAM3_MASK_DILATE_ITERS", "2"))
SAM3_MASK_BLUR = int(os.getenv("BURN_SAM3_MASK_BLUR", "5"))
SAM3_MASK_INTENSITY = float(os.getenv("BURN_SAM3_MASK_INTENSITY", "1.0"))
SAM3_PROMPT_STRIDE = 1
SAM3_PERSON_PROMPT = "person, human, face, body, hands, arms, legs"

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass
class Sam3MaskSession:
    session_id: str
    mask_dir: Path
    height: int
    width: int
    frame_count: int
    prompt: str
    input_fps: float | None
    sam3_fps: float | None


class Sam3MaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._predictor = None
        self._sessions: dict[str, Sam3MaskSession] = {}

        assets_dir = get_assets_dir()
        self._masks_dir = assets_dir / "sam3_masks"
        self._masks_dir.mkdir(parents=True, exist_ok=True)

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        with self._lock:
            if self._predictor is not None:
                return self._predictor

            try:
                from sam3.model_builder import build_sam3_video_predictor
            except Exception as exc:  # pragma: no cover - dependency error
                raise RuntimeError(
                    f"SAM3 import failed: {exc}. Ensure sam3 and its dependencies are installed."
                ) from exc

            self._predictor = build_sam3_video_predictor()
            return self._predictor

    def _mask_path(self, session_dir: Path, frame_index: int) -> Path:
        return session_dir / f"{frame_index:06d}.png"

    def generate_masks(
        self,
        video_base64: str,
        prompt: str,
        box: list[int] | None = None,
        input_fps: float | None = None,
    ) -> Sam3MaskSession:
        prompt = SAM3_PERSON_PROMPT

        predictor = self._get_predictor()
        session_id = str(uuid.uuid4())
        session_dir = self._masks_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        video_path = session_dir / "source.mp4"
        video_path.write_bytes(base64.b64decode(video_base64))

        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(video_path)}
        )
        sam3_session_id = response["session_id"]

        frame_count_guess = None
        sam3_fps = None
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
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

        predictor.handle_request({"type": "close_session", "session_id": sam3_session_id})

        if frame_count == 0:
            raise RuntimeError("SAM3 returned no masks for the provided prompt.")

        session = Sam3MaskSession(
            session_id=session_id,
            mask_dir=session_dir,
            height=height,
            width=width,
            frame_count=frame_count,
            prompt=prompt,
            input_fps=input_fps,
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

        if SAM3_DEBUG:
            logger.info(
                "SAM3 mask fetch (time): session=%s frames=%d hits=%d misses=%d sam3_fps=%s",
                session_id,
                len(frames),
                hits,
                misses,
                sam3_fps,
            )

        return frames


sam3_mask_manager = Sam3MaskManager()
