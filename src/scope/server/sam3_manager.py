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


@dataclass
class Sam3MaskSession:
    session_id: str
    mask_dir: Path
    height: int
    width: int
    frame_count: int
    prompt: str


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

    def generate_masks(self, video_base64: str, prompt: str) -> Sam3MaskSession:
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

        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": sam3_session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )

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
                mask_frame_idx = frame_idx % session.frame_count
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


sam3_mask_manager = Sam3MaskManager()
