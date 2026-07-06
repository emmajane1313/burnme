import logging
import os
import threading
from pathlib import Path
from typing import Iterable

import httpx
import numpy as np

from .models_config import get_assets_dir

logger = logging.getLogger(__name__)


class RealtimePersonMasker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._model_id = os.getenv("BURN_MASK_MODEL", "yolov8s-seg.pt")
        self._model_url = os.getenv("BURN_MASK_MODEL_URL")
        self._device = os.getenv("BURN_MASK_DEVICE", "cuda")
        self._imgsz = int(os.getenv("BURN_MASK_IMGSZ", "640"))

    def _resolve_model_path(self) -> Path:
        model_path = Path(self._model_id)
        if model_path.is_absolute():
            return model_path
        mask_dir = os.getenv("BURN_MASK_DIR")
        if mask_dir:
            base_dir = Path(mask_dir)
        else:
            base_dir = get_assets_dir() / "yolo_masks"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / self._model_id

    def _download_model(self, target_path: Path) -> None:
        if not self._model_url:
            raise RuntimeError(
                "Missing BURN_MASK_MODEL_URL for realtime mask download."
            )
        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        logger.info("Downloading realtime mask model: %s", self._model_url)
        with httpx.stream("GET", self._model_url, timeout=300.0) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(target_path)
        logger.info("Realtime mask model saved: %s", target_path)

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Ultralytics is required for realtime masks. "
                    "Install 'ultralytics' and provide a segmentation model."
                ) from exc

            model_source = self._model_id
            model_path = self._resolve_model_path()
            if model_path.exists():
                model_source = str(model_path)
            elif self._model_url:
                self._download_model(model_path)
                model_source = str(model_path)

            self._model = YOLO(model_source)
            return self._model

    def generate_masks(self, frames: Iterable[np.ndarray]) -> list[np.ndarray]:
        model = self._load_model()
        frame_list = list(frames)
        if not frame_list:
            return []
        results = model.predict(
            frame_list,
            verbose=False,
            device=self._device,
            imgsz=self._imgsz,
        )
        masks: list[np.ndarray] = []
        for frame, result in zip(frame_list, results, strict=False):
            height, width = frame.shape[:2]
            if result.masks is None or result.boxes is None:
                masks.append(np.zeros((height, width), dtype=np.uint8))
                continue
            cls = result.boxes.cls
            if cls is None or len(cls) == 0:
                masks.append(np.zeros((height, width), dtype=np.uint8))
                continue
            person_mask = None
            for idx, class_id in enumerate(cls.tolist()):
                if int(class_id) != 0:
                    continue
                mask_data = result.masks.data[idx].cpu().numpy()
                if mask_data.ndim == 3:
                    mask_data = mask_data[0]
                if mask_data.shape != (height, width):
                    try:
                        import cv2  # type: ignore
                    except Exception as exc:
                        raise RuntimeError(
                            "OpenCV is required to resize YOLO masks."
                        ) from exc
                    mask_data = cv2.resize(
                        mask_data,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_bin = (mask_data > 0.5).astype(np.uint8) * 255
                if person_mask is None:
                    person_mask = mask_bin
                else:
                    person_mask = np.maximum(person_mask, mask_bin)
            if person_mask is None:
                person_mask = np.zeros((height, width), dtype=np.uint8)
            masks.append(person_mask)
        return masks


realtime_person_masker = RealtimePersonMasker()
