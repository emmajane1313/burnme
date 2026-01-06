import asyncio
import fractions
import logging
import os
import threading
import time
from pathlib import Path

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager
from .sam3_manager import sam3_mask_manager

logger = logging.getLogger(__name__)
DEBUG_ALL = os.getenv("BURN_DEBUG_ALL") == "1"
FRAME_DEBUG = os.getenv("BURN_DEBUG_FRAMES") == "1"


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = None
        self.input_task = None
        self.input_task_running = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None
        self._input_frame_index = 0
        self._server_video_enabled = False
        self._server_video_loop = True
        self._server_video_reset = threading.Event()
        self._server_video_stop = threading.Event()
        self._server_video_thread: threading.Thread | None = None
        self._server_video_fps = None
        self._server_video_path: Path | None = None

        # Spout input mode - when enabled, frames come from Spout instead of WebRTC
        self._spout_receiver_enabled = False
        if initial_parameters:
            spout_receiver = initial_parameters.get("spout_receiver")
            if spout_receiver and spout_receiver.get("enabled"):
                self._spout_receiver_enabled = True
                logger.info("Spout input mode enabled")

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()

                # Store raw VideoFrame for later processing (tracks input FPS internally)
                self.frame_processor.put(input_frame)
                if (FRAME_DEBUG or DEBUG_ALL) and self._input_frame_index % 30 == 0:
                    logger.info(
                        "VideoProcessingTrack input: index=%s pts=%s",
                        self._input_frame_index,
                        getattr(input_frame, "pts", None),
                    )
                self._input_frame_index += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Stop the input loop on connection errors to avoid spam
                logger.error(f"Error in input loop, stopping: {e}")
                self.input_task_running = False
                break

    # Copied from https://github.com/livepeer/fastworld/blob/e649ef788cd33d78af6d8e1da915cd933761535e/backend/track.py#L267
    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override to control frame rate"""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "timestamp"):
            # Calculate wait time based on current frame rate
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time

            # Wait for the appropriate interval based on current FPS
            target_interval = self.frame_ptime  # Current frame period
            wait_time = target_interval - time_since_last_frame

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Update timestamp and last frame time
            self.timestamp += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            self.last_frame_time = time.time()
        else:
            self.start = time.time()
            self.last_frame_time = time.time()
            self.timestamp = 0

        return self.timestamp, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        if not self.frame_processor:
            self.frame_processor = FrameProcessor(
                pipeline_manager=self.pipeline_manager,
                initial_parameters=self.initial_parameters,
                notification_callback=self.notification_callback,
            )
            self.frame_processor.start()

    def initialize_input_processing(self, track: MediaStreamTrack):
        if self._server_video_enabled:
            logger.info("Server video source enabled; ignoring incoming track.")
            return
        self.track = track
        self.input_task_running = True
        self.input_task = asyncio.create_task(self.input_loop())

    def initialize_server_video(self, mask_id: str, loop: bool = True):
        session = sam3_mask_manager.get_session(mask_id)
        if session is None:
            raise RuntimeError(f"SAM3 session {mask_id} not found")
        self._server_video_path = session.video_path
        self._server_video_fps = session.input_fps or session.sam3_fps or 15.0
        self._server_video_loop = loop
        self._server_video_enabled = True
        self.input_task_running = True
        self.initialize_output_processing()
        self._server_video_stop.clear()
        self._server_video_reset.clear()
        self._server_video_thread = threading.Thread(
            target=self._server_video_loop_fn,
            name="server-video-input",
            daemon=True,
        )
        self._server_video_thread.start()
        logger.info(
            "Server video input started: path=%s fps=%s loop=%s",
            self._server_video_path,
            self._server_video_fps,
            self._server_video_loop,
        )

    def set_server_video_loop(self, loop: bool):
        self._server_video_loop = loop

    def reset_server_video(self):
        logger.info("Server video reset requested")
        self._server_video_reset.set()
        if self.frame_processor:
            self.frame_processor.reset_buffers("server_video_reset")

    def _server_video_loop_fn(self):
        try:
            import cv2  # type: ignore
        except Exception:
            logger.error("OpenCV not available; server video input disabled.")
            return

        if not self._server_video_path:
            return

        cap = cv2.VideoCapture(str(self._server_video_path))
        if not cap.isOpened():
            logger.error("Failed to open server video: %s", self._server_video_path)
            return

        fps = self._server_video_fps or float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        frame_period = 1.0 / fps if fps > 0 else 1.0 / 15.0
        frame_idx = 0
        next_time = time.time()

        while not self._server_video_stop.is_set():
            if self._server_video_reset.is_set():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                self._server_video_reset.clear()
                next_time = time.time()
                if self.notification_callback:
                    self.notification_callback({"type": "server_video_reset_done"})

            ret, frame = cap.read()
            if not ret:
                if self._server_video_loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                if self.notification_callback:
                    self.notification_callback({"type": "server_video_ended"})
                self.input_task_running = False
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            pts = int(frame_idx * frame_period * VIDEO_CLOCK_RATE)
            video_frame.pts = pts
            video_frame.time_base = VIDEO_TIME_BASE
            if self.frame_processor:
                self.frame_processor.put(video_frame)

            frame_idx += 1
            next_time += frame_period
            sleep_for = next_time - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()

        # Keep running while either WebRTC input is active OR Spout input is enabled
        while self.input_task_running or self._spout_receiver_enabled:
            try:
                # Update FPS: use minimum of input FPS and pipeline FPS
                if self.frame_processor:
                    self.fps = self.frame_processor.get_output_fps()
                    self.frame_ptime = 1.0 / self.fps

                # If paused, wait for the appropriate frame interval before returning
                with self._paused_lock:
                    paused = self._paused

                frame = None
                if paused:
                    # When video is paused, return the last frame to freeze the playback video
                    frame = self._last_frame
                else:
                    # When video is not paused, get the next frame from the frame processor
                    frame_tensor = self.frame_processor.get()
                    if frame_tensor is not None:
                        frame = VideoFrame.from_ndarray(
                            frame_tensor.numpy(), format="rgb24"
                        )

                if frame is not None:
                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    self._last_frame = frame
                    return frame

                # No frame available, wait a bit before trying again
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    def pause(self, paused: bool):
        """Pause or resume the video track processing"""
        with self._paused_lock:
            self._paused = paused
        logger.info(f"Video track {'paused' if paused else 'resumed'}")

    async def stop_async(self):
        self.input_task_running = False
        self._spout_receiver_enabled = False
        if self._server_video_enabled:
            self._server_video_stop.set()
            if self._server_video_thread:
                self._server_video_thread.join(timeout=1.0)

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        await super().stop()

    def stop(self):
        """Synchronous stop for aiortc compatibility."""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(self.stop_async())
        else:
            asyncio.run(self.stop_async())
