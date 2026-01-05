import asyncio
import os
import base64
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import tempfile
import warnings
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
from functools import wraps
from importlib.metadata import version
from logging.handlers import RotatingFileHandler
from pathlib import Path

import click
import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel

from .download_models import download_models
from .download_progress_manager import download_progress_manager
from .logs_config import (
    cleanup_old_logs,
    ensure_logs_dir,
    get_current_log_file,
    get_logs_dir,
    get_most_recent_log_file,
)
from .models_config import (
    ensure_models_dir,
    get_assets_dir,
    get_models_dir,
    models_are_downloaded,
)
from .pipeline_manager import PipelineManager
from .sam3_manager import sam3_mask_manager
from .schema import (
    AssetFileInfo,
    AssetsResponse,
    HardwareInfoResponse,
    HealthResponse,
    IceCandidateRequest,
    IceServerConfig,
    IceServersResponse,
    PipelineLoadRequest,
    PipelineSchemasResponse,
    PipelineStatusResponse,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
)
from .webrtc import WebRTCManager

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class STUNErrorFilter(logging.Filter):
    """Filter to suppress STUN/TURN connection errors that are not critical."""

    def filter(self, record):
        # Suppress STUN  exeception that occurrs always during the stream restart
        if "Task exception was never retrieved" in record.getMessage():
            return False
        return True


# Ensure logs directory exists and clean up old logs
logs_dir = ensure_logs_dir()
cleanup_old_logs(max_age_days=1)  # Delete logs older than 1 day
log_file = get_current_log_file()

# Configure logging - set root to WARNING to keep non-app libraries quiet by default
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler handles INFO
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, RotatingFileHandler
    ):
        handler.setLevel(logging.INFO)

# Add rotating file handler
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=5,  # Keep 5 backup files
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Add the filter to suppress STUN/TURN errors
stun_filter = STUNErrorFilter()
logging.getLogger("asyncio").addFilter(stun_filter)

# Set INFO level for your app modules
logging.getLogger("scope.server").setLevel(logging.INFO)
logging.getLogger("scope.core").setLevel(logging.INFO)

# Optional debug override for full logs
if os.getenv("BURN_DEBUG_ALL") == "1":
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers:
        handler.setLevel(logging.DEBUG)
    logging.getLogger("scope.server").setLevel(logging.DEBUG)
    logging.getLogger("scope.core").setLevel(logging.DEBUG)

# Set INFO level for uvicorn
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

# Enable verbose logging for other libraries when needed
if os.getenv("VERBOSE_LOGGING"):
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("aiortc").setLevel(logging.INFO)

# Select pipeline depending on the "PIPELINE" environment variable
PIPELINE = os.getenv("PIPELINE", None)

logger = logging.getLogger(__name__)


def suppress_init_output(func):
    """Decorator to suppress all initialization output (logging, warnings, stdout/stderr)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            # Temporarily disable all logging
            logging.disable(logging.CRITICAL)
            try:
                return func(*args, **kwargs)
            finally:
                # Re-enable logging
                logging.disable(logging.NOTSET)

    return wrapper


def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Git commit hash if available, otherwise a fallback message.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
            cwd=Path(__file__).parent,  # Run in the project directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown (not a git repository)"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown (git error)"
    except FileNotFoundError:
        return "unknown (git not installed)"
    except Exception:
        return "unknown"


def print_version_info():
    """Print version information and exit."""
    try:
        pkg_version = version("burnmewhileimhot")
    except Exception:
        pkg_version = "unknown"

    git_hash = get_git_commit_hash()

    print(f"burnmewhileimhot: {pkg_version}")
    print(f"git commit: {git_hash}")


def configure_static_files():
    """Configure static file serving for production."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )
        logger.info(f"Serving static assets from {frontend_dist / 'assets'}")
    else:
        logger.info("Frontend dist directory not found - running in development mode")


# Global WebRTC manager instance
webrtc_manager = None
# Global pipeline manager instance
pipeline_manager = None


async def prewarm_pipeline(pipeline_id: str):
    """Background task to pre-warm the pipeline without blocking startup."""
    try:
        await asyncio.wait_for(
            pipeline_manager.load_pipeline(pipeline_id),
            timeout=300,  # 5 minute timeout for pipeline loading
        )
    except Exception as e:
        logger.error(f"Error pre-warming pipeline {pipeline_id} in background: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Startup
    global webrtc_manager, pipeline_manager

    # Check CUDA availability and warn if not available
    if not torch.cuda.is_available():
        warning_msg = (
            "CUDA is not available on this system. "
            "Some pipelines may not work without a CUDA-compatible GPU. "
            "The application will start, but pipeline functionality may be limited."
        )
        logger.warning(warning_msg)

    # Log logs directory
    logs_dir = get_logs_dir()
    logger.info(f"Logs directory: {logs_dir}")

    # Ensure models directory and subdirectories exist
    models_dir = ensure_models_dir()
    logger.info(f"Models directory: {models_dir}")

    # Ensure assets directory exists for VACE reference images and other media (at same level as models)
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Assets directory: {assets_dir}")

    # Initialize pipeline manager (but don't load pipeline yet)
    pipeline_manager = PipelineManager()
    logger.info("Pipeline manager initialized")

    # Pre-warm the default pipeline
    if PIPELINE is not None:
        asyncio.create_task(prewarm_pipeline(PIPELINE))

    webrtc_manager = WebRTCManager()
    logger.info("WebRTC manager initialized")

    yield

    # Shutdown
    if webrtc_manager:
        logger.info("Shutting down WebRTC manager...")
        await webrtc_manager.stop()
        logger.info("WebRTC manager shutdown complete")

    if pipeline_manager:
        logger.info("Shutting down pipeline manager...")
        pipeline_manager.unload_pipeline()
        logger.info("Pipeline manager shutdown complete")


def get_webrtc_manager() -> WebRTCManager:
    """Dependency to get WebRTC manager instance."""
    return webrtc_manager


def get_pipeline_manager() -> PipelineManager:
    """Dependency to get pipeline manager instance."""
    return pipeline_manager


app = FastAPI(
    lifespan=lifespan,
    title="Scope",
    description="A tool for running and customizing real-time, interactive generative AI pipelines and models",
    version=version("burnmewhileimhot"),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())


@app.get("/")
async def root():
    """Serve the frontend at the root URL."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        return {"message": "Scope API - Frontend not built"}

    # Serve the frontend index.html with no-cache headers
    # This ensures clients like Electron alway fetch the latest HTML (which references hashed assets)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return {"message": "Scope API - Frontend index.html not found"}


@app.post("/api/v1/pipeline/load")
async def load_pipeline(
    request: PipelineLoadRequest,
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Load a pipeline."""
    try:
        # Convert pydantic model to dict for pipeline manager
        load_params_dict = None
        if request.load_params:
            load_params_dict = request.load_params.model_dump()

        # Start loading in background without blocking
        asyncio.create_task(
            pipeline_manager.load_pipeline(request.pipeline_id, load_params_dict)
        )
        return {"message": "Pipeline loading initiated successfully"}
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Get current pipeline status."""
    try:
        status_info = await pipeline_manager.get_status_info_async()
        return PipelineStatusResponse(**status_info)
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipelines/schemas", response_model=PipelineSchemasResponse)
async def get_pipeline_schemas():
    """Get configuration schemas and defaults for all available pipelines.

    Returns the output of each pipeline's get_schema_with_metadata() method,
    which includes:
    - Pipeline metadata (id, name, description, version)
    - supported_modes: List of supported input modes ("text", "video")
    - default_mode: Default input mode for this pipeline
    - mode_defaults: Mode-specific default overrides (if any)
    - config_schema: Full JSON schema with defaults

    The frontend should use this as the source of truth for parameter defaults.
    """
    from scope.core.pipelines.registry import PipelineRegistry

    pipelines: dict = {}

    for pipeline_id in PipelineRegistry.list_pipelines():
        config_class = PipelineRegistry.get_config_class(pipeline_id)
        if config_class:
            # get_schema_with_metadata() includes supported_modes, default_mode,
            # and mode_defaults directly from the config class
            schema_data = config_class.get_schema_with_metadata()
            pipelines[pipeline_id] = schema_data

    return PipelineSchemasResponse(pipelines=pipelines)


@app.get("/api/v1/webrtc/ice-servers", response_model=IceServersResponse)
async def get_ice_servers(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Return ICE server configuration for frontend WebRTC connection."""
    ice_servers = []

    for server in webrtc_manager.rtc_config.iceServers:
        ice_servers.append(
            IceServerConfig(
                urls=server.urls,
                username=server.username if hasattr(server, "username") else None,
                credential=server.credential if hasattr(server, "credential") else None,
            )
        )

    return IceServersResponse(iceServers=ice_servers)


@app.post("/api/v1/webrtc/offer", response_model=WebRTCOfferResponse)
async def handle_webrtc_offer(
    request: WebRTCOfferRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Handle WebRTC offer and return answer."""
    try:
        # Ensure pipeline is loaded before proceeding
        status_info = await pipeline_manager.get_status_info_async()
        if status_info["status"] != "loaded":
            raise HTTPException(
                status_code=400,
                detail="Pipeline not loaded. Please load pipeline first.",
            )

        return await webrtc_manager.handle_offer(request, pipeline_manager)

    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.patch(
    "/api/v1/webrtc/offer/{session_id}", status_code=204, response_class=Response
)
async def add_ice_candidate(
    session_id: str,
    candidate_request: IceCandidateRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Add ICE candidate(s) to an existing WebRTC session (Trickle ICE).

    This endpoint follows the Trickle ICE pattern, allowing clients to send
    ICE candidates as they are discovered.
    """
    # TODO: Validate that the Content-Type is 'application/trickle-ice-sdpfrag'
    # At the moment FastAPI defaults to validating that it is 'application/json'
    try:
        for candidate_init in candidate_request.candidates:
            await webrtc_manager.add_ice_candidate(
                session_id=session_id,
                candidate=candidate_init.candidate,
                sdp_mid=candidate_init.sdpMid,
                sdp_mline_index=candidate_init.sdpMLineIndex,
            )

            logger.debug(
                f"Added {len(candidate_request.candidates)} ICE candidates to session {session_id}"
            )

        # Return 204 No Content on success
        return Response(status_code=204)

    except ValueError as e:
        # Session not found or invalid candidate
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error adding ICE candidate to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ModelStatusResponse(BaseModel):
    downloaded: bool


class DownloadModelsRequest(BaseModel):
    pipeline_id: str


class LoRAFileInfo(BaseModel):
    """Metadata for an available LoRA file on disk."""

    name: str
    path: str
    size_mb: float
    folder: str | None = None


class LoRAFilesResponse(BaseModel):
    """Response containing all discoverable LoRA files."""

    lora_files: list[LoRAFileInfo]


@app.get("/api/v1/lora/list", response_model=LoRAFilesResponse)
async def list_lora_files():
    """List available LoRA files in the models/lora directory and its subdirectories."""

    def process_lora_file(file_path: Path, lora_dir: Path) -> LoRAFileInfo:
        """Extract LoRA file metadata."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        relative_path = file_path.relative_to(lora_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )
        return LoRAFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
        )

    try:
        lora_dir = get_models_dir() / "lora"
        lora_files: list[LoRAFileInfo] = []

        if lora_dir.exists() and lora_dir.is_dir():
            for pattern in ("*.safetensors", "*.bin", "*.pt"):
                for file_path in lora_dir.rglob(pattern):
                    if file_path.is_file():
                        lora_files.append(process_lora_file(file_path, lora_dir))

        lora_files.sort(key=lambda x: (x.folder or "", x.name))
        return LoRAFilesResponse(lora_files=lora_files)

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"list_lora_files: Error listing LoRA files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/assets", response_model=AssetsResponse)
async def list_assets(
    type: str | None = Query(None, description="Filter by asset type (image, video)"),
):
    """List available asset files in the assets directory and its subdirectories."""

    def process_asset_file(
        file_path: Path, assets_dir: Path, asset_type: str
    ) -> AssetFileInfo:
        """Extract asset file metadata."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        created_at = file_path.stat().st_ctime
        relative_path = file_path.relative_to(assets_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )
        return AssetFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
            type=asset_type,
            created_at=created_at,
        )

    try:
        assets_dir = get_assets_dir()
        asset_files: list[AssetFileInfo] = []

        if assets_dir.exists() and assets_dir.is_dir():
            # Define patterns based on type filter
            if type == "image" or type is None:
                image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
                for pattern in image_patterns:
                    for file_path in assets_dir.rglob(pattern):
                        if file_path.is_file():
                            asset_files.append(
                                process_asset_file(file_path, assets_dir, "image")
                            )

            if type == "video" or type is None:
                video_patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
                for pattern in video_patterns:
                    for file_path in assets_dir.rglob(pattern):
                        if file_path.is_file():
                            asset_files.append(
                                process_asset_file(file_path, assets_dir, "video")
                            )

        # Sort by created_at (most recent first), then by folder and name
        asset_files.sort(key=lambda x: (-x.created_at, x.folder or "", x.name))
        return AssetsResponse(assets=asset_files)

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"list_assets: Error listing asset files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/assets", response_model=AssetFileInfo)
async def upload_asset(request: Request, filename: str = Query(...)):
    """Upload an asset file (image or video) to the assets directory."""
    try:
        # Validate file type - support both images and videos
        allowed_image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        allowed_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        allowed_extensions = allowed_image_extensions | allowed_video_extensions

        file_extension = Path(filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
            )

        # Determine asset type
        if file_extension in allowed_image_extensions:
            asset_type = "image"
        else:
            asset_type = "video"

        # Ensure assets directory exists
        assets_dir = get_assets_dir()
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Read file content from request body
        content = await request.body()

        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {max_size / (1024 * 1024):.0f}MB",
            )

        # Save file to assets directory
        file_path = assets_dir / filename
        file_path.write_bytes(content)

        # Return file info matching AssetFileInfo structure
        size_mb = len(content) / (1024 * 1024)
        created_at = file_path.stat().st_ctime
        relative_path = file_path.relative_to(assets_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )

        logger.info(f"upload_asset: Uploaded {asset_type} file: {file_path}")
        return AssetFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
            type=asset_type,
            created_at=created_at,
        )

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"upload_asset: Error uploading asset file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/assets/{asset_path:path}")
async def serve_asset(asset_path: str):
    """Serve an asset file (for thumbnails/previews)."""
    try:
        assets_dir = get_assets_dir()
        file_path = assets_dir / asset_path

        # Security check: ensure the path is within assets directory
        try:
            file_path = file_path.resolve()
            assets_dir_resolved = assets_dir.resolve()
            if not str(file_path).startswith(str(assets_dir_resolved)):
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception:
            raise HTTPException(status_code=403, detail="Invalid path") from None

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")

        # Determine media type based on extension
        file_extension = file_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        media_type = media_types.get(file_extension, "application/octet-stream")

        return FileResponse(file_path, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"serve_asset: Error serving asset file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/models/status")
async def get_model_status(pipeline_id: str):
    """Check if models for a pipeline are downloaded and get download progress."""
    try:
        progress = download_progress_manager.get_progress(pipeline_id)

        # If download is in progress, always report as not downloaded
        if progress and progress.get("is_downloading"):
            return {"downloaded": False, "progress": progress}

        # Check if files actually exist
        downloaded = models_are_downloaded(pipeline_id)

        # Clean up progress if download is complete
        if downloaded and progress:
            download_progress_manager.clear_progress(pipeline_id)
            progress = None

        return {"downloaded": downloaded, "progress": progress}
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/models/download")
async def download_pipeline_models(request: DownloadModelsRequest):
    """Download models for a specific pipeline."""
    try:
        if not request.pipeline_id:
            raise HTTPException(status_code=400, detail="pipeline_id is required")

        pipeline_id = request.pipeline_id

        # Check if download already in progress
        existing_progress = download_progress_manager.get_progress(pipeline_id)
        if existing_progress and existing_progress.get("is_downloading"):
            raise HTTPException(
                status_code=409,
                detail=f"Download already in progress for {pipeline_id}",
            )

        # Download in a background thread to avoid blocking
        import threading

        def download_in_background():
            """Run download in background thread."""
            try:
                download_models(pipeline_id)
                download_progress_manager.mark_complete(pipeline_id)
            except Exception as e:
                logger.error(f"Error downloading models for {pipeline_id}: {e}")
                download_progress_manager.clear_progress(pipeline_id)

        thread = threading.Thread(target=download_in_background)
        thread.daemon = True
        thread.start()

        return {"message": f"Model download started for {pipeline_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model download: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def is_spout_available() -> bool:
    """Check if Spout is available (native Windows only, not WSL)."""
    # Spout requires native Windows - it won't work in WSL/Linux
    return sys.platform == "win32"


@app.get("/api/v1/hardware/info", response_model=HardwareInfoResponse)
async def get_hardware_info():
    """Get hardware information including available VRAM and Spout availability."""
    try:
        vram_gb = None

        if torch.cuda.is_available():
            # Get total VRAM from the first GPU (in bytes), convert to GB
            _, total_mem = torch.cuda.mem_get_info(0)
            vram_gb = total_mem / (1024**3)

        return HardwareInfoResponse(
            vram_gb=vram_gb,
            spout_available=is_spout_available(),
        )
    except Exception as e:
        logger.error(f"Error getting hardware info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/logs/current")
async def get_current_logs():
    """Get the most recent application log file for bug reporting."""
    try:
        log_file_path = get_most_recent_log_file()

        if log_file_path is None or not log_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Log file not found. The application may not have logged anything yet.",
            )

        # Read the entire file into memory to avoid Content-Length issues
        # with actively written log files.
        # Use errors='replace' to handle non-UTF-8 bytes gracefully (e.g., Windows-1252
        # characters from subprocess output or exception messages on Windows).
        log_content = log_file_path.read_text(encoding="utf-8", errors="replace")

        # Return as a text response with proper headers for download
        return Response(
            content=log_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{log_file_path.name.replace(".log", ".txt")}"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving log file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/{path:path}")
async def serve_frontend(request: Request, path: str):
    """Serve the frontend for all non-API routes (fallback for client-side routing)."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")

    # Check if requesting a specific file that exists
    file_path = frontend_dist / path
    if file_path.exists() and file_path.is_file():
        # Determine media type based on extension to fix MIME type issues on Windows
        file_extension = file_path.suffix.lower()
        media_types = {
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".eot": "application/vnd.ms-fontobject",
        }
        media_type = media_types.get(file_extension)
        return FileResponse(file_path, media_type=media_type)

    # Fallback to index.html for SPA routing
    # This ensures clients like Electron alway fetch the latest HTML (which references hashed assets)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    raise HTTPException(status_code=404, detail="Frontend index.html not found")


def open_browser_when_ready(host: str, port: int, server):
    """Open browser when server is ready, with fallback to URL logging."""
    # Wait for server to be ready
    while not getattr(server, "started", False):
        time.sleep(0.1)

    # Determine the URL to open
    url = (
        f"http://localhost:{port}"
        if host in ["0.0.0.0", "127.0.0.1"]
        else f"http://{host}:{port}"
    )

    try:
        success = webbrowser.open(url)
        if success:
            logger.info(f"🌐 Opened browser at {url}")
    except Exception:
        success = False

    if not success:
        logger.info(f"🌐 UI is available at: {url}")


def run_server(reload: bool, host: str, port: int, no_browser: bool):
    """Run the Burn me while i'm hot server."""

    from scope.core.pipelines.registry import (
        PipelineRegistry,  # noqa: F401 - imported for side effects (registry initialization)
    )

    # Configure static file serving
    configure_static_files()

    # Check if we're in production mode (frontend dist exists)
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    is_production = frontend_dist.exists()

    if is_production:
        # Create server instance for production mode
        config = uvicorn.Config(
            "scope.server.app:app",
            host=host,
            port=port,
            reload=reload,
            log_config=None,  # Use our logging config, don't override it
        )
        server = uvicorn.Server(config)

        # Start browser opening thread (unless disabled)
        if not no_browser:
            browser_thread = threading.Thread(
                target=open_browser_when_ready,
                args=(host, port, server),
                daemon=True,
            )
            browser_thread.start()
        else:
            logger.info("main: Skipping browser auto-launch due to --no-browser")

        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # Clean shutdown on Ctrl+C
    else:
        # Development mode - just run normally
        uvicorn.run(
            "scope.server.app:app",
            host=host,
            port=port,
            reload=reload,
            log_config=None,  # Use our logging config, don't override it
        )


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information and exit")
@click.option(
    "--reload", is_flag=True, help="Enable auto-reload for development (default: False)"
)
@click.option("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
@click.option("--port", default=8000, help="Port to bind to (default: 8000)")
@click.option(
    "-N",
    "--no-browser",
    is_flag=True,
    help="Do not automatically open a browser window after the server starts",
)
@click.pass_context
def main(ctx, version: bool, reload: bool, host: str, port: int, no_browser: bool):
    # Handle version flag
    if version:
        print_version_info()
        sys.exit(0)

    # If no subcommand was invoked, run the server
    if ctx.invoked_subcommand is None:
        run_server(reload, host, port, no_browser)


def _is_preview_enabled():
    """Check if the DAYDREAM_SCOPE_PREVIEW feature flag is enabled."""
    return os.environ.get("DAYDREAM_SCOPE_PREVIEW", "").lower() in ("1", "true", "yes")


@main.command(hidden=not _is_preview_enabled())
def plugins():
    """List all installed plugins."""

    @suppress_init_output
    def _load_plugins():
        from scope.core.plugins import load_plugins, pm

        load_plugins()
        return pm.get_plugins()

    plugin_list = _load_plugins()

    if not plugin_list:
        click.echo("No plugins installed.")
        return

    click.echo(f"{len(plugin_list)} plugin(s) installed:\n")

    # List each plugin
    for plugin in plugin_list:
        plugin_name = plugin.__name__ if hasattr(plugin, "__name__") else str(plugin)
        click.echo(f"  • {plugin_name}")


@main.command()
def pipelines():
    """List all available pipelines."""

    @suppress_init_output
    def _load_pipelines():
        from scope.core.pipelines.registry import PipelineRegistry

        return PipelineRegistry.list_pipelines()

    all_pipelines = _load_pipelines()

    if not all_pipelines:
        click.echo("No pipelines available.")
        return

    click.echo(f"{len(all_pipelines)} pipeline(s) available:\n")

    # List all pipelines
    for pipeline_id in all_pipelines:
        click.echo(f"  • {pipeline_id}")


@main.command(hidden=not _is_preview_enabled())
@click.argument("packages", nargs=-1, required=False)
@click.option("--upgrade", is_flag=True, help="Upgrade packages to the latest version")
@click.option(
    "-e", "--editable", help="Install a project in editable mode from this path"
)
@click.option("--force-reinstall", is_flag=True, help="Force reinstall packages")
@click.option("--no-cache-dir", is_flag=True, help="Disable the cache")
@click.option(
    "--pre", is_flag=True, help="Include pre-release and development versions"
)
def install(packages, upgrade, editable, force_reinstall, no_cache_dir, pre):
    """Install a plugin."""
    args = ["uv", "pip", "install"]
    if upgrade:
        args.append("--upgrade")
    if editable:
        args += ["--editable", editable]
    if force_reinstall:
        args.append("--force-reinstall")
    if no_cache_dir:
        args.append("--no-cache-dir")
    if pre:
        args.append("--pre")
    args += list(packages)

    result = subprocess.run(args, capture_output=False)

    if result.returncode != 0:
        sys.exit(result.returncode)


@main.command(hidden=not _is_preview_enabled())
@click.argument("packages", nargs=-1, required=True)
@click.option("-y", "--yes", is_flag=True, help="Don't ask for confirmation")
def uninstall(packages, yes):
    """Uninstall a plugin."""
    args = ["uv", "pip", "uninstall"]
    args += list(packages)
    if yes:
        args.append("-y")

    result = subprocess.run(args, capture_output=False)

    if result.returncode != 0:
        sys.exit(result.returncode)


from .mp4p import (
    encrypt_video,
    create_public_mp4p,
    decrypt_video,
    burn_video,
    add_synthed_video,
    decrypt_synthed_video,
    MP4PData,
    VisualCipherMetadata,
)
from uuid import uuid4


class EncryptVideoRequest(BaseModel):
    videoBase64: str
    expiresAt: int


class CreateMP4PRequest(BaseModel):
    videoId: str | None = None


class DecryptVideoRequest(BaseModel):
    mp4pData: MP4PData


class BurnVideoRequest(BaseModel):
    mp4pData: MP4PData
    apiKey: str


class AddSynthedVideoRequest(BaseModel):
    mp4pData: MP4PData
    synthedVideoBase64: str
    promptsUsed: list
    visualCipher: dict | None = None
    encryptedMaskFrames: list[str] | None = None
    maskFrameIndexMap: list[int] | None = None
    maskPayloadCodec: str | None = None


class VisualCipherRequest(BaseModel):
    mp4pData: MP4PData
    synthedVideoBase64: str
    synthedMimeType: str | None = None
    originalVideoBase64: str | None = None
    maskId: str
    prompt: str
    params: dict
    seed: int
    pipelineId: str
    maskMode: str = "inside"


class LoadMP4PRequest(BaseModel):
    mp4pData: MP4PData
    burnIndex: int | None = None


class Sam3MaskRequest(BaseModel):
    videoBase64: str
    prompt: str
    box: list[int] | None = None
    input_fps: float | None = None


class RestoreMP4PRequest(BaseModel):
    mp4pData: MP4PData
    visualCipher: dict
    burnIndex: int | None = None


@app.post("/api/v1/mp4p/encrypt")
async def encrypt_video_endpoint(request: EncryptVideoRequest):
    try:
        video_data = base64.b64decode(request.videoBase64)
        video_id = str(uuid4())

        mp4p_data = await encrypt_video(video_data, request.expiresAt, video_id)

        return {"success": True, "data": mp4p_data.model_dump()}
    except Exception as e:
        logger.error(f"Error encrypting video: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/create")
async def create_mp4p_endpoint(request: CreateMP4PRequest):
    try:
        video_id = request.videoId or str(uuid4())
        mp4p_data = create_public_mp4p(video_id)
        return {"success": True, "data": mp4p_data.model_dump()}
    except Exception as e:
        logger.error(f"Error creating MP4P: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/decrypt")
async def decrypt_video_endpoint(request: DecryptVideoRequest):
    try:
        video_buffer = await decrypt_video(request.mp4pData)

        return {
            "success": True,
            "videoBase64": base64.b64encode(video_buffer).decode(),
            "metadata": request.mp4pData.metadata.model_dump()
        }
    except Exception as e:
        logger.error(f"Error decrypting video: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/burn")
async def burn_video_endpoint(request: BurnVideoRequest):
    try:
        video_buffer = await decrypt_video(request.mp4pData)

        burned_mp4p = await burn_video(
            request.mp4pData,
            request.apiKey,
            video_buffer
        )

        return {"success": True, "data": burned_mp4p.model_dump()}
    except Exception as e:
        logger.error(f"Error burning video: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/add-synthed")
async def add_synthed_video_endpoint(request: AddSynthedVideoRequest):
    try:
        synthed_video_data = base64.b64decode(request.synthedVideoBase64)

        visual_cipher = (
            VisualCipherMetadata.model_validate(request.visualCipher)
            if request.visualCipher is not None
            else None
        )

        updated_mp4p = await add_synthed_video(
            request.mp4pData,
            synthed_video_data,
            request.promptsUsed,
            visual_cipher=visual_cipher,
            encrypted_mask_frames=request.encryptedMaskFrames,
            mask_frame_index_map=request.maskFrameIndexMap,
            mask_payload_codec=request.maskPayloadCodec,
        )

        return {"success": True, "data": updated_mp4p.model_dump()}
    except Exception as e:
        logger.error(f"Error adding synthed video: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/visual-cipher")
async def generate_visual_cipher_endpoint(request: VisualCipherRequest):
    try:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for visual cipher generation.")

        session = sam3_mask_manager.get_session(request.maskId)
        if session is None:
            raise RuntimeError(f"SAM3 session {request.maskId} not found")

        logger.info(
            "VisualCipher start: mp4p_id=%s mask_id=%s pipeline=%s seed=%s prompt_len=%s",
            request.mp4pData.metadata.id,
            request.maskId,
            request.pipelineId,
            request.seed,
            len(request.prompt or ""),
        )

        if request.originalVideoBase64:
            original_video = base64.b64decode(request.originalVideoBase64)
        else:
            original_video = await decrypt_video(request.mp4pData)
        synthed_video = base64.b64decode(request.synthedVideoBase64)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "original.mp4"
            synth_ext = "mp4"
            if request.synthedMimeType:
                if "webm" in request.synthedMimeType:
                    synth_ext = "webm"
                elif "mp4" in request.synthedMimeType:
                    synth_ext = "mp4"
            synth_path = Path(tmpdir) / f"synth.{synth_ext}"
            original_path.write_bytes(original_video)
            synth_path.write_bytes(synthed_video)

            cap_orig = cv2.VideoCapture(str(original_path))
            cap_synth = cv2.VideoCapture(str(synth_path))
            if not cap_orig.isOpened() or not cap_synth.isOpened():
                raise RuntimeError("Failed to open video streams for visual cipher.")

            mask_dir = session.mask_dir
            mask_width = session.width
            mask_height = session.height
            fps = session.input_fps or session.sam3_fps or float(
                cap_orig.get(cv2.CAP_PROP_FPS) or 15.0
            )
            out_width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            out_height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if out_width <= 0 or out_height <= 0:
                out_width = mask_width
                out_height = mask_height

            out_path = Path(tmpdir) / "burn_composite.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_width, out_height))
            if not writer.isOpened():
                raise RuntimeError("Failed to open burn video writer.")

            payload_frames: list[str] = []
            index_map: list[int] = []

            prompt_bytes = request.prompt.encode()
            params_bytes = json.dumps(request.params, sort_keys=True).encode()
            seed_bytes = str(request.seed).encode()
            base_key_material = prompt_bytes + b"|" + params_bytes + b"|" + seed_bytes
            base_key = hashes.Hash(hashes.SHA256(), backend=default_backend())
            base_key.update(base_key_material)
            base_key_bytes = base_key.finalize()

            frame_index = 0
            while True:
                ret_orig, frame_orig = cap_orig.read()
                ret_synth, frame_synth = cap_synth.read()
                if not ret_orig or not ret_synth:
                    break

                mask_path = mask_dir / f"{frame_index:06d}.png"
                if not mask_path.exists():
                    frame_index += 1
                    continue

                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.shape[0] != mask_height or mask.shape[1] != mask_width:
                    mask = cv2.resize(mask, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)

                orig_rgb = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
                synth_rgb = cv2.cvtColor(frame_synth, cv2.COLOR_BGR2RGB)
                if orig_rgb.shape[:2] != (mask_height, mask_width):
                    orig_rgb = cv2.resize(orig_rgb, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)
                if synth_rgb.shape[:2] != (mask_height, mask_width):
                    synth_rgb = cv2.resize(synth_rgb, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)

                mask_bool = mask > 0
                if request.maskMode == "outside":
                    mask_bool = ~mask_bool

                composite_frame = orig_rgb.copy()
                composite_frame[mask_bool] = synth_rgb[mask_bool]

                frame_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
                frame_hash.update(composite_frame.tobytes())
                frame_hash_bytes = frame_hash.finalize()

                h = hmac.HMAC(base_key_bytes, hashes.SHA256(), backend=default_backend())
                h.update(frame_index.to_bytes(4, "big"))
                h.update(frame_hash_bytes)
                frame_key = h.finalize()

                key_stream = bytearray()
                counter = 0
                target_len = mask_height * mask_width * 3
                while len(key_stream) < target_len:
                    h = hmac.HMAC(frame_key, hashes.SHA256(), backend=default_backend())
                    h.update(counter.to_bytes(4, "big"))
                    key_stream.extend(h.finalize())
                    counter += 1
                key_bytes = np.frombuffer(bytes(key_stream[:target_len]), dtype=np.uint8).reshape(
                    (mask_height, mask_width, 3)
                )

                encrypted = np.bitwise_xor(orig_rgb, key_bytes)
                encrypted[mask == 0] = 0

                rgba = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)
                rgba[:, :, :3] = encrypted
                rgba[:, :, 3] = mask

                img = Image.fromarray(rgba, mode="RGBA")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                payload_frames.append(base64.b64encode(buf.getvalue()).decode())
                index_map.append(frame_index)

                if (mask_height, mask_width) != (out_height, out_width):
                    composite_frame = cv2.resize(
                        composite_frame,
                        (out_width, out_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                writer.write(cv2.cvtColor(composite_frame, cv2.COLOR_RGB2BGR))
                frame_index += 1

            cap_orig.release()
            cap_synth.release()
            writer.release()
            logger.info(
                "VisualCipher done: frames=%s mask_res=%sx%s fps=%s",
                len(payload_frames),
                mask_width,
                mask_height,
                fps,
            )

            composite_bytes = out_path.read_bytes()

        visual_cipher = VisualCipherMetadata(
            version=1,
            pipelineId=request.pipelineId,
            pipelineVersionHash=None,
            prompt=request.prompt,
            params=request.params,
            seed=request.seed,
            maskMode=request.maskMode,
            maskResolution={"width": mask_width, "height": mask_height},
            frameCount=len(payload_frames),
            fps=fps,
        )

        return {
            "success": True,
            "visualCipher": visual_cipher.model_dump(exclude_none=True),
            "encryptedMaskFrames": payload_frames,
            "maskFrameIndexMap": index_map,
            "maskPayloadCodec": "png-rgba",
            "compositedVideoBase64": base64.b64encode(composite_bytes).decode(),
        }
    except Exception as e:
        logger.error(f"Error generating visual cipher: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/load")
async def load_mp4p_endpoint(request: LoadMP4PRequest):
    try:
        synthed_video = await decrypt_synthed_video(
            request.mp4pData, request.burnIndex
        )
        return {
            "success": True,
            "showSynthed": True,
            "videoBase64": base64.b64encode(synthed_video).decode() if synthed_video else None,
            "metadata": request.mp4pData.metadata.model_dump(),
            "selectedBurnIndex": request.burnIndex
        }
    except Exception as e:
        logger.error(f"Error loading MP4P: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/mp4p/restore")
async def restore_mp4p_endpoint(request: RestoreMP4PRequest):
    try:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for restore.")

        visual_cipher = VisualCipherMetadata.model_validate(request.visualCipher)
        logger.info(
            "Restore start: mp4p_id=%s burn_index=%s prompt_len=%s",
            request.mp4pData.metadata.id,
            request.burnIndex,
            len(visual_cipher.prompt or ""),
        )
        synthed_video = await decrypt_synthed_video(
            request.mp4pData, request.burnIndex
        )
        if not synthed_video:
            raise RuntimeError("No burn video available for restore.")

        encrypted_frames = request.mp4pData.encryptedMaskFrames or []
        index_map = request.mp4pData.maskFrameIndexMap or []
        if not encrypted_frames or not index_map:
            raise RuntimeError("Missing encrypted mask payload.")

        payload_by_index = {
            index_map[i]: encrypted_frames[i] for i in range(len(index_map))
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            synth_path = Path(tmpdir) / "synth.mp4"
            out_path = Path(tmpdir) / "restored.mp4"
            synth_path.write_bytes(synthed_video)

            cap = cv2.VideoCapture(str(synth_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open burn video.")

            fps = visual_cipher.fps or float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError("Invalid burn video resolution.")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Failed to open output writer.")

            mask_width = visual_cipher.maskResolution.get("width", 0)
            mask_height = visual_cipher.maskResolution.get("height", 0)
            if mask_width <= 0 or mask_height <= 0:
                mask_width = width
                mask_height = height

            prompt_bytes = visual_cipher.prompt.encode()
            params_bytes = json.dumps(visual_cipher.params, sort_keys=True).encode()
            seed_bytes = str(visual_cipher.seed).encode()
            base_key_material = prompt_bytes + b"|" + params_bytes + b"|" + seed_bytes
            base_key = hashes.Hash(hashes.SHA256(), backend=default_backend())
            base_key.update(base_key_material)
            base_key_bytes = base_key.finalize()

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                synth_frame = rgb_frame
                if (mask_height, mask_width) != (height, width):
                    synth_frame = cv2.resize(
                        rgb_frame, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR
                    )

                payload_b64 = payload_by_index.get(frame_index)
                if payload_b64:
                    payload_bytes = base64.b64decode(payload_b64)
                    payload_img = Image.open(io.BytesIO(payload_bytes)).convert("RGBA")
                    payload = np.array(payload_img)
                    if mask_width <= 0 or mask_height <= 0:
                        mask_height, mask_width = payload.shape[:2]
                    encrypted = payload[:, :, :3]
                    mask = payload[:, :, 3]

                    frame_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
                    frame_hash.update(synth_frame.tobytes())
                    frame_hash_bytes = frame_hash.finalize()

                    h = hmac.HMAC(base_key_bytes, hashes.SHA256(), backend=default_backend())
                    h.update(frame_index.to_bytes(4, "big"))
                    h.update(frame_hash_bytes)
                    frame_key = h.finalize()

                    key_stream = bytearray()
                    counter = 0
                    target_len = mask_height * mask_width * 3
                    while len(key_stream) < target_len:
                        h = hmac.HMAC(frame_key, hashes.SHA256(), backend=default_backend())
                        h.update(counter.to_bytes(4, "big"))
                        key_stream.extend(h.finalize())
                        counter += 1
                    key_bytes = np.frombuffer(
                        bytes(key_stream[:target_len]), dtype=np.uint8
                    ).reshape((mask_height, mask_width, 3))

                    decrypted = np.bitwise_xor(encrypted, key_bytes)
                    if visual_cipher.maskMode == "inside":
                        synth_frame[mask > 0] = decrypted[mask > 0]
                    else:
                        synth_frame[mask == 0] = decrypted[mask == 0]

                if (mask_height, mask_width) != (height, width):
                    synth_frame = cv2.resize(
                        synth_frame, (width, height), interpolation=cv2.INTER_LINEAR
                    )

                out_bgr = cv2.cvtColor(synth_frame, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)
                frame_index += 1

            cap.release()
            writer.release()

            restored_bytes = out_path.read_bytes()

        logger.info("Restore done: bytes=%s", len(restored_bytes))
        return {
            "success": True,
            "videoBase64": base64.b64encode(restored_bytes).decode(),
        }
    except Exception as e:
        logger.error(f"Error restoring MP4P: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/sam3/mask")
async def generate_sam3_mask(request: Sam3MaskRequest):
    try:
        session = await asyncio.to_thread(
            sam3_mask_manager.generate_masks,
            request.videoBase64,
            request.prompt,
            request.box,
            request.input_fps,
        )
        return {
            "success": True,
            "maskId": session.session_id,
            "frameCount": session.frame_count,
            "height": session.height,
            "width": session.width,
            "inputFps": session.input_fps,
            "sam3Fps": session.sam3_fps,
        }
    except Exception as e:
        logger.error(f"Error generating SAM3 masks: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    main()
