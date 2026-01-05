"""Pipelines package."""


def __getattr__(name):
    """Lazy import for pipeline and config classes to avoid triggering heavy imports."""
    # Pipeline classes
    if name == "StreamDiffusionV2Pipeline":
        from .streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

        return StreamDiffusionV2Pipeline
    elif name == "MemFlowPipeline":
        from .memflow.pipeline import MemFlowPipeline

        return MemFlowPipeline
    # Config classes
    elif name == "BasePipelineConfig":
        from .base_schema import BasePipelineConfig

        return BasePipelineConfig
    elif name == "StreamDiffusionV2Config":
        from .streamdiffusionv2.schema import StreamDiffusionV2Config

        return StreamDiffusionV2Config
    elif name == "MemFlowConfig":
        from .memflow.schema import MemFlowConfig

        return MemFlowConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Pipeline classes
    "StreamDiffusionV2Pipeline",
    "MemFlowPipeline",
    # Config classes
    "BasePipelineConfig",
    "StreamDiffusionV2Config",
    "MemFlowConfig",
]
