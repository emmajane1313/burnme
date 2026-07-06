"""
Defines which artifacts each pipeline requires.
"""

from .artifacts import HuggingfaceRepoArtifact

# Common artifacts shared across pipelines
WAN_1_3B_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    files=["config.json", "Wan2.1_VAE.pth", "google"],
)

UMT5_ENCODER_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    files=["umt5-xxl-enc-fp8_e4m3fn.safetensors"],
)

VACE_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    files=["Wan2_1-VACE_module_1_3B_bf16.safetensors"],
)

# Pipeline-specific artifacts
PIPELINE_ARTIFACTS = {
    "streamdiffusionv2": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="jerryfeng/StreamDiffusionV2",
            files=["wan_causal_dmd_v2v/model.pt"],
        ),
    ],
    "memflow": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="KlingTeam/MemFlow",
            files=["base.pt", "lora.pt"],
        ),
    ],
}
