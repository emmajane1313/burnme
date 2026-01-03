from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam


class MaskCompositeBlock(ModularPipelineBlocks):
    @property
    def description(self) -> str:
        return "Composite decoded output with original video using a SAM3 mask."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "output_video",
                required=True,
                type_hint=torch.Tensor,
                description="Decoded video frames",
            ),
            InputParam(
                "video_raw",
                default=None,
                type_hint=torch.Tensor | None,
                description="Preprocessed original video for compositing",
            ),
            InputParam(
                "mask_frames",
                default=None,
                type_hint=torch.Tensor | None,
                description="Preprocessed SAM3 mask frames",
            ),
            InputParam(
                "sam3_mask_mode",
                default="inside",
                type_hint=str,
                description="Composite mode: inside or outside mask",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "output_video",
                type_hint=torch.Tensor,
                description="Masked composite output",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        if block_state.mask_frames is None or block_state.video_raw is None:
            self.set_block_state(state, block_state)
            return components, state

        output_video = block_state.output_video
        mask = block_state.mask_frames
        video_raw = block_state.video_raw

        # Align frame count if needed
        min_frames = min(output_video.shape[2], mask.shape[2], video_raw.shape[2])
        output_video = output_video[:, :, :min_frames]
        mask = mask[:, :, :min_frames]
        video_raw = video_raw[:, :, :min_frames]

        # Convert mask from [-1, 1] to [0, 1] if needed
        if mask.min() < 0:
            mask = (mask + 1.0) / 2.0
        mask = mask.clamp(0, 1)

        # Expand mask to match channels
        if mask.shape[1] == 1 and output_video.shape[1] != 1:
            mask = mask.expand(-1, output_video.shape[1], -1, -1, -1)

        if block_state.sam3_mask_mode == "outside":
            mask = 1.0 - mask

        block_state.output_video = output_video * mask + video_raw * (1.0 - mask)

        self.set_block_state(state, block_state)
        return components, state
