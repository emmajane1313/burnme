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

        output_video = block_state.output_video
        mask = block_state.mask_frames
        video_raw = block_state.video_raw

        if video_raw is None:
            self.set_block_state(state, block_state)
            return components, state

        def to_btchw(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim != 5:
                return tensor
            # Heuristic: B C T H W if channel dim is small and T is large
            if tensor.shape[1] in (1, 3, 4) and tensor.shape[2] > 4:
                return tensor.permute(0, 2, 1, 3, 4)
            return tensor

        output_video = to_btchw(output_video)
        video_raw = to_btchw(video_raw)

        if mask is None:
            # No mask available yet; fall back to the original video to avoid
            # showing the raw synth output outside the mask.
            output_channels = output_video.shape[2]
            raw_channels = video_raw.shape[2]
            if output_channels != raw_channels:
                if raw_channels == 1 and output_channels > 1:
                    video_raw = video_raw.expand(-1, -1, output_channels, -1, -1)
                elif output_channels == 1 and raw_channels > 1:
                    video_raw = video_raw[:, :, :1]
                else:
                    min_channels = min(output_channels, raw_channels)
                    video_raw = video_raw[:, :, :min_channels]
            block_state.output_video = video_raw
            self.set_block_state(state, block_state)
            return components, state

        mask = to_btchw(mask)

        # Align frame count on T dimension
        min_frames = min(output_video.shape[1], mask.shape[1], video_raw.shape[1])
        output_video = output_video[:, :min_frames]
        mask = mask[:, :min_frames]
        video_raw = video_raw[:, :min_frames]

        # Normalize mask channels to single channel for compositing
        if mask.shape[2] != 1:
            mask = mask.mean(dim=2, keepdim=True)

        # Align channel count between output and original video.
        output_channels = output_video.shape[2]
        raw_channels = video_raw.shape[2]
        if output_channels != raw_channels:
            if output_channels == 1 and raw_channels > 1:
                output_video = output_video.expand(-1, -1, raw_channels, -1, -1)
            elif raw_channels == 1 and output_channels > 1:
                video_raw = video_raw.expand(-1, -1, output_channels, -1, -1)
            else:
                min_channels = min(output_channels, raw_channels)
                output_video = output_video[:, :, :min_channels]
                video_raw = video_raw[:, :, :min_channels]

        # Normalize mask to [0, 1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        if mask.min() < 0:
            mask = (mask + 1.0) / 2.0
        mask = mask.clamp(0, 1)
        # Hard threshold to prevent synth bleed outside the mask.
        mask = (mask >= 0.5).float()

        # Match video_raw range to output_video to avoid grayscale bleed.
        output_min = output_video.amin().item()
        output_max = output_video.amax().item()
        raw_min = video_raw.amin().item()
        raw_max = video_raw.amax().item()
        output_is_01 = output_min >= -0.01 and output_max <= 1.01
        raw_is_neg = raw_min < -0.01
        if output_is_01 and raw_is_neg:
            video_raw = (video_raw + 1.0) / 2.0
        elif not output_is_01 and raw_min >= -0.01 and raw_max <= 1.01:
            video_raw = video_raw * 2.0 - 1.0

        # Expand mask to match channels (BTCHW -> channel dim is 2)
        if mask.shape[2] == 1 and output_video.shape[2] != 1:
            mask = mask.expand(-1, -1, output_video.shape[2], -1, -1)

        if block_state.sam3_mask_mode == "outside":
            mask = 1.0 - mask

        block_state.output_video = output_video * mask + video_raw * (1.0 - mask)

        self.set_block_state(state, block_state)
        return components, state
