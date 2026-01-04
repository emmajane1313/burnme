import type { InputMode } from "../types";

// Default prompts by mode - used across all pipelines for consistency
export const DEFAULT_PROMPTS: Record<InputMode, string> = {
  text: "A luminous humanoid figure standing upright, clearly human in silhouette but entirely composed of swirling cosmic matter, with no visible skin or facial details, its interior filled with dense star fields, nebula clouds, flowing fire-like energy, and liquid rainbow colors that continuously move and circulate inside the body, emitting a soft yet intense glow; the colors shift smoothly from deep blues and violets to bright reds, greens, and golds, as if powered by an internal stellar engine, with particles drifting, igniting, and dissolving in slow motion; the edges of the figure shimmer and pulse with light, slightly translucent, giving the impression of pure energy held in human form, set against a dark, minimal background to maximize contrast, ultra-detailed, high dynamic range, cinematic lighting, ethereal, otherworldly, and visually coherent rather than abstract chaos.",
  video:
    "A luminous humanoid figure standing upright, clearly human in silhouette but entirely composed of swirling cosmic matter, with no visible skin or facial details, its interior filled with dense star fields, nebula clouds, flowing fire-like energy, and liquid rainbow colors that continuously move and circulate inside the body, emitting a soft yet intense glow; the colors shift smoothly from deep blues and violets to bright reds, greens, and golds, as if powered by an internal stellar engine, with particles drifting, igniting, and dissolving in slow motion; the edges of the figure shimmer and pulse with light, slightly translucent, giving the impression of pure energy held in human form, set against a dark, minimal background to maximize contrast, ultra-detailed, high dynamic range, cinematic lighting, ethereal, otherworldly, and visually coherent rather than abstract chaos.",
};

export function getDefaultPromptForMode(mode: InputMode): string {
  return DEFAULT_PROMPTS[mode];
}
