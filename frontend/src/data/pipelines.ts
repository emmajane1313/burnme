import type { InputMode } from "../types";

// Default prompts by mode - used across all pipelines for consistency
export const DEFAULT_PROMPTS: Record<InputMode, string> = {
  text: "A thousand splendid suns.",
  video:
    "A thousand splendid suns.",
};

export function getDefaultPromptForMode(mode: InputMode): string {
  return DEFAULT_PROMPTS[mode];
}
