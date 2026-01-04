import { useEffect } from "react";
import type { PromptItem } from "../lib/api";
import { usePromptManager } from "../hooks/usePromptManager";
import { PromptField } from "./shared/PromptField";
import { TemporalTransitionControls } from "./shared/TemporalTransitionControls";

interface PromptInputProps {
  className?: string;
  prompts: PromptItem[];
  onPromptsChange?: (prompts: PromptItem[]) => void;
  onPromptSend?: () => void;
  disabled?: boolean;
  interpolationMethod?: "linear" | "slerp";
  onInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  temporalInterpolationMethod?: "linear" | "slerp";
  onTemporalInterpolationMethodChange?: (method: "linear" | "slerp") => void;
  isLive?: boolean;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isStreaming?: boolean;
  transitionSteps?: number;
  onTransitionStepsChange?: (steps: number) => void;
}

export function PromptInput({
  className = "",
  prompts,
  onPromptsChange,
  onPromptSend,
  disabled = false,
  interpolationMethod: _interpolationMethod = "linear",
  onInterpolationMethodChange: _onInterpolationMethodChange,
  temporalInterpolationMethod = "slerp",
  onTemporalInterpolationMethodChange,
  isLive: _isLive = false,
  transitionSteps = 4,
  onTransitionStepsChange,
}: PromptInputProps) {

  const {
    prompts: managedPrompts,
    handlePromptTextChange,
    handleRemovePrompt,
  } = usePromptManager({
    prompts: prompts,
    maxPrompts: 1,
    defaultWeight: 100,
    onPromptsChange: onPromptsChange,
  });

  useEffect(() => {
    if (managedPrompts.length === 0) {
      onPromptsChange?.([{ text: "", weight: 100 }]);
      return;
    }
    if (managedPrompts.length > 1) {
      onPromptsChange?.([managedPrompts[0]]);
    }
  }, [managedPrompts, onPromptsChange]);

  const singlePrompt = managedPrompts[0] || { text: "", weight: 100 };

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-start">
        <PromptField
          prompt={singlePrompt}
          index={0}
          placeholder="A luminous humanoid figure standing upright, clearly human in silhouette but entirely composed of swirling cosmic matter, with no visible skin or facial details, its interior filled with dense star fields, nebula clouds, flowing fire-like energy, and liquid rainbow colors that continuously move and circulate inside the body, emitting a soft yet intense glow; the colors shift smoothly from deep blues and violets to bright reds, greens, and golds, as if powered by an internal stellar engine, with particles drifting, igniting, and dissolving in slow motion; the edges of the figure shimmer and pulse with light, slightly translucent, giving the impression of pure energy held in human form, set against a dark, minimal background to maximize contrast, ultra-detailed, high dynamic range, cinematic lighting, ethereal, otherworldly, and visually coherent rather than abstract chaos."
          showRemove={false}
          onSubmit={onPromptSend}
          onTextChange={handlePromptTextChange}
          onRemove={handleRemovePrompt}
          disabled={disabled}
        />
      </div>

      <div className="space-y-2">
        <TemporalTransitionControls
          transitionSteps={transitionSteps}
          onTransitionStepsChange={steps => onTransitionStepsChange?.(steps)}
          temporalInterpolationMethod={temporalInterpolationMethod}
          onTemporalInterpolationMethodChange={method =>
            onTemporalInterpolationMethodChange?.(method)
          }
          disabled={disabled}
          className="space-y-2"
        />

        <div />
      </div>
    </div>
  );
}
