import { useEffect } from "react";
import type { PromptItem } from "../lib/api";
import type { TimelinePrompt } from "./PromptTimeline";
import { usePromptManager } from "../hooks/usePromptManager";
import { PromptField } from "./shared/PromptField";
import { TemporalTransitionControls } from "./shared/TemporalTransitionControls";

interface PromptInputProps {
  className?: string;
  prompts: PromptItem[];
  onPromptsChange?: (prompts: PromptItem[]) => void;
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
  timelinePrompts?: TimelinePrompt[];
}

export function PromptInput({
  className = "",
  prompts,
  onPromptsChange,
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
          placeholder="a thousand splendid suns"
          showRemove={false}
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
