import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { Toggle } from "./ui/toggle";
import { SliderWithInput } from "./ui/slider-with-input";
import { Minus, Plus } from "lucide-react";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type { PipelineId, SettingsState, PipelineInfo } from "../types";


interface SettingsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  seed?: number;
  onSeedChange?: (seed: number) => void;
  quantization?: "fp8_e4m3fn" | null;
  onQuantizationChange?: (quantization: "fp8_e4m3fn" | null) => void;
  kvCacheAttentionBias?: number;
  onKvCacheAttentionBiasChange?: (bias: number) => void;
  // Spout settings
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;
  // Whether Spout is available (server-side detection for native Windows, not WSL)
  spoutAvailable?: boolean;
  isVideoPaused?: boolean;
}

export function SettingsPanel({
  className = "",
  pipelines,
  pipelineId,
  onPipelineIdChange,
  isStreaming = false,
  isLoading = false,
  seed = 42,
  onSeedChange,
  quantization = "fp8_e4m3fn",
  onQuantizationChange,
  kvCacheAttentionBias = 0.3,
  onKvCacheAttentionBiasChange,
  spoutSender,
  onSpoutSenderChange,
  spoutAvailable = false,
  isVideoPaused = false,
}: SettingsPanelProps) {
  // Local slider state management hooks
  const kvCacheAttentionBiasSlider = useLocalSliderValue(
    kvCacheAttentionBias,
    onKvCacheAttentionBiasChange
  );
  // Validation error states
  const [seedError, setSeedError] = useState<string | null>(null);

  const handlePipelineIdChange = (value: string) => {
    if (pipelines && value in pipelines) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };


  const handleSeedChange = (value: number) => {
    const minValue = 0;
    const maxValue = 2147483647;

    // Validate and set error state
    if (value < minValue) {
      setSeedError(`Must be at least ${minValue}`);
    } else if (value > maxValue) {
      setSeedError(`Must be at most ${maxValue}`);
    } else {
      setSeedError(null);
    }

    // Always update the value (even if invalid)
    onSeedChange?.(value);
  };

  const incrementSeed = () => {
    const maxValue = 2147483647;
    const newValue = Math.min(maxValue, seed + 1);
    handleSeedChange(newValue);
  };

  const decrementSeed = () => {
    const minValue = 0;
    const newValue = Math.max(minValue, seed - 1);
    handleSeedChange(newValue);
  };

  const isControlsLocked = isLoading || (isStreaming && !isVideoPaused);

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium text-white">Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Pipeline ID</h3>
          <Select
            value={pipelineId}
            onValueChange={handlePipelineIdChange}
            disabled={isControlsLocked}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a pipeline" />
            </SelectTrigger>
            <SelectContent>
              {pipelines &&
                Object.keys(pipelines).map(id => (
                  <SelectItem key={id} value={id}>
                    {id}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>


        {pipelines?.[pipelineId]?.supportsQuantization && (
          <div className="space-y-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <LabelWithTooltip
                  label={PARAMETER_METADATA.seed.label}
                  tooltip={PARAMETER_METADATA.seed.tooltip}
                  className="text-sm text-foreground w-14"
                />
                <div
                  className={`flex-1 flex items-center border rounded-full overflow-hidden h-8 ${seedError ? "border-red-500" : ""}`}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={decrementSeed}
                    disabled={isControlsLocked}
                  >
                    <Minus className="h-3.5 w-3.5" />
                  </Button>
                  <Input
                    type="number"
                    value={seed}
                    onChange={e => {
                      const value = parseInt(e.target.value);
                      if (!isNaN(value)) {
                        handleSeedChange(value);
                      }
                    }}
                    disabled={isControlsLocked}
                    className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    min={0}
                    max={2147483647}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={incrementSeed}
                    disabled={isControlsLocked}
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
              {seedError && (
                <p className="text-xs text-red-500 ml-16">{seedError}</p>
              )}
            </div>
          </div>
        )}

        {pipelines?.[pipelineId]?.supportsCacheManagement && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
               
                {pipelines?.[pipelineId]?.supportsKvCacheBias && (
                  <SliderWithInput
                    label={PARAMETER_METADATA.kvCacheAttentionBias.label}
                    tooltip={PARAMETER_METADATA.kvCacheAttentionBias.tooltip}
                    value={kvCacheAttentionBiasSlider.localValue}
                    onValueChange={kvCacheAttentionBiasSlider.handleValueChange}
                    onValueCommit={kvCacheAttentionBiasSlider.handleValueCommit}
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    incrementAmount={0.01}
                    labelClassName="text-sm text-foreground w-20"
                    valueFormatter={kvCacheAttentionBiasSlider.formatValue}
                    inputParser={v => parseFloat(v) || 1.0}
                  />
                )}

              </div>
            </div>
          </div>
        )}

   
     
        {pipelines?.[pipelineId]?.supportsQuantization && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.quantization.label}
                    tooltip={PARAMETER_METADATA.quantization.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Select
                    value={quantization || "none"}
                    onValueChange={value => {
                      onQuantizationChange?.(
                        value === "none" ? null : (value as "fp8_e4m3fn")
                      );
                    }}
                    disabled={isControlsLocked}
                  >
                    <SelectTrigger className="w-[140px] h-7">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">None</SelectItem>
                      <SelectItem value="fp8_e4m3fn">
                        fp8_e4m3fn (Dynamic)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>
        )}

        {spoutAvailable && (
          <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <LabelWithTooltip
                label={PARAMETER_METADATA.spoutSender.label}
                tooltip={PARAMETER_METADATA.spoutSender.tooltip}
                className="text-sm text-foreground"
              />
              <Toggle
                pressed={spoutSender?.enabled ?? false}
                onPressedChange={enabled => {
                  onSpoutSenderChange?.({
                    enabled,
                    name: spoutSender?.name ?? "ScopeOut",
                  });
                }}
                variant="outline"
                size="sm"
                className="h-7"
              >
                {spoutSender?.enabled ? "ON" : "OFF"}
              </Toggle>
            </div>

            {spoutSender?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label="Sender Name:"
                  tooltip="The name of the sender that will send video to Spout-compatible apps like TouchDesigner, Resolume, OBS."
                  className="text-xs text-muted-foreground whitespace-nowrap"
                />
                <Input
                  type="text"
                  value={spoutSender?.name ?? "ScopeOut"}
                  onChange={e => {
                    onSpoutSenderChange?.({
                      enabled: spoutSender?.enabled ?? false,
                      name: e.target.value,
                    });
                  }}
                  disabled={isControlsLocked}
                  className="h-8 text-sm flex-1"
                  placeholder="ScopeOut"
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
