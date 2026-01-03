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
import { Minus, Plus, RotateCcw } from "lucide-react";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type {
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  SettingsState,
  InputMode,
  PipelineInfo,
} from "../types";
import { LoRAManager } from "./LoRAManager";


interface SettingsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  seed?: number;
  onSeedChange?: (seed: number) => void;
  denoisingSteps?: number[];
  onDenoisingStepsChange?: (denoisingSteps: number[]) => void;
  // Default denoising steps for reset functionality - derived from backend schema
  defaultDenoisingSteps: number[];
  noiseScale?: number;
  onNoiseScaleChange?: (noiseScale: number) => void;
  noiseController?: boolean;
  onNoiseControllerChange?: (enabled: boolean) => void;
  manageCache?: boolean;
  onManageCacheChange?: (enabled: boolean) => void;
  quantization?: "fp8_e4m3fn" | null;
  onQuantizationChange?: (quantization: "fp8_e4m3fn" | null) => void;
  kvCacheAttentionBias?: number;
  onKvCacheAttentionBiasChange?: (bias: number) => void;
  onResetCache?: () => void;
  loras?: LoRAConfig[];
  onLorasChange: (loras: LoRAConfig[]) => void;
  loraMergeStrategy?: LoraMergeStrategy;
  // Input mode for conditional rendering of noise controls
  inputMode?: InputMode;
  // Whether this pipeline supports noise controls in video mode (schema-derived)
  supportsNoiseControls?: boolean;
  // Spout settings
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;
  // Whether Spout is available (server-side detection for native Windows, not WSL)
  spoutAvailable?: boolean;
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
  denoisingSteps = [700, 500],
  onDenoisingStepsChange,
  defaultDenoisingSteps,
  noiseScale = 0.7,
  onNoiseScaleChange,
  noiseController = true,
  onNoiseControllerChange,
  manageCache = true,
  onManageCacheChange,
  quantization = "fp8_e4m3fn",
  onQuantizationChange,
  kvCacheAttentionBias = 0.3,
  onKvCacheAttentionBiasChange,
  onResetCache,
  loras = [],
  onLorasChange,
  loraMergeStrategy = "permanent_merge",
  inputMode,
  supportsNoiseControls = false,
  spoutSender,
  onSpoutSenderChange,
  spoutAvailable = false,
}: SettingsPanelProps) {
  // Local slider state management hooks
  const noiseScaleSlider = useLocalSliderValue(noiseScale, onNoiseScaleChange);
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

  const currentPipeline = pipelines?.[pipelineId];

  return (
    <Card className={`h-full flex flex-col y2k-panel ${className}`}>
      <CardHeader className="flex-shrink-0 y2k-panel-header">
        <CardTitle className="text-base font-medium">Settings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Pipeline ID</h3>
          <Select
            value={pipelineId}
            onValueChange={handlePipelineIdChange}
            disabled={isStreaming || isLoading}
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


        {currentPipeline?.supportsLoRA && (
          <div className="space-y-4">
            <LoRAManager
              loras={loras}
              onLorasChange={onLorasChange}
              disabled={isLoading}
              isStreaming={isStreaming}
              loraMergeStrategy={loraMergeStrategy}
            />
          </div>
        )}

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
                    disabled={isStreaming}
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
                    disabled={isStreaming}
                    className="text-center border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-8 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    min={0}
                    max={2147483647}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-none hover:bg-accent"
                    onClick={incrementSeed}
                    disabled={isStreaming}
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

                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.manageCache.label}
                    tooltip={PARAMETER_METADATA.manageCache.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Toggle
                    pressed={manageCache}
                    onPressedChange={onManageCacheChange || (() => {})}
                    variant="outline"
                    size="sm"
                    className="h-7"
                  >
                    {manageCache ? "ON" : "OFF"}
                  </Toggle>
                </div>

                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.resetCache.label}
                    tooltip={PARAMETER_METADATA.resetCache.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Button
                    type="button"
                    onClick={onResetCache || (() => {})}
                    disabled={manageCache}
                    variant="outline"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        {pipelines?.[pipelineId]?.supportsQuantization && (
          <SliderWithInput
            label="Denoise"
            tooltip={PARAMETER_METADATA.denoisingSteps.tooltip}
            value={denoisingSteps[0] ?? defaultDenoisingSteps[0] ?? 700}
            onValueChange={value =>
              onDenoisingStepsChange?.([Math.round(value)])
            }
            onValueCommit={value =>
              onDenoisingStepsChange?.([Math.round(value)])
            }
            min={0}
            max={1000}
            step={1}
            incrementAmount={10}
            disabled={isStreaming}
            labelClassName="text-sm text-foreground w-16"
          />
        )}

   
        {inputMode === "video" && supportsNoiseControls && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
                <div className="flex items-center justify-between gap-2">
                  <LabelWithTooltip
                    label={PARAMETER_METADATA.noiseController.label}
                    tooltip={PARAMETER_METADATA.noiseController.tooltip}
                    className="text-sm text-foreground"
                  />
                  <Toggle
                    pressed={noiseController}
                    onPressedChange={onNoiseControllerChange || (() => {})}
                    disabled={isStreaming}
                    variant="outline"
                    size="sm"
                    className="h-7"
                  >
                    {noiseController ? "ON" : "OFF"}
                  </Toggle>
                </div>
              </div>

              <SliderWithInput
                label={PARAMETER_METADATA.noiseScale.label}
                tooltip={PARAMETER_METADATA.noiseScale.tooltip}
                value={noiseScaleSlider.localValue}
                onValueChange={noiseScaleSlider.handleValueChange}
                onValueCommit={noiseScaleSlider.handleValueCommit}
                min={0.0}
                max={1.0}
                step={0.01}
                incrementAmount={0.01}
                disabled={noiseController}
                labelClassName="text-sm text-foreground w-20"
                valueFormatter={noiseScaleSlider.formatValue}
                inputParser={v => parseFloat(v) || 0.0}
              />
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
                    disabled={isStreaming}
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
                  disabled={isStreaming}
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
