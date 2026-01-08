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
import { Toggle } from "./ui/toggle";
import { SliderWithInput } from "./ui/slider-with-input";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import { useLocalSliderValue } from "../hooks/useLocalSliderValue";
import type { PipelineId, SettingsState, PipelineInfo } from "../types";
import { useI18n } from "../i18n";


interface SettingsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  pipelineId: PipelineId;
  onPipelineIdChange?: (pipelineId: PipelineId) => void;
  isStreaming?: boolean;
  isLoading?: boolean;
  quantization?: "fp8_e4m3fn" | null;
  onQuantizationChange?: (quantization: "fp8_e4m3fn" | null) => void;
  kvCacheAttentionBias?: number;
  onKvCacheAttentionBiasChange?: (bias: number) => void;
  defaultLoraEnabled?: boolean;
  onDefaultLoraEnabledChange?: (enabled: boolean) => void;
  spoutSender?: SettingsState["spoutSender"];
  onSpoutSenderChange?: (spoutSender: SettingsState["spoutSender"]) => void;
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
  quantization = "fp8_e4m3fn",
  onQuantizationChange,
  kvCacheAttentionBias = 0.3,
  onKvCacheAttentionBiasChange,
  defaultLoraEnabled = true,
  onDefaultLoraEnabledChange,
  spoutSender,
  onSpoutSenderChange,
  spoutAvailable = false,
  isVideoPaused = false,
}: SettingsPanelProps) {
  const { t } = useI18n();
  const kvCacheAttentionBiasSlider = useLocalSliderValue(
    kvCacheAttentionBias,
    onKvCacheAttentionBiasChange
  );

  const handlePipelineIdChange = (value: string) => {
    if (pipelines && value in pipelines) {
      onPipelineIdChange?.(value as PipelineId);
    }
  };

  const isControlsLocked = isLoading || (isStreaming && !isVideoPaused);
  const loraToggleDisabled = isStreaming && !isVideoPaused;

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium text-white">
          {t("settings.title")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div className="space-y-2">
          <h3 className="text-sm font-medium">{t("settings.pipelineId")}</h3>
          <Select
            value={pipelineId}
            onValueChange={handlePipelineIdChange}
            disabled={isControlsLocked}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder={t("settings.selectPipeline")} />
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

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <LabelWithTooltip
              label={t("settings.lora.label")}
              tooltip={t("settings.lora.tooltip")}
              className="text-sm text-foreground"
            />
            <Toggle
              pressed={defaultLoraEnabled}
              onPressedChange={(value) => onDefaultLoraEnabledChange?.(value)}
              disabled={loraToggleDisabled}
              size="sm"
            >
              {defaultLoraEnabled ? t("settings.toggle.on") : t("settings.toggle.off")}
            </Toggle>
          </div>
          {loraToggleDisabled ? (
            <p className="text-xs text-muted-foreground">
              {t("settings.lora.pauseToToggle")}
            </p>
          ) : null}
        </div>


        {pipelines?.[pipelineId]?.supportsCacheManagement && (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="space-y-2 pt-2">
               
                {pipelines?.[pipelineId]?.supportsKvCacheBias && (
                  <SliderWithInput
                    label={t(PARAMETER_METADATA.kvCacheAttentionBias.label)}
                    tooltip={t(PARAMETER_METADATA.kvCacheAttentionBias.tooltip)}
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
                    label={t(PARAMETER_METADATA.quantization.label)}
                    tooltip={t(PARAMETER_METADATA.quantization.tooltip)}
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
                      <SelectItem value="none">{t("settings.quantization.none")}</SelectItem>
                      <SelectItem value="fp8_e4m3fn">
                        {t("settings.quantization.fp8")}
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
                label={t(PARAMETER_METADATA.spoutSender.label)}
                tooltip={t(PARAMETER_METADATA.spoutSender.tooltip)}
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
                {spoutSender?.enabled ? t("settings.spout.on") : t("settings.spout.off")}
              </Toggle>
            </div>

            {spoutSender?.enabled && (
              <div className="flex items-center gap-3">
                <LabelWithTooltip
                  label={t("settings.spout.senderName")}
                  tooltip={t("settings.spout.senderTooltip")}
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
                  placeholder={t("settings.spout.placeholder")}
                />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
