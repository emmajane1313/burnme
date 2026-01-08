/**
 * Parameter metadata including labels and tooltip descriptions
 * for the SettingsPanel and related components.
 *
 * This centralized configuration makes it easy to maintain
 * parameter descriptions across the application.
 */

export interface ParameterMetadata {
  label: string;
  tooltip: string;
}

export const PARAMETER_METADATA: Record<string, ParameterMetadata> = {
  height: {
    label: "param.height.label",
    tooltip: "param.height.tooltip",
  },
  width: {
    label: "param.width.label",
    tooltip: "param.width.tooltip",
  },
  seed: {
    label: "param.seed.label",
    tooltip: "param.seed.tooltip",
  },
  denoisingSteps: {
    label: "param.denoisingSteps.label",
    tooltip: "param.denoisingSteps.tooltip",
  },
  noiseController: {
    label: "param.noiseController.label",
    tooltip: "param.noiseController.tooltip",
  },
  noiseScale: {
    label: "param.noiseScale.label",
    tooltip: "param.noiseScale.tooltip",
  },
  quantization: {
    label: "param.quantization.label",
    tooltip: "param.quantization.tooltip",
  },
  kvCacheAttentionBias: {
    label: "param.kvCacheAttentionBias.label",
    tooltip: "param.kvCacheAttentionBias.tooltip",
  },
  loraMergeStrategy: {
    label: "param.loraMergeStrategy.label",
    tooltip: "param.loraMergeStrategy.tooltip",
  },
  loraScale: {
    label: "param.loraScale.label",
    tooltip: "param.loraScale.tooltip",
  },
  loraScaleDisabledDuringStream: {
    label: "param.loraScaleDisabledDuringStream.label",
    tooltip: "param.loraScaleDisabledDuringStream.tooltip",
  },
  spoutSender: {
    label: "param.spoutSender.label",
    tooltip: "param.spoutSender.tooltip",
  },
};
