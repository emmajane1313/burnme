import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { PipelineInfo } from "../types";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import {
  createMP4P,
  downloadMP4P,
  addSynthedVideoBase64,
  blobToBase64,
  type MP4PData,
} from "../lib/mp4p-api";
import { useI18n } from "../i18n";

interface InputAndControlsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  isStreaming: boolean;
  isConnecting: boolean;
  isLoading?: boolean;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  baseMp4pData?: MP4PData | null;
  prefillVideoFile?: File | null;
  hideLocalPreview?: boolean;
  pipelineId: string;
  onPipelineIdChange?: (pipelineId: string) => void;
  seed?: number;
  prompts: PromptItem[];
  onPromptsChange: (prompts: PromptItem[]) => void;
  onTransitionSubmit: (transition: PromptTransition) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isVideoPaused?: boolean;
  confirmedSynthedBlob: Blob | null;
  confirmedSynthedFps?: number | null;
  isRecordingSynthed: boolean;
  isSynthCapturing: boolean;
  synthLockedPrompt: string;
  onStartSynth: () => void;
  onCancelSynth: () => void;
  onDeleteBurn?: () => void;
  onTogglePause?: () => void;
  sourceVideoBlocked?: boolean;
}

const PROMPT_PRESETS = [
  {
    id: "Chrome Interface Fever",
    prompt:
      "y2k, A hyper-chrome digital surface where molten silver gradients collide with icy cyan, neon orange, and ultraviolet tones, flowing across the frame like reflections on polished metal; floating interface icons such as hearts, arrows, loading bars, and pixel stars drift through the composition, creating the feel of an early 2000s desktop fantasy. Tribal flame motifs and sharp techno curves appear and fade inside the glossy layers as if embedded in liquid chrome. The texture feels overtly synthetic and machine-perfect, echoing the futuristic optimism of turn-of-the-millennium design, with micro glitter flecks suspended throughout that shimmer under uniform studio lighting without shadows. The overall effect is sleek yet loud, projecting confident energy through its metallic sheen and saturated contrasts, where every surface looks engineered for maximum visual impact, abstract, digital art, chrome texture, metallic gradients, cyber icons, tribal flames, techno curves, glossy finish, synthetic surface, neon accents, high contrast, bold colors, y2k aesthetic, early 2000s style, futuristic retro, edgy, energetic, seamless background, repetitive pattern, reflective surface, prismatic highlights, artificial light, glamorous, maximalist, visually dense, eye-catching, vibrant hues, saturated colors, digital medium, decorative, fashionable, trendy, intricate details, luminous glow, bold aesthetic.",
    image: "/assets/images/chrome.png",
  },
  {
    id: "Bubblegum Pop Collage",
    prompt:
      "y2k, A candy-coated pastel dreamscape where bubblegum pink, baby blue, butter yellow, and mint green ripple across the frame in soft, glossy waves like melted plastic toys under studio lights; oversized bubble typography, smiley faces, and butterfly stickers float through the scene, giving the impression of a playful early 2000s pop collage. Checkerboard grids and daisy motifs peek through the layers, dissolving into the shimmer as if seen through clear vinyl. The texture feels intentionally artificial and toy-like, channeling the sweet maximalism of Y2K pop culture, with fine sparkle dust embedded throughout that glints evenly without harsh shadows. The overall effect is cute yet bold, radiating upbeat energy through its saturated pastels and dense decoration, where every surface feels coated in glossy nostalgia, abstract, digital art, pastel gradients, bubble letters, smiley icons, butterfly motifs, checkerboard pattern, daisy print, glossy plastic texture, synthetic finish, soft glow, high saturation, y2k aesthetic, early 2000s pop style, playful, energetic, seamless background, repetitive pattern, smooth surface, artificial light, decorative, trendy, fashionable, maximalist design, intricate details, luminous highlights, eye-catching, vibrant pastels, digital medium, modern retro, cheerful, bold aesthetic.",
    image: "/assets/images/bubble.png",
  },
  {
    id: "Neon Cyber Rush",
    prompt:
      "y2k, A dark neon cyber backdrop where jet black and deep indigo surfaces pulse with streaks of acid green, hot magenta, and laser blue, slicing across the frame like nightclub lights in a futuristic arcade; pixel grids and wireframe tunnels stretch into the distance, while chrome butterflies and glowing stars hover in layered depth. Matrix-style code textures and flame decals emerge briefly within the glow, then dissolve back into the digital haze. The texture feels deliberately artificial and high-tech, evoking the edgy side of early 2000s cyber culture, with tiny luminous particles scattered throughout that sparkle under consistent, shadow-free lighting. The overall effect is intense and electric, pushing aggressive contrast and visual overload, where every surface looks engineered to scream speed, energy, and digital rebellion, abstract, digital art, neon glow, dark background, wireframe graphics, pixel grid, chrome butterflies, star shapes, flame decals, cyber texture, synthetic surface, glossy finish, high contrast, bold neon colors, y2k aesthetic, early 2000s cyber style, edgy, dynamic, seamless background, repetitive pattern, reflective elements, prismatic highlights, artificial light, futuristic, glamorous, maximalist, visually striking, saturated colors, digital medium, decorative, trendy, intricate details, luminous effects, bold aesthetic.",
    image: "/assets/images/neon.png",
  },
  {
    id: "Denim Graffiti Dreams",
    prompt:
      "y2k, A textured denim-blue canvas layered with spray-paint neon pink, acid yellow, and electric turquoise, splashed across the frame like street art on low-rise jeans; rhinestone hearts, safety pins, and graffiti tags scatter through the composition, giving the feel of an early 2000s fashion zine come to life. Bandana patterns and checker stripes fade in and out beneath the paint, as if stitched into the surface itself. The texture feels deliberately artificial yet tactile, echoing the DIY glamour of Y2K street style, with fine metallic dust embedded throughout that sparkles evenly under soft studio lighting. The overall effect is rebellious and playful, radiating bold confidence through its color clashes and layered chaos, where every surface feels styled for maximum attitude, abstract, digital art, denim texture, graffiti paint, rhinestone accents, safety pin motifs, bandana pattern, checker stripes, glossy finish, synthetic surface, neon splashes, high contrast, bold colors, y2k aesthetic, early 2000s street fashion style, edgy, energetic, seamless background, repetitive pattern, decorative, trendy, fashionable, maximalist design, intricate details, luminous highlights, eye-catching, saturated colors, digital medium, bold aesthetic.",
    image: "/assets/images/denim.png",
  },
  {
    id: "Plasma Disco Mirage",
    prompt:
      "y2k, A luminous disco-inspired surface where molten gold, champagne pink, and ultraviolet violet swirl together like liquid light on a mirrored dance floor; star cutouts, disco balls, and glowing crescents float through the scene, creating a glamorous early 2000s club fantasy. Zebra stripes and metallic polka dots surface briefly within the glow, then dissolve back into the shimmer as if seen through heat waves. The texture feels overtly synthetic and polished, channeling the flashy maximalism of Y2K nightlife visuals, with ultra-fine glitter particles suspended throughout that sparkle under even, shadow-free lighting. The overall effect is bold and seductive, radiating high-energy glamour through its reflective surfaces and dense decoration, where every inch gleams with unapologetic excess, abstract, digital art, metallic gradients, disco motifs, star shapes, zebra pattern, polka dots, glossy texture, synthetic finish, luminous glow, high contrast, bold colors, y2k aesthetic, early 2000s club style, glamorous, energetic, seamless background, repetitive pattern, reflective surface, prismatic highlights, artificial light, decorative, trendy, fashionable, maximalist design, intricate details, visually striking, saturated colors, digital medium, bold aesthetic.",
    image: "/assets/images/disco.png",
  },
];

const SNAP_TRANSITION_STEPS = 0;

export function InputAndControlsPanel({
  className = "",
  pipelines,
  localStream,
  isInitializing,
  error,
  isStreaming,
  isConnecting,
  isLoading = false,
  onVideoFileUpload,
  baseMp4pData = null,
  prefillVideoFile = null,
  hideLocalPreview = false,
  pipelineId,
  onPipelineIdChange,
  seed = 42,
  prompts,
  onPromptsChange,
  onTransitionSubmit,
  onLivePromptSubmit,
  isVideoPaused = false,
  confirmedSynthedBlob,
  confirmedSynthedFps = null,
  isRecordingSynthed,
  isSynthCapturing,
  synthLockedPrompt,
  onStartSynth,
  onCancelSynth,
  onDeleteBurn,
  onTogglePause,
  sourceVideoBlocked = false,
}: InputAndControlsPanelProps) {
  const { t } = useI18n();

  const handlePresetSelect = (preset: (typeof PROMPT_PRESETS)[number]) => {
    const nextPrompts = [{ text: preset.prompt, weight: 100 }];
    onPromptsChange(nextPrompts);
    if (!isStreaming || isLoading || isSynthCapturing) {
      return;
    }
    if (SNAP_TRANSITION_STEPS > 0) {
      onTransitionSubmit({
        target_prompts: nextPrompts,
        num_steps: SNAP_TRANSITION_STEPS,
        temporal_interpolation_method: "slerp",
      });
    } else {
      onLivePromptSubmit?.(nextPrompts);
    }
  };
  const [uploadedVideoFile, setUploadedVideoFile] = useState<File | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pipeline = pipelines?.[pipelineId];

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = null;
      videoRef.current.srcObject = localStream;
      void videoRef.current.play();
    }
  }, [localStream]);

  useEffect(() => {
    if (isSynthCapturing && videoRef.current) {
      videoRef.current.pause();
    }
  }, [isSynthCapturing]);

  useEffect(() => {
    if (prefillVideoFile) {
      setUploadedVideoFile(prefillVideoFile);
    }
  }, [prefillVideoFile]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadedVideoFile(file);

    if (onVideoFileUpload) {
      try {
        await onVideoFileUpload(file);
      } catch (error) {
        console.error("Video upload failed:", error);
      }
    }

    event.target.value = "";
  };

  const handleTriggerFilePicker = () => {
    fileInputRef.current?.click();
  };

  const handleExportMP4P = async () => {
    if (!uploadedVideoFile && !baseMp4pData) {
      console.error("Missing required data for MP4P export");
      return;
    }

    try {
      setIsExporting(true);
      let mp4pData = baseMp4pData;
      if (!mp4pData) {
        mp4pData = await createMP4P();
      }

      if (confirmedSynthedBlob) {
        const promptTexts = synthLockedPrompt
          ? [synthLockedPrompt]
          : prompts.map(prompt => prompt.text);
        const publicLabels: string[] = [];
        const mimeType = confirmedSynthedBlob.type || "video/webm";
        const extension = mimeType.includes("mp4") ? "mp4" : "webm";
        const synthedBase64 = await blobToBase64(
          confirmedSynthedBlob,
          `synthed.${extension}`
        );

        mp4pData = await addSynthedVideoBase64(
          mp4pData,
          synthedBase64,
          publicLabels,
          mimeType,
          undefined,
          undefined,
          undefined,
          undefined
        );
      }

      const filename = uploadedVideoFile
        ? uploadedVideoFile.name.replace(/\.[^.]+$/, "")
        : `burn-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
      await downloadMP4P(mp4pData, filename);
      console.log("MP4P file exported successfully");
    } catch (error) {
      console.error("Failed to export MP4P:", error);
    } finally {
      setIsExporting(false);
    }
  };

  const canStartSynth =
    !isSynthCapturing &&
    !!prompts[0]?.text?.trim() &&
    isStreaming &&
    !isLoading;

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0 py-3 px-4">
        <CardTitle className="text-sm font-medium text-white">
          {t("inputControls.title")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 overflow-y-auto flex-1 px-4 py-3 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div>
          <h3 className="text-xs font-medium mb-1.5">{t("videoInput.title")}</h3>
          <div className="rounded-lg flex items-center justify-center bg-muted/10 overflow-hidden relative min-h-[120px]">
            {onVideoFileUpload && (
              <input
                type="file"
                accept="video/mp4"
                onChange={handleFileUpload}
                className="hidden"
                id="video-upload"
                ref={fileInputRef}
              />
            )}
            {isInitializing ? (
              <div className="text-center text-muted-foreground text-sm">
                {t("videoInput.initializing")}
              </div>
            ) : error ? (
              <div className="text-center text-red-500 text-sm p-4">
                <p>{t("videoInput.errorLabel")}</p>
                <p className="text-xs mt-1">{error}</p>
              </div>
            ) : localStream && !hideLocalPreview ? (
              <div className="relative w-full h-full">
                <video
                  key={`${uploadedVideoFile?.name ?? "preview"}-${uploadedVideoFile?.lastModified ?? 0}-${uploadedVideoFile?.size ?? 0}-${localStream?.id ?? "stream"}`}
                  ref={videoRef}
                  className="w-full h-full object-contain bg-black/20"
                  autoPlay
                  muted
                  playsInline
                />
                {isSynthCapturing ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/60">
                    <Spinner size={22} />
                    <span className="text-xs text-muted-foreground">
                      {t("videoInput.burning")}
                    </span>
                  </div>
                ) : null}
                {sourceVideoBlocked ? (
                  <div className="absolute inset-0 bg-black/20" />
                ) : null}
              </div>
            ) : (
              onVideoFileUpload && (
                <>
                  <label
                    htmlFor="video-upload"
                    className="mac-frosted-button px-4 py-3 text-sm text-center cursor-pointer"
                  >
                    {t("videoInput.uploadToBegin")}
                  </label>
                </>
              )
            )}
          </div>
          {localStream && !hideLocalPreview && onVideoFileUpload ? (
            <div className="mt-2 flex justify-center">
              <Button
                variant="secondary"
                size="xs"
                onClick={handleTriggerFilePicker}
              >
                {t("videoInput.changeVideo")}
              </Button>
            </div>
          ) : null}
          {pipeline?.supportsPrompts !== false && (
            <div className="flex items-center justify-center gap-2 mt-2">
              <Button
                onClick={onTogglePause}
                disabled={isSynthCapturing}
                size="xs"
                variant="secondary"
              >
                {isVideoPaused ? t("videoInput.play") : t("videoInput.pause")}
              </Button>
            </div>
          )}
          {pipelines ? (
            <div className="mt-3 space-y-1">
              <h3 className="text-xs font-medium">{t("pipeline.title")}</h3>
              <Select
                value={pipelineId}
                onValueChange={value => onPipelineIdChange?.(value)}
                disabled={isSynthCapturing || isLoading || isConnecting}
              >
                <SelectTrigger className="w-full h-8">
                  <SelectValue placeholder={t("pipeline.selectPlaceholder")} />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(pipelines).map(id => (
                    <SelectItem key={id} value={id}>
                      {id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}
        </div>

        <div>
          {pipeline?.supportsPrompts !== false && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium">{t("style.title")}</h3>
              </div>
              <div className="prompt-orb-grid">
                {PROMPT_PRESETS.map(preset => {
                  const isSelected = prompts[0]?.text === preset.prompt;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      className={`prompt-orb ${isSelected ? "is-selected" : ""}`}
                      onClick={() => handlePresetSelect(preset)}
                      disabled={isSynthCapturing}
                    >
                      <span className="prompt-orb-frame">
                        <img src={preset.image} alt={preset.id} />
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-medium">{t("burn.title")}</h3>
          <div className="flex flex-wrap items-center text-xs gap-2">
            <Button
              onClick={onStartSynth}
              disabled={!canStartSynth || isConnecting || isLoading}
              size="xs"
            >
              {t("burn.start")}
            </Button>
            {isSynthCapturing ? (
              <Button
                onClick={onCancelSynth}
                disabled={isLoading}
                size="xs"
                variant="destructive"
              >
                {t("burn.cancel")}
              </Button>
            ) : null}
            {confirmedSynthedBlob && !isSynthCapturing ? (
              <Button onClick={onDeleteBurn} size="xs" variant="destructive">
                {t("burn.delete")}
              </Button>
            ) : null}
          </div>
          {isSynthCapturing && (
            <div className="mt-2 text-xs text-muted-foreground">
              {t("burn.statusFromStart", {
                status: isRecordingSynthed
                  ? t("burn.status.recording")
                  : t("burn.status.preparing"),
              })}
            </div>
          )}
        </div>

        <div>
          <Button
            onClick={handleExportMP4P}
            disabled={
              (!uploadedVideoFile && !baseMp4pData) ||
              !confirmedSynthedBlob ||
              isConnecting ||
              isSynthCapturing ||
              isExporting
            }
            className="w-full"
            size="sm"
          >
            {isExporting ? t("export.exporting") : t("export.exportMp4p")}
          </Button>
        </div>

      </CardContent>
    </Card>
  );
}
