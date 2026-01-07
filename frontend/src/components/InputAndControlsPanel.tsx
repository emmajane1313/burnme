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
  generateVisualCipherPayload,
  type MP4PData,
} from "../lib/mp4p-api";

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
  idMascaraSam?: string | null;
  onGenerarMascara?: () => void;
  sam3SinDetecciones?: boolean;
  cajaSamPromptActiva?: boolean;
  cajaSam?: [number, number, number, number] | null;
  onCajaSamChange?: (box: [number, number, number, number] | null) => void;
  onCajaSamPromptActiva?: () => void;
  onCajaSamPromptCancelar?: () => void;
  sam3Listo?: boolean;
  estadoMascaraSam?: string | null;
  sam3Ta3mel?: boolean;
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
  idMascaraSam = null,
  onGenerarMascara,
  sam3SinDetecciones = false,
  cajaSamPromptActiva = false,
  cajaSam = null,
  onCajaSamChange,
  onCajaSamPromptActiva,
  onCajaSamPromptCancelar,
  sam3Listo = false,
  estadoMascaraSam = null,
  sam3Ta3mel = false,
  sourceVideoBlocked = false,
}: InputAndControlsPanelProps) {
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
  const [pendingKeyDownload, setPendingKeyDownload] = useState<{
    filename: string;
    payload: string;
  } | null>(null);
  const [boxDisplay, setBoxDisplay] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  const isDrawingBoxRef = useRef(false);
  const boxStartRef = useRef<{ x: number; y: number } | null>(null);
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
    if (!cajaSamPromptActiva) {
      setBoxDisplay(null);
    }
  }, [cajaSamPromptActiva]);

  useEffect(() => {
    if (!cajaSam) {
      setBoxDisplay(null);
    }
  }, [cajaSam]);

  useEffect(() => {
    if (prefillVideoFile) {
      setUploadedVideoFile(prefillVideoFile);
    }
  }, [prefillVideoFile]);

  const handleBoxPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!cajaSamPromptActiva || !videoRef.current) {
      return;
    }
    const rect = videoRef.current.getBoundingClientRect();
    const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
    const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
    isDrawingBoxRef.current = true;
    boxStartRef.current = { x, y };
    setBoxDisplay({ x, y, width: 0, height: 0 });
    onCajaSamChange?.(null);
  };

  const handleBoxPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isDrawingBoxRef.current || !boxStartRef.current || !videoRef.current) {
      return;
    }
    const rect = videoRef.current.getBoundingClientRect();
    const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
    const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
    const start = boxStartRef.current;
    const left = Math.min(start.x, x);
    const top = Math.min(start.y, y);
    const width = Math.abs(x - start.x);
    const height = Math.abs(y - start.y);
    setBoxDisplay({ x: left, y: top, width, height });
  };

  const handleBoxPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isDrawingBoxRef.current || !boxStartRef.current || !videoRef.current) {
      return;
    }
    isDrawingBoxRef.current = false;
    const rect = videoRef.current.getBoundingClientRect();
    const endX = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
    const endY = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
    const start = boxStartRef.current;
    boxStartRef.current = null;
    const left = Math.min(start.x, endX);
    const top = Math.min(start.y, endY);
    const width = Math.abs(endX - start.x);
    const height = Math.abs(endY - start.y);
    if (width < 4 || height < 4) {
      setBoxDisplay(null);
      onCajaSamChange?.(null);
      return;
    }
    setBoxDisplay({ x: left, y: top, width, height });
    const videoWidth = videoRef.current.videoWidth || rect.width;
    const videoHeight = videoRef.current.videoHeight || rect.height;
    const scaleX = videoWidth / rect.width;
    const scaleY = videoHeight / rect.height;
    const x1 = Math.round(left * scaleX);
    const y1 = Math.round(top * scaleY);
    const x2 = Math.round((left + width) * scaleX);
    const y2 = Math.round((top + height) * scaleY);
    onCajaSamChange?.([x1, y1, x2, y2]);
  };

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
      setPendingKeyDownload(null);
      let keyDownload: { filename: string; payload: string } | null = null;
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

        let visualCipher;
        let encryptedMaskFrames;
        let maskFrameIndexMap;
        let maskPayloadCodec;
        let keyMaterial: string | null = null;
        if (idMascaraSam && promptTexts[0]) {
          if (!uploadedVideoFile) {
            throw new Error(
              "Missing original video for visual cipher payload."
            );
          }
          if (!globalThis.crypto?.getRandomValues) {
            throw new Error("Crypto API not available for key generation.");
          }
          const keyBytes = new Uint8Array(32);
          globalThis.crypto.getRandomValues(keyBytes);
          keyMaterial = Array.from(keyBytes, byte =>
            byte.toString(16).padStart(2, "0")
          ).join("");
          const originalVideoBase64 = uploadedVideoFile
            ? await blobToBase64(uploadedVideoFile, uploadedVideoFile.name)
            : undefined;
          const params = {
            denoising_step_list: [1000, 750, 500, 250],
            prompt_interpolation_method: "linear",
            noise_scale: 1.0,
            noise_controller: false,
          };
          const payload = await generateVisualCipherPayload(
            mp4pData,
            synthedBase64,
            mimeType,
            originalVideoBase64,
            idMascaraSam,
            promptTexts[0],
            params,
            seed ?? 42,
            pipelineId,
            "inside",
            keyMaterial,
            confirmedSynthedFps
          );
          visualCipher = payload.visualCipher;
          encryptedMaskFrames = payload.encryptedMaskFrames;
          maskFrameIndexMap = payload.maskFrameIndexMap;
          maskPayloadCodec = payload.maskPayloadCodec;

          mp4pData = await addSynthedVideoBase64(
            mp4pData,
            synthedBase64,
            publicLabels,
            mimeType,
            visualCipher,
            encryptedMaskFrames,
            maskFrameIndexMap,
            maskPayloadCodec
          );
        }

        if (!idMascaraSam || !promptTexts[0]) {
          mp4pData = await addSynthedVideoBase64(
            mp4pData,
            synthedBase64,
            publicLabels,
            mimeType,
            visualCipher,
            encryptedMaskFrames,
            maskFrameIndexMap,
            maskPayloadCodec
          );
        }

        if (visualCipher && keyMaterial) {
          const keyData = {
            mp4pId: mp4pData.metadata.id,
            burnIndex: (mp4pData.metadata.synthedVersions?.length || 1) - 1,
            visualCipher: {
              ...visualCipher,
              keyMaterial,
            },
          };
          const keyName = uploadedVideoFile
            ? uploadedVideoFile.name.replace(/\.[^.]+$/, "")
            : `burn-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
          keyDownload = {
            filename: `${keyName}.mp4p-key.json`,
            payload: JSON.stringify(keyData, null, 2),
          };
          setPendingKeyDownload(keyDownload);
        }
      }

      const filename = uploadedVideoFile
        ? uploadedVideoFile.name.replace(/\.[^.]+$/, "")
        : `burn-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
      await downloadMP4P(mp4pData, filename);
      if (keyDownload) {
        const keyBlob = new Blob([keyDownload.payload], {
          type: "application/json",
        });
        const keyUrl = URL.createObjectURL(keyBlob);
        const anchor = document.createElement("a");
        anchor.href = keyUrl;
        anchor.download = keyDownload.filename;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(keyUrl);
      }

      console.log("MP4P file exported successfully");
    } catch (error) {
      console.error("Failed to export MP4P:", error);
    } finally {
      setIsExporting(false);
    }
  };

  const canStartSynth =
    !isSynthCapturing &&
    !!uploadedVideoFile &&
    !!prompts[0]?.text?.trim() &&
    isStreaming &&
    !isLoading &&
    sam3Listo;

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0 py-3 px-4">
        <CardTitle className="text-sm font-medium text-white">
          Input & Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 overflow-y-auto flex-1 px-4 py-3 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div>
          <h3 className="text-xs font-medium mb-1.5">Video Input</h3>
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
                Initializing video...
              </div>
            ) : error ? (
              <div className="text-center text-red-500 text-sm p-4">
                <p>Video error:</p>
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
                {cajaSamPromptActiva ? (
                  <div
                    className="absolute inset-0 cursor-crosshair"
                    onPointerDown={handleBoxPointerDown}
                    onPointerMove={handleBoxPointerMove}
                    onPointerUp={handleBoxPointerUp}
                  >
                    <div className="absolute left-2 top-2 rounded-full bg-black/60 px-3 py-1 text-[11px] text-white">
                      Drag to box the person
                    </div>
                    {boxDisplay ? (
                      <div
                        className="absolute border-2 border-emerald-300 bg-emerald-300/10"
                        style={{
                          left: boxDisplay.x,
                          top: boxDisplay.y,
                          width: boxDisplay.width,
                          height: boxDisplay.height,
                        }}
                      />
                    ) : null}
                  </div>
                ) : null}
                {isSynthCapturing ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/60">
                    <Spinner size={22} />
                    <span className="text-xs text-muted-foreground">
                      Burning...
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
                    Upload a vid to begin
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
                Change Video
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
                {isVideoPaused ? "Play" : "Pause"}
              </Button>
            </div>
          )}
          {pipelines ? (
            <div className="mt-3 space-y-1">
              <h3 className="text-xs font-medium">Pipeline</h3>
              <Select
                value={pipelineId}
                onValueChange={value => onPipelineIdChange?.(value)}
                disabled={isSynthCapturing || isLoading || isConnecting}
              >
                <SelectTrigger className="w-full h-8">
                  <SelectValue placeholder="Select a pipeline" />
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
                <h3 className="text-sm font-medium">Style</h3>
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

        {onGenerarMascara && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">SAM3 Mask</h3>
            <div className="text-xs text-muted-foreground">
              Requires HF access to{" "}
              <a
                href="https://huggingface.co/facebook/sam3"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                facebook/sam3
              </a>
              . Request access before generating masks.
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Button
                size="xs"
                onClick={onGenerarMascara}
                disabled={
                  sam3Ta3mel ||
                  isConnecting ||
                  isLoading ||
                  (cajaSamPromptActiva && !cajaSam)
                }
              >
                {sam3Ta3mel
                  ? "Generating..."
                  : cajaSamPromptActiva
                    ? "Regenerate with Box"
                    : "Regenerate Mask"}
              </Button>
              {!cajaSamPromptActiva ? (
                <Button
                  size="xs"
                  variant="secondary"
                  onClick={onCajaSamPromptActiva}
                  disabled={sam3Ta3mel || isConnecting || isLoading}
                >
                  Use Box Prompt
                </Button>
              ) : null}
              {cajaSamPromptActiva ? (
                <Button
                  size="xs"
                  variant="ghost"
                  onClick={onCajaSamPromptCancelar}
                  disabled={sam3Ta3mel}
                >
                  Cancel Box
                </Button>
              ) : null}
            </div>
            {sam3SinDetecciones && !cajaSamPromptActiva ? (
              <div className="text-xs text-muted-foreground">
                No mask detected. Try box prompt for better focus.
              </div>
            ) : null}
            {cajaSamPromptActiva ? (
              <div className="text-xs text-muted-foreground">
                Draw a box on the preview, then regenerate.
              </div>
            ) : null}
            {estadoMascaraSam && (
              <div className="text-xs text-muted-foreground">{estadoMascaraSam}</div>
            )}
          </div>
        )}

        <div className="space-y-2">
          <h3 className="text-sm font-medium">Burn</h3>
          <div className="flex flex-wrap items-center text-xs gap-2">
            <Button
              onClick={onStartSynth}
              disabled={!canStartSynth || isConnecting || isLoading}
              size="xs"
            >
              Start Burn
            </Button>
            {isSynthCapturing ? (
              <Button
                onClick={onCancelSynth}
                disabled={isLoading}
                size="xs"
                variant="destructive"
              >
                Cancel Burn
              </Button>
            ) : null}
            {confirmedSynthedBlob && !isSynthCapturing ? (
              <Button onClick={onDeleteBurn} size="xs" variant="destructive">
                Delete Burn
              </Button>
            ) : null}
          </div>
          {isSynthCapturing && (
            <div className="mt-2 text-xs text-muted-foreground">
              {isRecordingSynthed ? "Recording" : "Preparing"} from start.
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
            {isExporting ? "Exporting..." : "Export MP4P"}
          </Button>
        </div>

        {pendingKeyDownload && (
          <div>
            <Button
              onClick={() => {
                const keyBlob = new Blob([pendingKeyDownload.payload], {
                  type: "application/json",
                });
                const keyUrl = URL.createObjectURL(keyBlob);
                const anchor = document.createElement("a");
                anchor.href = keyUrl;
                anchor.download = pendingKeyDownload.filename;
                document.body.appendChild(anchor);
                anchor.click();
                document.body.removeChild(anchor);
                URL.revokeObjectURL(keyUrl);
              }}
              className="w-full"
              size="sm"
              variant="secondary"
            >
              Download Key File
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
