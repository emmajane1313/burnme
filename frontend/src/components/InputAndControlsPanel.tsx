import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { PipelineInfo } from "../types";
import { Button } from "./ui/button";
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
  seed?: number;
  prompts: PromptItem[];
  onPromptsChange: (prompts: PromptItem[]) => void;
  onTransitionSubmit: (transition: PromptTransition) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isVideoPaused?: boolean;
  confirmedSynthedBlob: Blob | null;
  isRecordingSynthed: boolean;
  isSynthCapturing: boolean;
  synthLockedPrompt: string;
  onStartSynth: () => void;
  onCancelSynth: () => void;
  onDeleteBurn?: () => void;
  onTogglePause?: () => void;
  sam3MaskId?: string | null;
  onSam3Generate?: () => void;
  sam3Ready?: boolean;
  sam3Status?: string | null;
  isSam3Generating?: boolean;
  sourceVideoBlocked?: boolean;
}

const makePresetSvg = (label: string, start: string, end: string) => {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'>
    <defs>
      <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
        <stop offset='0%' stop-color='${start}'/>
        <stop offset='100%' stop-color='${end}'/>
      </linearGradient>
    </defs>
    <rect width='200' height='200' fill='url(%23g)'/>
    <circle cx='100' cy='100' r='72' fill='rgba(255,255,255,0.28)'/>
    <text x='100' y='112' text-anchor='middle' font-family='Handjet, Arial, sans-serif' font-size='40' fill='#2a0a18'>${label}</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
};

const PROMPT_PRESETS = [
  {
    id: "chrome-pop",
    prompt:
      "A luminous silhouette composed of swirling cosmic matter, filled with dense star fields, nebula clouds, flowing fire-like energy, and liquid rainbow colors that continuously move and circulate, emitting a soft yet intense glow; the colors shift smoothly from deep blues and violets to bright reds, greens, and golds, as if powered by an internal stellar engine, with particles drifting, igniting, and dissolving in slow motion; the edges shimmer and pulse with light, slightly translucent, energy crystals, high dynamic range, cinematic lighting, ethereal, otherworldly.",
    image: makePresetSvg("CHR", "#ffd0e6", "#b7ecff"),
  },
  {
    id: "pixel-dream",
    prompt:
      "Y2K exploding chrome hearts made of swirling cosmic matter and liquid rainbow light, packed with dense star fields, nebula clouds, and fire‑like energy streams; the hearts crack open and burst with glittering particles, prismatic shards, and plasma ribbons that circulate and flare in slow motion, emitting a soft but intense glow; colors shift smoothly from deep blues and violets into hot pinks, reds, neon greens, and molten gold, as if powered by an internal stellar engine; edges shimmer and pulse with translucent crystal highlights, high dynamic range, cinematic lighting, ethereal, otherworldly, glossy early‑2000s pop aesthetic.",
    image: makePresetSvg("PX", "#ffc7d9", "#c7d7ff"),
  },
  {
    id: "angel-heat",
    prompt:
      "Y2K firecore hearts and flames in glossy chrome and molten glass, blazing with neon orange, magenta, and electric blue heat; liquid flame ribbons spiral and explode outward with sparkling particles and glitter dust, pulsing like a club‑era screensaver; the fire glows from within, shifting from deep ember reds to hot pink and golden highlights, with shimmering translucent edges, high dynamic range, cinematic lighting, early‑2000s pop futurism.",
    image: makePresetSvg("ANG", "#ffe1f0", "#ffc6b0"),
  },
  {
    id: "neon-glass",
    prompt:
      "Y2K candy‑burst hearts and bubbles made of liquid sugar glass, coated in glossy chrome and pastel neon icing; swirling rainbow syrup, star‑glitter, and gummy‑like particles float and pop in slow motion, glowing from within; colors shift through cotton‑candy pinks, turquoise, lemon yellow, and electric purple with a soft but intense glow; translucent edges shimmer like hard candy, high dynamic range, cinematic lighting, early‑2000s pop aesthetic.",
    image: makePresetSvg("NEO", "#f7c2ff", "#c2ffe4"),
  },
  {
    id: "satin-fire",
    prompt:
      "Y2K leopard‑to‑dalmation print hearts in glossy chrome and translucent vinyl, covered with high‑contrast black‑and‑white spots and hot‑pink accents; the pattern ripples and warps across the surface like liquid plastic, shimmering with micro‑glitter and metallic highlights; bold 2000s pop vibe, high dynamic range, cinematic lighting, playful, glossy, otherworldly.",
    image: makePresetSvg("SAT", "#ffd2c2", "#f6b0e0"),
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
  seed = 42,
  prompts,
  onPromptsChange,
  onTransitionSubmit,
  onLivePromptSubmit,
  isVideoPaused = false,
  confirmedSynthedBlob,
  isRecordingSynthed,
  isSynthCapturing,
  synthLockedPrompt,
  onStartSynth,
  onCancelSynth,
  onDeleteBurn,
  onTogglePause,
  sam3MaskId = null,
  onSam3Generate,
  sam3Ready = false,
  sam3Status = null,
  isSam3Generating = false,
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pipeline = pipelines?.[pipelineId];

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

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

    // Reset the input value so the same file can be selected again
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
        if (sam3MaskId && promptTexts[0]) {
          if (!uploadedVideoFile) {
            throw new Error("Missing original video for visual cipher payload.");
          }
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
            sam3MaskId,
            promptTexts[0],
            params,
            seed ?? 42,
            pipelineId,
            "inside"
          );
          visualCipher = payload.visualCipher;
          encryptedMaskFrames = payload.encryptedMaskFrames;
          maskFrameIndexMap = payload.maskFrameIndexMap;
          maskPayloadCodec = payload.maskPayloadCodec;

          if (payload.compositedVideoBase64) {
            mp4pData = await addSynthedVideoBase64(
              mp4pData,
              payload.compositedVideoBase64,
              publicLabels,
              "video/webm",
              undefined,
              encryptedMaskFrames,
              maskFrameIndexMap,
              maskPayloadCodec
            );
          } else {
            mp4pData = await addSynthedVideoBase64(
              mp4pData,
              synthedBase64,
              publicLabels,
              mimeType,
              undefined,
              encryptedMaskFrames,
              maskFrameIndexMap,
              maskPayloadCodec
            );
          }
        }

        if (!sam3MaskId || !promptTexts[0]) {
          mp4pData = await addSynthedVideoBase64(
            mp4pData,
            synthedBase64,
            publicLabels,
            mimeType,
            undefined,
            encryptedMaskFrames,
            maskFrameIndexMap,
            maskPayloadCodec
          );
        }

        if (visualCipher) {
          const keyData = {
            mp4pId: mp4pData.metadata.id,
            burnIndex: (mp4pData.metadata.synthedVersions?.length || 1) - 1,
            visualCipher,
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
    sam3Ready;

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
          <div
            className="rounded-lg flex items-center justify-center bg-muted/10 overflow-hidden relative min-h-[120px] cursor-pointer"
            onClick={() => {
              if (localStream && !hideLocalPreview) {
                handleTriggerFilePicker();
              }
            }}
          >
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
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  muted
                  playsInline
                />
                {sourceVideoBlocked ? (
                  <div className="absolute inset-0 bg-black/20" />
                ) : null}
                <div className="absolute inset-x-0 bottom-2 flex justify-center pointer-events-none">
                  <span className="mac-frosted-button px-3 py-1 text-[11px] text-white opacity-80">
                    Click to change video
                  </span>
                </div>
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
        </div>

        <div>
          {pipeline?.supportsPrompts !== false && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium">Style</h3>
                {isSynthCapturing && (
                  <Badge variant="secondary" className="text-xs">
                    Locked
                  </Badge>
                )}
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

        {onSam3Generate && (
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
                onClick={onSam3Generate}
                disabled={isSam3Generating || isConnecting || isLoading}
              >
                {isSam3Generating ? "Generating..." : "Regenerate Mask"}
              </Button>
            </div>
            {sam3Status && (
              <div className="text-xs text-muted-foreground">{sam3Status}</div>
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
            <Button
              onClick={onCancelSynth}
              disabled={!isSynthCapturing || isLoading}
              size="xs"
              variant="destructive"
            >
              Cancel Burn
            </Button>
            {confirmedSynthedBlob && !isSynthCapturing && (
              <Button onClick={onDeleteBurn} size="xs" variant="destructive">
                Delete Burn
              </Button>
            )}
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
