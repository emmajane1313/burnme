import { useState, useEffect, useRef } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { SettingsPanel } from "../components/SettingsPanel";
import { PlayPanel } from "../components/PlayPanel";
import { useWebRTC } from "../hooks/useWebRTC";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { usePipelines } from "../hooks/usePipelines";
import { useVideoRecorder } from "../hooks/useVideoRecorder";
import { getDefaultPromptForMode } from "../data/pipelines";
import { adjustResolutionForPipeline } from "../lib/utils";
import type {
  InputMode,
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  DownloadProgress,
} from "../types";
import type { PromptItem, PromptTransition } from "../lib/api";
import {
  checkModelStatus,
  downloadPipelineModels,
  startSam3MaskJob,
  getSam3MaskJob,
} from "../lib/api";
import { toast } from "sonner";
import { sendLoRAScaleUpdates } from "../utils/loraHelpers";

// Delay before resetting video reinitialization flag (ms)
// This allows useVideoSource to detect the flag change and trigger reinitialization
const VIDEO_REINITIALIZE_DELAY_MS = 100;
const MAX_VIDEO_WIDTH = 496;
const MAX_VIDEO_HEIGHT = 384;
const RESOLUTION_STEP = 16;
const MAX_DENOISING_STEPS = [1000, 750, 500, 250];
const HARD_NOISE_SCALE = 1.0;
const HARD_NOISE_CONTROLLER = false;

function fitResolutionToBounds(
  resolution: { width: number; height: number }
): { width: number; height: number } {
  const maxScale = Math.min(
    MAX_VIDEO_WIDTH / resolution.width,
    MAX_VIDEO_HEIGHT / resolution.height,
    1
  );

  let width = Math.floor((resolution.width * maxScale) / RESOLUTION_STEP) * RESOLUTION_STEP;
  let height = Math.floor((resolution.height * maxScale) / RESOLUTION_STEP) * RESOLUTION_STEP;

  if (width < RESOLUTION_STEP) width = RESOLUTION_STEP;
  if (height < RESOLUTION_STEP) height = RESOLUTION_STEP;

  return { width, height };
}

function buildLoRAParams(
  loras?: LoRAConfig[],
  strategy?: LoraMergeStrategy
): {
  loras?: { path: string; scale: number; merge_mode?: string }[];
  lora_merge_mode: string;
} {
  return {
    loras: loras?.map(({ path, scale, mergeMode }) => ({
      path,
      scale,
      ...(mergeMode && { merge_mode: mergeMode }),
    })),
    lora_merge_mode: strategy ?? "permanent_merge",
  };
}

function getVaceParams(
  refImages?: string[],
  vaceContextScale?: number
):
  | { vace_ref_images: string[]; vace_context_scale: number }
  | Record<string, never> {
  if (refImages && refImages.length > 0) {
    return {
      vace_ref_images: refImages,
      vace_context_scale: vaceContextScale ?? 1.0,
    };
  }
  return {};
}

interface StreamPageProps {
  videoControls?: React.ReactNode;
  onStatsChange?: (stats: { fps: number; bitrate: number }) => void;
}

export function StreamPage({ onStatsChange }: StreamPageProps = {}) {
  // Fetch available pipelines dynamically
  const { pipelines } = usePipelines();

  // Helper to get default mode for a pipeline
  const getPipelineDefaultMode = (_pipelineId: string): InputMode => {
    return "video";
  };

  // Use the stream state hook for settings management
  const {
    settings,
    updateSettings,
    getDefaults,
    spoutAvailable,
  } = useStreamState();

  // Prompt state - use unified default prompts based on mode
  const initialMode =
    settings.inputMode || getPipelineDefaultMode(settings.pipelineId);
  const [promptItems, setPromptItems] = useState<PromptItem[]>([
    { text: getDefaultPromptForMode(initialMode), weight: 100 },
  ]);
  const [interpolationMethod] = useState<"linear" | "slerp">("linear");

  // Track when we need to reinitialize video source
  const [shouldReinitializeVideo, setShouldReinitializeVideo] = useState(false);

  // Store custom video resolution from user uploads - persists across mode/pipeline changes
  const [customVideoResolution, setCustomVideoResolution] = useState<{
    width: number;
    height: number;
  } | null>(null);

  const [confirmedSynthedBlob, setConfirmedSynthedBlob] = useState<Blob | null>(
    null
  );
  const [isSynthCapturing, setIsSynthCapturing] = useState(false);
  const [synthEndPending, setSynthEndPending] = useState(false);
  const [synthLockedPrompt, setSynthLockedPrompt] = useState<string>("");
  const pendingSynthRef = useRef<{
    stream: MediaStream;
    prompt: string;
    pipelineId: PipelineId;
  } | null>(null);
  const [isWaitingForFrames, setIsWaitingForFrames] = useState(false);
  const [burnedVideoUrl, setBurnedVideoUrl] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"upload" | "play">("upload");
  const hideBurnSourcePreview = false;
  const [uploadedVideoFile, setUploadedVideoFile] = useState<File | null>(null);
  const [sam3MaskId, setSam3MaskId] = useState<string | null>(null);
  const sam3MaskMode: "inside" | "outside" = "inside";
  const [sam3Status, setSam3Status] = useState<string | null>(null);
  const [isSam3Generating, setIsSam3Generating] = useState(false);
  const [isSam3Downloading, setIsSam3Downloading] = useState(false);
  const [sam3AutoGenerated, setSam3AutoGenerated] = useState(false);
  const [sam3AutoFailed, setSam3AutoFailed] = useState(false);
  const startStreamInFlightRef = useRef(false);
  const isDebugEnabled =
    typeof window !== "undefined" &&
    window.localStorage.getItem("burn-debug") === "1";
  const debugLog = (...args: unknown[]) => {
    if (isDebugEnabled) {
      console.log("[burn-debug]", ...args);
    }
  };


  // Download state
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [pipelineNeedsModels, setPipelineNeedsModels] = useState<string | null>(
    null
  );


  // Pipeline management
  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
    pipelineInfo,
  } = usePipeline();

  // WebRTC for streaming
  const {
    remoteStream,
    isStreaming,
    isConnecting,
    peerConnectionRef,
    startStream,
    stopStream,
    updateVideoTrack,
    sendParameterUpdate,
    sendFrameMeta,
  } = useWebRTC({
    onServerVideoEnded: () => {
      stopRecording();
      sendParameterUpdate({ capture_mask_indices: false });
      stopStream();
      setSynthEndPending(true);
    },
  });
  const {
    isRecording: isRecordingSynthed,
    recordedBlob: recordedSynthedBlob,
    recordedFps: recordedSynthedFps,
    startRecording,
    stopRecording,
    resetRecording,
  } = useVideoRecorder();
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const autoUnpauseForSam3Ref = useRef(false);

  const isLoading = isDownloading || isPipelineLoading || isConnecting;

  const webrtcStats = useWebRTCStats({
    peerConnectionRef,
    isStreaming,
  });

  useEffect(() => {
    if (onStatsChange) {
      onStatsChange(webrtcStats);
    }
  }, [webrtcStats, onStatsChange]);

  // Video source for preview (camera or video)
  // Enable based on input mode, not pipeline category
  const [videoInputFps, setVideoInputFps] = useState<number | null>(null);
  const {
    localStream,
    isInitializing,
    error: videoSourceError,
    videoResolution,
    handleVideoFileUpload,
    restartVideoStream,
    sourceVideoBlocked,
    resumeSourceVideo,
  } = useVideoSource({
    onStreamUpdate: updateVideoTrack,
    onStopStream: stopStream,
    shouldReinitialize: shouldReinitializeVideo,
    enabled: settings.inputMode === "video",
    // Sync output resolution when user uploads a custom video
    // Store the custom resolution so it persists across mode/pipeline changes
    onCustomVideoResolution: resolution => {
      const fitted = fitResolutionToBounds(resolution);
      setCustomVideoResolution(fitted);
      updateSettings({
        resolution: { height: fitted.height, width: fitted.width },
      });
    },
    fpsOverride: videoInputFps,
    onFrameMeta: meta => {
      debugLog("FrameMeta send", meta);
      sendFrameMeta(meta);
    },
  });

  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const waitForSam3Models = async () => {
    setIsSam3Downloading(true);
    setSam3Status("Downloading SAM3 models...");

    const poll = async (resolve: () => void, reject: (error: Error) => void) => {
      try {
        const status = await checkModelStatus("sam3");
        if (status.progress) {
          setSam3Status(
            `Downloading SAM3 models... ${status.progress.percentage.toFixed(0)}%`
          );
        }

        if (status.downloaded) {
          setIsSam3Downloading(false);
          resolve();
          return;
        }

        setTimeout(() => poll(resolve, reject), 2000);
      } catch (error) {
        reject(error as Error);
      }
    };

    return new Promise<void>((resolve, reject) => {
      poll(resolve, reject);
    });
  };

  const handleGenerateSam3Mask = async () => {
    if (!uploadedVideoFile) {
      setSam3Status("Upload a video before generating masks.");
      return;
    }
    if (sam3AutoGenerated || sam3AutoFailed) {
      setSam3AutoGenerated(false);
      setSam3AutoFailed(false);
    }
    if (!videoResolution) {
      setSam3Status("Video not ready yet. Try again in a moment.");
      return;
    }

    setIsSam3Generating(true);
    setSam3Status("Preparing SAM3...");
    debugLog("SAM3: start mask generation", {
      fileName: uploadedVideoFile.name,
    });

    try {
      const status = await checkModelStatus("sam3");
      debugLog("SAM3: model status", status);
      if (!status.downloaded) {
        await downloadPipelineModels("sam3");
        await waitForSam3Models();
      }

      const base64 = await fileToBase64(uploadedVideoFile);
      const job = await startSam3MaskJob(base64, "", null, null);
      debugLog("SAM3: job started", job);
      setSam3Status("Generating SAM3 mask...");

      const poll = async (): Promise<void> => {
        const status = await getSam3MaskJob(job.jobId);
        if (status.status === "completed" && status.result) {
          debugLog("SAM3: mask generated", status.result);
          setSam3MaskId(status.result.maskId);
          setSam3Status("Mask ready.");
          setSam3AutoGenerated(true);
          setSam3AutoFailed(false);
          if (status.result.inputFps && status.result.inputFps > 0) {
            setVideoInputFps(status.result.inputFps);
            await restartVideoStream({ loop: true });
          }

          if (isStreaming) {
            debugLog("SAM3: sending mask to stream", {
              maskId: status.result.maskId,
              mode: sam3MaskMode,
            });
            sendParameterUpdate({
              sam3_mask_id: status.result.maskId,
              sam3_mask_mode: sam3MaskMode,
            });
          }
          setIsSam3Generating(false);
          return;
        }

        if (status.status === "failed") {
          throw new Error(status.error || "Mask generation failed.");
        }

        setTimeout(() => {
          poll().catch(error => {
            console.error("SAM3 mask generation failed:", error);
            const message =
              error instanceof Error ? error.message : "Mask generation failed.";
            setSam3Status(message);
            setSam3AutoFailed(true);
            toast.error("SAM3 Mask Error", {
              description: message,
            });
            setIsSam3Generating(false);
          });
        }, 2000);
      };

      poll().catch(error => {
        console.error("SAM3 mask generation failed:", error);
        const message =
          error instanceof Error ? error.message : "Mask generation failed.";
        setSam3Status(message);
        setSam3AutoFailed(true);
        toast.error("SAM3 Mask Error", {
          description: message,
        });
        setIsSam3Generating(false);
      });
    } catch (error) {
      console.error("SAM3 mask generation failed:", error);
      const message =
        error instanceof Error ? error.message : "Mask generation failed.";
      setSam3Status(message);
      setSam3AutoFailed(true);
      toast.error("SAM3 Mask Error", {
        description: message,
      });
      setIsSam3Generating(false);
    }
  };

  const handleTogglePause = () => {
    const nextPaused = !settings.paused;
    updateSettings({ paused: nextPaused });
    if (isStreaming) {
      sendParameterUpdate({ paused: nextPaused });
    }
  };

  const handleTransitionSubmit = (transition: PromptTransition) => {
    setPromptItems(transition.target_prompts);

    // Send transition to backend
    sendParameterUpdate({
      transition,
    });
  };

  const handlePipelineIdChange = (pipelineId: PipelineId) => {
    // Stop the stream if it's currently running
    if (isStreaming) {
      stopStream();
    }

    const modeToUse: InputMode = "video";
    const currentMode = settings.inputMode || "video";

    if (modeToUse === "video" && currentMode !== "video") {
      setShouldReinitializeVideo(true);
      setTimeout(
        () => setShouldReinitializeVideo(false),
        VIDEO_REINITIALIZE_DELAY_MS
      );
    }

    const defaults = getDefaults(pipelineId, modeToUse);

    // Update prompts to mode-specific defaults (unified per mode, not per pipeline)
    setPromptItems([{ text: getDefaultPromptForMode(modeToUse), weight: 100 }]);

    // Use custom video resolution if mode is video and one exists
    // This preserves the user's uploaded video resolution across pipeline switches
    const resolution =
      modeToUse === "video" && customVideoResolution
        ? customVideoResolution
        : { height: defaults.height, width: defaults.width };

    // Update the pipeline in settings with the appropriate mode and defaults
    updateSettings({
      pipelineId,
      inputMode: modeToUse,
      denoisingSteps: MAX_DENOISING_STEPS,
      resolution,
      noiseScale: defaults.noiseScale,
      noiseController: defaults.noiseController,
      loras: [], // Clear LoRA controls when switching pipelines
    });
  };

  const handleDownloadModels = async (pipelineId?: PipelineId) => {
    const pipelineIdToDownload =
      pipelineId || (pipelineNeedsModels as PipelineId | null);
    if (!pipelineIdToDownload || isDownloading) return;

    setPipelineNeedsModels(pipelineIdToDownload);
    setIsDownloading(true);
    setDownloadProgress(null);

    try {
      await downloadPipelineModels(pipelineIdToDownload);

      // Enhanced polling with progress updates
      const checkDownloadProgress = async () => {
        try {
          const status = await checkModelStatus(pipelineIdToDownload);

          // Update progress state
          if (status.progress) {
            setDownloadProgress(status.progress);
          }

          if (status.downloaded) {
            // Download complete
            setIsDownloading(false);
            setDownloadProgress(null);
            setPipelineNeedsModels(null);

            // Now update the pipeline since download is complete
            const pipelineId = pipelineIdToDownload;

            // Preserve the current input mode that the user selected before download
            // Only fall back to pipeline's default mode if no mode is currently set
            const newPipeline = pipelines?.[pipelineId];
            const currentMode =
              settings.inputMode || newPipeline?.defaultMode || "text";
            const defaults = getDefaults(pipelineId, currentMode);

            // Use custom video resolution if mode is video and one exists
            const resolution =
              currentMode === "video" && customVideoResolution
                ? customVideoResolution
                : { height: defaults.height, width: defaults.width };

            // Only update pipeline-related settings, preserving current input mode and prompts
            updateSettings({
              pipelineId,
              inputMode: currentMode,
              denoisingSteps: MAX_DENOISING_STEPS,
              resolution,
              noiseScale: defaults.noiseScale,
              noiseController: defaults.noiseController,
            });
            if (pendingSynthRef.current) {
              const pending = pendingSynthRef.current;
              pendingSynthRef.current = null;
              await handleStartStream(
                pending.pipelineId,
                [{ text: pending.prompt, weight: 100 }],
                pending.stream,
                false
              );
            }

          } else {
            setTimeout(checkDownloadProgress, 2000);
          }
        } catch (error) {
          console.error("Error checking download status:", error);
          setIsDownloading(false);
          setDownloadProgress(null);
        }
      };

      // Start checking
      setTimeout(checkDownloadProgress, 5000);
    } catch (error) {
      console.error("Error downloading models:", error);
      setIsDownloading(false);
      setDownloadProgress(null);
      setPipelineNeedsModels(null);
    }
  };

  const handleSeedChange = (seed: number) => {
    updateSettings({ seed });
  };

  const handleUploadVideoFile = async (file: File) => {
    setUploadedVideoFile(file);
    setVideoInputFps(null);
    setSam3MaskId(null);
    setSam3Status(null);
    setSam3AutoGenerated(false);
    setSam3AutoFailed(false);
    setConfirmedSynthedBlob(null);
    if (isStreaming) {
      autoUnpauseForSam3Ref.current = false;
      updateSettings({ paused: false });
      stopStream();
    }
    return handleVideoFileUpload(file);
  };

  const handleQuantizationChange = (quantization: "fp8_e4m3fn" | null) => {
    updateSettings({ quantization });
    // Note: This setting requires pipeline reload, so we don't send parameter update here
  };

  const handleKvCacheAttentionBiasChange = (bias: number) => {
    updateSettings({ kvCacheAttentionBias: bias });
    // Send KV cache attention bias update to backend
    sendParameterUpdate({
      kv_cache_attention_bias: bias,
    });
  };

  const handleLorasChange = (loras: LoRAConfig[]) => {
    updateSettings({ loras });

    // If streaming, send scale updates to backend for runtime adjustment
    if (isStreaming) {
      sendLoRAScaleUpdates(
        loras,
        pipelineInfo?.loaded_lora_adapters,
        ({ lora_scales }) => {
          // Forward only the lora_scales field over the data channel.
          sendParameterUpdate({
            // TypeScript doesn't know about lora_scales on this payload yet.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ...({ lora_scales } as any),
          });
        }
      );
    }
    // Note: Adding/removing LoRAs requires pipeline reload
  };


  const handleSpoutSenderChange = (
    spoutSender: { enabled: boolean; name: string } | undefined
  ) => {
    updateSettings({ spoutSender });
    // Send Spout output settings to backend
    if (isStreaming) {
      sendParameterUpdate({
        spout_sender: spoutSender,
      });
    }
  };

  const handleLivePromptSubmit = (prompts: PromptItem[]) => {
    // Also send the updated parameters to the backend immediately
    // Preserve the full blend while live
    sendParameterUpdate({
      prompts,
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: MAX_DENOISING_STEPS,
    });
  };

  useEffect(() => {
    remoteStreamRef.current = remoteStream;
  }, [remoteStream]);

  useEffect(() => {
    if (synthEndPending && recordedSynthedBlob) {
      setConfirmedSynthedBlob(recordedSynthedBlob);
      setSynthEndPending(false);
      setIsSynthCapturing(false);
      if (isStreaming) {
        sendParameterUpdate({ capture_mask_indices: false });
      }
    }
  }, [synthEndPending, recordedSynthedBlob, isStreaming, sendParameterUpdate]);

  useEffect(() => {
    if (!confirmedSynthedBlob) {
      setBurnedVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(confirmedSynthedBlob);
    setBurnedVideoUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [confirmedSynthedBlob]);

  useEffect(() => {
    if (!localStream) {
      return;
    }
    if (sourceVideoBlocked) {
      return;
    }
    if (isStreaming || isConnecting || isSynthCapturing || confirmedSynthedBlob) {
      return;
    }
    if (pipelineNeedsModels) {
      return;
    }
    if (uploadedVideoFile && !sam3MaskId) {
      return;
    }
    void handleStartStream();
  }, [
    localStream,
    sourceVideoBlocked,
    isStreaming,
    isConnecting,
    isSynthCapturing,
    pipelineNeedsModels,
    sam3MaskId,
    uploadedVideoFile,
  ]);

  useEffect(() => {
    if (!isStreaming || !sam3MaskId) {
      return;
    }
    debugLog("SAM3: resending mask on stream connect", {
      maskId: sam3MaskId,
      mode: sam3MaskMode,
    });
    sendParameterUpdate({
      sam3_mask_id: sam3MaskId,
      sam3_mask_mode: sam3MaskMode,
    });
  }, [isStreaming, sam3MaskId, sam3MaskMode, sendParameterUpdate]);

  useEffect(() => {
    if (!uploadedVideoFile || !videoResolution) {
      return;
    }
    if (
      isWaitingForFrames ||
      isSam3Generating ||
      sam3AutoGenerated ||
      sam3AutoFailed
    ) {
      return;
    }
    void handleGenerateSam3Mask();
  }, [
    uploadedVideoFile,
    videoResolution,
    isWaitingForFrames,
    isSam3Generating,
    sam3AutoGenerated,
    sam3AutoFailed,
  ]);

  useEffect(() => {
    if (!sam3MaskId || !isStreaming) {
      return;
    }
    if (!autoUnpauseForSam3Ref.current) {
      return;
    }
    if (settings.paused) {
      updateSettings({ paused: false });
      sendParameterUpdate({ paused: false });
    }
    autoUnpauseForSam3Ref.current = false;
  }, [sam3MaskId, isStreaming, settings.paused, sendParameterUpdate, updateSettings]);

  const sam3Ready = Boolean(sam3MaskId);
  const sam3AutoPending =
    Boolean(uploadedVideoFile) &&
    !isWaitingForFrames &&
    !sam3Ready &&
    !sam3AutoFailed;

  const handleStartSynth = async () => {
    const promptText = promptItems[0]?.text?.trim();
    if (!promptText) {
      return;
    }

    debugLog("Burn: start", { prompt: promptText });
    const serverVideoEnabled = Boolean(sam3MaskId);
    setSynthLockedPrompt(promptText);
    setIsSynthCapturing(true);
    setSynthEndPending(false);
    setConfirmedSynthedBlob(null);
    resetRecording();

    let restartedStream: MediaStream | null = null;
    if (serverVideoEnabled) {
      sendParameterUpdate({ server_video_reset: true, server_video_loop: false });
    } else {
      restartedStream = await restartVideoStream({
        loop: false,
        onEnded: () => {
          stopRecording();
          sendParameterUpdate({ capture_mask_indices: false });
          stopStream();
          setSynthEndPending(true);
        },
      });
      if (!restartedStream) {
        setIsSynthCapturing(false);
        return;
      }
    }

    onVideoPlayingCallbackRef.current = () => {
      const streamToRecord = remoteStreamRef.current;
      if (streamToRecord) {
        sendParameterUpdate({
          capture_mask_indices: true,
          capture_mask_reset: true,
        });
        startRecording(streamToRecord);
      }
    };

    const canReuseStream = isStreaming && remoteStreamRef.current;

    if (!canReuseStream) {
      const synthStarted = await handleStartStream(
        settings.pipelineId,
        [{ text: promptText, weight: 100 }],
        serverVideoEnabled ? null : restartedStream,
        false
      );

      if (!synthStarted && restartedStream) {
        pendingSynthRef.current = {
          stream: restartedStream,
          prompt: promptText,
          pipelineId: settings.pipelineId,
        };
      }
      return;
    }

    if (!serverVideoEnabled && restartedStream) {
      // Keep current session alive and just replace the input track
      const trackReplaced = await updateVideoTrack(restartedStream);
      if (!trackReplaced) {
        const synthStarted = await handleStartStream(
          settings.pipelineId,
          [{ text: promptText, weight: 100 }],
          restartedStream,
          false
        );

        if (!synthStarted && restartedStream) {
          pendingSynthRef.current = {
            stream: restartedStream,
            prompt: promptText,
            pipelineId: settings.pipelineId,
          };
        }
        return;
      }
    }

    // Send current prompt + parameters without resetting the session
    sendParameterUpdate({
      prompts: [{ text: promptText, weight: 100 }],
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: MAX_DENOISING_STEPS,
    });
  };

  const handleCancelSynth = async () => {
    debugLog("Burn: cancel");
    setSynthEndPending(false);
    setIsSynthCapturing(false);
    setSynthLockedPrompt("");
    pendingSynthRef.current = null;
    setIsWaitingForFrames(false);
    stopRecording();
    sendParameterUpdate({ capture_mask_indices: false });
    stopStream();
    if (sam3MaskId) {
      sendParameterUpdate({ server_video_reset: true, server_video_loop: true });
    } else {
      await restartVideoStream({ loop: true });
    }
  };

  const handleDeleteBurn = async () => {
    setConfirmedSynthedBlob(null);
    resetRecording();
    setSynthLockedPrompt("");
    sendParameterUpdate({ capture_mask_indices: false });
    await restartVideoStream({ loop: true });
    await handleStartStream();
  };

  // Clear prompts if pipeline doesn't support them
  useEffect(() => {
    const pipeline = pipelines?.[settings.pipelineId];
    if (pipeline?.supportsPrompts === false) {
      setPromptItems([{ text: "", weight: 1.0 }]);
    }
  }, [settings.pipelineId, pipelines]);

  // Ref to store callback that should execute when video starts playing
  const onVideoPlayingCallbackRef = useRef<(() => void) | null>(null);

  // Note: We intentionally do NOT auto-sync videoResolution to settings.resolution.
  // Mode defaults from the backend schema take precedence. Users can manually
  // adjust resolution if needed. This prevents the video source resolution from
  // overriding the carefully tuned per-mode defaults.

  const handleStartStream = async (
    overridePipelineId?: PipelineId,
    overridePrompts?: PromptItem[],
    overrideStream?: MediaStream | null,
    forcePaused?: boolean
  ): Promise<boolean> => {
    if (isStreaming) {
      stopStream();
      return true;
    }
    if (startStreamInFlightRef.current) {
      debugLog("Stream: start ignored (already starting)");
      return false;
    }
    startStreamInFlightRef.current = true;

    // Use override pipeline ID if provided, otherwise use current settings
    const pipelineIdToUse = overridePipelineId || settings.pipelineId;
    debugLog("Stream: start requested", {
      pipelineId: pipelineIdToUse,
      hasOverrideStream: Boolean(overrideStream),
      hasLocalStream: Boolean(localStream),
      inputMode: settings.inputMode,
      sam3MaskId,
    });

    try {
      setIsWaitingForFrames(true);
      // Check if models are needed but not downloaded
      const pipelineInfo = pipelines?.[pipelineIdToUse];
      if (pipelineInfo?.requiresModels) {
        try {
          const status = await checkModelStatus(pipelineIdToUse);
          if (!status.downloaded) {
            // Auto-download models and show progress in the video panel
            void handleDownloadModels(pipelineIdToUse);
            return false; // Stream did not start
          }
        } catch (error) {
          console.error("Error checking model status:", error);
          // Continue anyway if check fails
        }
      }

      // Always load pipeline with current parameters - backend will handle the rest
      console.log(`Loading ${pipelineIdToUse} pipeline...`);

      // Determine current input mode
      const currentMode =
        settings.inputMode || getPipelineDefaultMode(pipelineIdToUse) || "text";

      // Use settings.resolution if available, otherwise fall back to videoResolution
      let resolution = settings.resolution || videoResolution;

      // Adjust resolution to be divisible by required scale factor for the pipeline
      if (resolution) {
        const { resolution: adjustedResolution, wasAdjusted } =
          adjustResolutionForPipeline(pipelineIdToUse, resolution);

        if (wasAdjusted) {
          // Update settings with adjusted resolution
          updateSettings({ resolution: adjustedResolution });
          resolution = adjustedResolution;
        }
      }

      // Build load parameters dynamically based on pipeline capabilities and settings
      // The backend will use only the parameters it needs based on the pipeline schema
      const currentPipeline = pipelines?.[pipelineIdToUse];
      let loadParams: Record<string, unknown> | null = null;

      if (resolution) {
        // Start with common parameters
        loadParams = {
          height: resolution.height,
          width: resolution.width,
        };

        // Add seed if pipeline supports quantization (implies it needs seed)
        if (currentPipeline?.supportsQuantization) {
          loadParams.seed = settings.seed ?? 42;
          loadParams.quantization = settings.quantization ?? null;
        }

        // Add LoRA parameters if pipeline supports LoRA
        if (currentPipeline?.supportsLoRA && settings.loras) {
          const loraParams = buildLoRAParams(
            settings.loras,
            settings.loraMergeStrategy
          );
          loadParams = { ...loadParams, ...loraParams };
        }

        // Add VACE parameters if pipeline supports VACE
        if (currentPipeline?.supportsVACE) {
          const vaceEnabled = settings.vaceEnabled ?? currentMode !== "video";
          loadParams.vace_enabled = vaceEnabled;

          // Add VACE reference images if provided
          const vaceParams = getVaceParams(
            settings.refImages,
            settings.vaceContextScale
          );
          loadParams = { ...loadParams, ...vaceParams };
        }

        console.log(
          `Loading ${pipelineIdToUse} with resolution ${resolution.width}x${resolution.height}`,
          loadParams
        );
      }

      const loadSuccess = await loadPipeline(
        pipelineIdToUse,
        loadParams || undefined
      );
      if (!loadSuccess) {
        console.error("Failed to load pipeline, cannot start stream");
        setIsWaitingForFrames(false);
        debugLog("Stream: pipeline load failed", { pipelineId: pipelineIdToUse });
        return false;
      }
      debugLog("Stream: pipeline loaded", {
        pipelineId: pipelineIdToUse,
        resolution,
      });

      // Check video requirements based on input mode
      const serverVideoEnabled = Boolean(sam3MaskId);
      const needsVideoInput = !serverVideoEnabled;
      const isSpoutMode = false;

      // Only send video stream for pipelines that need video input (not in Spout mode)
      const streamToSend =
        needsVideoInput && !isSpoutMode
          ? overrideStream || localStream || undefined
          : undefined;

      if (needsVideoInput && !isSpoutMode && !localStream) {
        console.error("Video input required but no local stream available");
        setIsWaitingForFrames(false);
        debugLog("Stream: missing local stream for video input");
        return false;
      }

      // Build initial parameters based on pipeline type
      const initialParameters: {
        input_mode?: "text" | "video";
        prompts?: PromptItem[];
        prompt_interpolation_method?: "linear" | "slerp";
        denoising_step_list?: number[];
        noise_scale?: number;
        noise_controller?: boolean;
        manage_cache?: boolean;
        kv_cache_attention_bias?: number;
        paused?: boolean;
        spout_sender?: { enabled: boolean; name: string };
        spout_receiver?: { enabled: boolean; name: string };
        vace_ref_images?: string[];
        vace_context_scale?: number;
        sam3_mask_id?: string | null;
        sam3_mask_mode?: "inside" | "outside";
        server_video_source?: "sam3";
        server_video_mask_id?: string;
        server_video_loop?: boolean;
        capture_mask_indices?: boolean;
        capture_mask_reset?: boolean;
      } = {
        // Signal the intended input mode to the backend so it doesn't
        // briefly fall back to text mode before video frames arrive
        input_mode: currentMode,
      };

      // Common parameters for pipelines that support prompts
      if (pipelineInfo?.supportsPrompts !== false) {
        initialParameters.prompts = overridePrompts ?? promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.denoising_step_list = MAX_DENOISING_STEPS;
      }

      // KV cache bias for pipelines that support it
      if (pipelineInfo?.supportsKvCacheBias) {
        initialParameters.kv_cache_attention_bias =
          settings.kvCacheAttentionBias ?? 1.0;
      }

      // VACE-specific parameters - backend will ignore if not supported
      const vaceParams = getVaceParams(
        settings.refImages,
        settings.vaceContextScale
      );
      if ("vace_ref_images" in vaceParams) {
        initialParameters.vace_ref_images = vaceParams.vace_ref_images;
        initialParameters.vace_context_scale = vaceParams.vace_context_scale;
      }

      // Video mode parameters - hardcoded for maximum denoise behavior
      if (currentMode === "video") {
        initialParameters.noise_scale = HARD_NOISE_SCALE;
        initialParameters.noise_controller = HARD_NOISE_CONTROLLER;
      }

      // Spout settings - send if enabled
      if (settings.spoutSender?.enabled) {
        initialParameters.spout_sender = settings.spoutSender;
      }
      if (settings.spoutReceiver?.enabled) {
        initialParameters.spout_receiver = settings.spoutReceiver;
      }

      if (sam3MaskId) {
        initialParameters.sam3_mask_id = sam3MaskId;
        initialParameters.sam3_mask_mode = sam3MaskMode;
        initialParameters.server_video_source = "sam3";
        initialParameters.server_video_mask_id = sam3MaskId;
        initialParameters.server_video_loop = true;
      }

      if (isSynthCapturing) {
        initialParameters.capture_mask_indices = true;
        initialParameters.capture_mask_reset = true;
      }

      // Control paused state when starting a fresh stream
      if (forcePaused !== undefined) {
        autoUnpauseForSam3Ref.current = Boolean(forcePaused);
        updateSettings({ paused: forcePaused });
        initialParameters.paused = forcePaused;
      } else {
        updateSettings({ paused: false });
      }

      // Pipeline is loaded, now start WebRTC stream
      startStream(initialParameters, streamToSend);
      debugLog("Stream: startStream called", {
        pipelineId: pipelineIdToUse,
        hasStream: Boolean(streamToSend),
        initialParameters,
      });

      return true; // Stream started successfully
    } catch (error) {
      console.error("Error during stream start:", error);
      setIsWaitingForFrames(false);
      debugLog("Stream: start failed", error);
      return false;
    } finally {
      startStreamInFlightRef.current = false;
    }
  };

  return (
    <div className="h-full flex flex-col bg-transparent">
      <Header mode={viewMode} onModeChange={setViewMode} />

      {viewMode === "upload" ? (
        <>
          <div className="mac-upload-skin flex-1 flex relative px-2 md:px-4 py-4 overflow-y-auto justify-center items-start">
            <div className="flex relative gap-2 md:gap-4 w-full h-full max-w-[900px] flex-col md:flex-row">
              <div className="w-full md:w-64 h-full">
                <InputAndControlsPanel
                  className="h-full"
                  pipelines={pipelines}
                  localStream={localStream}
                  isInitializing={isInitializing}
                  error={videoSourceError}
                  isStreaming={isStreaming}
                  isConnecting={isConnecting}
                  isLoading={isLoading}
                  onVideoFileUpload={handleUploadVideoFile}
                  hideLocalPreview={hideBurnSourcePreview}
                  sourceVideoBlocked={sourceVideoBlocked}
                  pipelineId={settings.pipelineId}
                  seed={settings.seed ?? 42}
                  prompts={promptItems}
                  onPromptsChange={setPromptItems}
                  onTransitionSubmit={handleTransitionSubmit}
                  onLivePromptSubmit={handleLivePromptSubmit}
                  isVideoPaused={settings.paused}
                  confirmedSynthedBlob={confirmedSynthedBlob}
                  confirmedSynthedFps={recordedSynthedFps}
                  isRecordingSynthed={isRecordingSynthed}
                  isSynthCapturing={isSynthCapturing}
                  synthLockedPrompt={synthLockedPrompt}
                  onStartSynth={handleStartSynth}
                  onCancelSynth={handleCancelSynth}
                  onDeleteBurn={handleDeleteBurn}
                  onTogglePause={handleTogglePause}
                  sam3MaskId={sam3MaskId}
                  onSam3Generate={handleGenerateSam3Mask}
                  sam3Ready={sam3Ready}
                  sam3Status={
                    sam3Status ||
                    (isSam3Downloading ? "Downloading SAM3 models..." : null)
                  }
                  isSam3Generating={isSam3Generating}
                />
              </div>

              <div className="w-full md:w-[560px] h-full flex relative flex-col min-h-0">
                <div className="flex-1 relative min-h-0">
                  <VideoOutput
                    className="h-full"
                    remoteStream={remoteStream}
                    fallbackStream={localStream}
                    burnedVideoUrl={burnedVideoUrl}
                    isPipelineLoading={isPipelineLoading}
                    isConnecting={isConnecting}
                    pipelineError={pipelineError}
                    isPlaying={!settings.paused}
                    isDownloading={isDownloading}
                    downloadProgress={downloadProgress}
                    pipelineNeedsModels={pipelineNeedsModels}
                    isWaitingForFrames={isWaitingForFrames}
                    sourceVideoBlocked={sourceVideoBlocked}
                    onResumeSourceVideo={resumeSourceVideo}
                    isSam3Generating={isSam3Generating}
                    sam3AutoPending={sam3AutoPending}
                    sam3Status={sam3Status}
                    onVideoPlaying={() => {
                      setIsWaitingForFrames(false);
                      if (onVideoPlayingCallbackRef.current) {
                        onVideoPlayingCallbackRef.current();
                        onVideoPlayingCallbackRef.current = null;
                      }
                    }}
                  />
                </div>
              </div>

              <div className="w-full md:w-64 h-full">
                <SettingsPanel
                  className="h-full"
                  pipelines={pipelines}
                  pipelineId={settings.pipelineId}
                  onPipelineIdChange={handlePipelineIdChange}
                  isStreaming={isStreaming}
                  isLoading={isLoading}
                  seed={settings.seed ?? 42}
                  onSeedChange={handleSeedChange}
                  quantization={
                    settings.quantization !== undefined
                      ? settings.quantization
                      : "fp8_e4m3fn"
                  }
                  onQuantizationChange={handleQuantizationChange}
                  kvCacheAttentionBias={settings.kvCacheAttentionBias ?? 0.3}
                  onKvCacheAttentionBiasChange={handleKvCacheAttentionBiasChange}
                  loras={settings.loras || []}
                  onLorasChange={handleLorasChange}
                  loraMergeStrategy={settings.loraMergeStrategy ?? "permanent_merge"}
                  spoutSender={settings.spoutSender}
                  onSpoutSenderChange={handleSpoutSenderChange}
                  spoutAvailable={spoutAvailable}
                  isVideoPaused={settings.paused}
                />
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="flex-1 flex px-4 pb-4 min-h-0 overflow-hidden justify-center items-start">
          <div className="w-full max-w-[720px] h-full">
            <PlayPanel className="h-full" />
          </div>
        </div>
      )}
    </div>
  );
}
