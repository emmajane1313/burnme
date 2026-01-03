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
import { checkModelStatus, downloadPipelineModels } from "../lib/api";
import { sendLoRAScaleUpdates } from "../utils/loraHelpers";

// Delay before resetting video reinitialization flag (ms)
// This allows useVideoSource to detect the flag change and trigger reinitialization
const VIDEO_REINITIALIZE_DELAY_MS = 100;
const MAX_VIDEO_WIDTH = 496;
const MAX_VIDEO_HEIGHT = 384;
const RESOLUTION_STEP = 16;

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
    supportsNoiseControls,
    spoutAvailable,
  } = useStreamState();

  // Prompt state - use unified default prompts based on mode
  const initialMode =
    settings.inputMode || getPipelineDefaultMode(settings.pipelineId);
  const [promptItems, setPromptItems] = useState<PromptItem[]>([
    { text: getDefaultPromptForMode(initialMode), weight: 100 },
  ]);
  const [interpolationMethod, setInterpolationMethod] = useState<
    "linear" | "slerp"
  >("linear");
  const [temporalInterpolationMethod, setTemporalInterpolationMethod] =
    useState<"linear" | "slerp">("slerp");
  const [transitionSteps, setTransitionSteps] = useState(4);

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
  } = useWebRTC();
  const {
    isRecording: isRecordingSynthed,
    recordedBlob: recordedSynthedBlob,
    startRecording,
    stopRecording,
    resetRecording,
  } = useVideoRecorder();
  const remoteStreamRef = useRef<MediaStream | null>(null);

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
  const {
    localStream,
    isInitializing,
    error: videoSourceError,
    videoResolution,
    handleVideoFileUpload,
    restartVideoStream,
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
  });

  const handlePromptsSubmit = (prompts: PromptItem[]) => {
    setPromptItems(prompts);

    if (isStreaming) {
      sendParameterUpdate({
        prompts,
        prompt_interpolation_method: interpolationMethod,
        denoising_step_list: settings.denoisingSteps || [700, 500],
      });
    }
  };

  const handleSendPrompt = () => {
    const prompts = promptItems.length
      ? promptItems
      : [{ text: "", weight: 100 }];

    if (isStreaming) {
      const effectiveTransitionSteps =
        transitionSteps > 0 ? transitionSteps : 0;

      if (effectiveTransitionSteps > 0) {
        sendParameterUpdate({
          transition: {
            target_prompts: prompts,
            num_steps: effectiveTransitionSteps,
            temporal_interpolation_method: temporalInterpolationMethod,
          },
        });
      } else {
        sendParameterUpdate({
          prompts,
          prompt_interpolation_method: interpolationMethod,
          denoising_step_list: settings.denoisingSteps || [700, 500],
        });
      }
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
      denoisingSteps: defaults.denoisingSteps,
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
              denoisingSteps: defaults.denoisingSteps,
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
                pending.stream
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

  const handleDenoisingStepsChange = (denoisingSteps: number[]) => {
    updateSettings({ denoisingSteps });
    // Send denoising steps update to backend
    sendParameterUpdate({
      denoising_step_list: denoisingSteps,
    });
  };

  const handleNoiseScaleChange = (noiseScale: number) => {
    updateSettings({ noiseScale });
    // Send noise scale update to backend
    sendParameterUpdate({
      noise_scale: noiseScale,
    });
  };

  const handleNoiseControllerChange = (enabled: boolean) => {
    updateSettings({ noiseController: enabled });
    // Send noise controller update to backend
    sendParameterUpdate({
      noise_controller: enabled,
    });
  };

  const handleManageCacheChange = (enabled: boolean) => {
    updateSettings({ manageCache: enabled });
    // Send manage cache update to backend
    sendParameterUpdate({
      manage_cache: enabled,
    });
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


  const handleResetCache = () => {
    // Send reset cache command to backend
    sendParameterUpdate({
      reset_cache: true,
    });
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
      denoising_step_list: settings.denoisingSteps || [700, 500],
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
    }
  }, [synthEndPending, recordedSynthedBlob]);

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
    if (isStreaming || isConnecting || isSynthCapturing || confirmedSynthedBlob) {
      return;
    }
    if (pipelineNeedsModels) {
      return;
    }
    void handleStartStream();
  }, [localStream, isStreaming, isConnecting, isSynthCapturing, pipelineNeedsModels]);

  const handleStartSynth = async () => {
    const promptText = promptItems[0]?.text?.trim();
    if (!promptText) {
      return;
    }

    setSynthLockedPrompt(promptText);
    setIsSynthCapturing(true);
    setSynthEndPending(false);
    setConfirmedSynthedBlob(null);
    resetRecording();

    const restartedStream = await restartVideoStream({
      loop: false,
      onEnded: () => {
        stopRecording();
        stopStream();
        setSynthEndPending(true);
      },
    });

    if (!restartedStream) {
      setIsSynthCapturing(false);
      return;
    }

    onVideoPlayingCallbackRef.current = () => {
      const streamToRecord = remoteStreamRef.current;
      if (streamToRecord) {
        startRecording(streamToRecord);
      }
    };

    const canReuseStream = isStreaming && remoteStreamRef.current;

    if (!canReuseStream) {
      const synthStarted = await handleStartStream(
        settings.pipelineId,
        [{ text: promptText, weight: 100 }],
        restartedStream
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

    // Keep current session alive and just replace the input track
    const trackReplaced = await updateVideoTrack(restartedStream);
    if (!trackReplaced) {
      const synthStarted = await handleStartStream(
        settings.pipelineId,
        [{ text: promptText, weight: 100 }],
        restartedStream
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

    // Send current prompt + parameters without resetting the session
    sendParameterUpdate({
      prompts: [{ text: promptText, weight: 100 }],
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: settings.denoisingSteps || [700, 500],
    });
  };

  const handleCancelSynth = async () => {
    setSynthEndPending(false);
    setIsSynthCapturing(false);
    setSynthLockedPrompt("");
    pendingSynthRef.current = null;
    setIsWaitingForFrames(false);
    stopRecording();
    stopStream();
    await restartVideoStream({ loop: true });
  };

  const handleDeleteBurn = async () => {
    setConfirmedSynthedBlob(null);
    resetRecording();
    setSynthLockedPrompt("");
    await restartVideoStream({ loop: true });
    await handleStartStream();
  };

  // Update temporal interpolation defaults and clear prompts when pipeline changes
  useEffect(() => {
    const pipeline = pipelines?.[settings.pipelineId];
    if (pipeline) {
      const defaultMethod =
        pipeline.defaultTemporalInterpolationMethod || "slerp";
      const defaultSteps = pipeline.defaultTemporalInterpolationSteps ?? 4;

      setTemporalInterpolationMethod(defaultMethod);
      setTransitionSteps(defaultSteps);

      // Clear prompts if pipeline doesn't support them
      if (pipeline.supportsPrompts === false) {
        setPromptItems([{ text: "", weight: 1.0 }]);
      }
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
    overrideStream?: MediaStream | null
  ): Promise<boolean> => {
    if (isStreaming) {
      stopStream();
      return true;
    }

    // Use override pipeline ID if provided, otherwise use current settings
    const pipelineIdToUse = overridePipelineId || settings.pipelineId;

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
        return false;
      }

      // Check video requirements based on input mode
      const needsVideoInput = true;
      const isSpoutMode = false;

      // Only send video stream for pipelines that need video input (not in Spout mode)
      const streamToSend =
        needsVideoInput && !isSpoutMode
          ? overrideStream || localStream || undefined
          : undefined;

      if (needsVideoInput && !isSpoutMode && !localStream) {
        console.error("Video input required but no local stream available");
        setIsWaitingForFrames(false);
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
        spout_sender?: { enabled: boolean; name: string };
        spout_receiver?: { enabled: boolean; name: string };
        vace_ref_images?: string[];
        vace_context_scale?: number;
      } = {
        // Signal the intended input mode to the backend so it doesn't
        // briefly fall back to text mode before video frames arrive
        input_mode: currentMode,
      };

      // Common parameters for pipelines that support prompts
      if (pipelineInfo?.supportsPrompts !== false) {
        initialParameters.prompts = overridePrompts ?? promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.denoising_step_list = settings.denoisingSteps || [
          700, 500,
        ];
      }

      // Cache management for pipelines that support it
      if (pipelineInfo?.supportsCacheManagement) {
        initialParameters.manage_cache = settings.manageCache ?? true;
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

      // Video mode parameters - applies to all pipelines in video mode
      if (currentMode === "video") {
        initialParameters.noise_scale = settings.noiseScale ?? 0.7;
        initialParameters.noise_controller = settings.noiseController ?? true;
      }

      // Spout settings - send if enabled
      if (settings.spoutSender?.enabled) {
        initialParameters.spout_sender = settings.spoutSender;
      }
      if (settings.spoutReceiver?.enabled) {
        initialParameters.spout_receiver = settings.spoutReceiver;
      }

      // Reset paused state when starting a fresh stream
      updateSettings({ paused: false });

      // Pipeline is loaded, now start WebRTC stream
      startStream(initialParameters, streamToSend);

      return true; // Stream started successfully
    } catch (error) {
      console.error("Error during stream start:", error);
      setIsWaitingForFrames(false);
      return false;
    }
  };

  return (
    <div className="h-full flex flex-col bg-transparent">
      <Header mode={viewMode} onModeChange={setViewMode} />

      {viewMode === "upload" ? (
        <>
          <div className="flex-1 flex relative px-2 md:px-4 py-4 overflow-y-auto justify-center items-start">
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
                  onVideoFileUpload={handleVideoFileUpload}
                  pipelineId={settings.pipelineId}
                  prompts={promptItems}
                  onPromptsChange={setPromptItems}
                  onPromptsSubmit={handlePromptsSubmit}
                  onTransitionSubmit={handleTransitionSubmit}
                  interpolationMethod={interpolationMethod}
                  onInterpolationMethodChange={setInterpolationMethod}
                  temporalInterpolationMethod={temporalInterpolationMethod}
                  onTemporalInterpolationMethodChange={setTemporalInterpolationMethod}
                  onLivePromptSubmit={handleLivePromptSubmit}
                  isVideoPaused={settings.paused}
                  transitionSteps={transitionSteps}
                  onTransitionStepsChange={setTransitionSteps}
                  confirmedSynthedBlob={confirmedSynthedBlob}
                  isRecordingSynthed={isRecordingSynthed}
                  isSynthCapturing={isSynthCapturing}
                  synthLockedPrompt={synthLockedPrompt}
                  onStartSynth={handleStartSynth}
                  onCancelSynth={handleCancelSynth}
                  onDeleteBurn={handleDeleteBurn}
                  onPromptSend={handleSendPrompt}
                  onTogglePause={handleTogglePause}
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
                  denoisingSteps={
                    settings.denoisingSteps ||
                    getDefaults(settings.pipelineId, settings.inputMode)
                      .denoisingSteps || [750, 250]
                  }
                  onDenoisingStepsChange={handleDenoisingStepsChange}
                  defaultDenoisingSteps={
                    getDefaults(settings.pipelineId, settings.inputMode)
                      .denoisingSteps || [750, 250]
                  }
                  noiseScale={settings.noiseScale ?? 0.7}
                  onNoiseScaleChange={handleNoiseScaleChange}
                  noiseController={settings.noiseController ?? true}
                  onNoiseControllerChange={handleNoiseControllerChange}
                  manageCache={settings.manageCache ?? true}
                  onManageCacheChange={handleManageCacheChange}
                  quantization={
                    settings.quantization !== undefined
                      ? settings.quantization
                      : "fp8_e4m3fn"
                  }
                  onQuantizationChange={handleQuantizationChange}
                  kvCacheAttentionBias={settings.kvCacheAttentionBias ?? 0.3}
                  onKvCacheAttentionBiasChange={handleKvCacheAttentionBiasChange}
                  onResetCache={handleResetCache}
                  loras={settings.loras || []}
                  onLorasChange={handleLorasChange}
                  loraMergeStrategy={settings.loraMergeStrategy ?? "permanent_merge"}
                  inputMode={settings.inputMode}
                  supportsNoiseControls={supportsNoiseControls(settings.pipelineId)}
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
