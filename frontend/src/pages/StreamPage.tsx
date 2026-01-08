import { useState, useEffect, useRef, useCallback } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { PlayPanel } from "../components/PlayPanel";
import { AboutPanel } from "../components/AboutPanel";
import { useWebRTC } from "../hooks/useWebRTC";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { usePipelines } from "../hooks/usePipelines";
import { useVideoRecorder } from "../hooks/useVideoRecorder";
import { getDefaultPromptForMode } from "../data/pipelines";
import { adjustResolutionForPipeline } from "../lib/utils";
import type { InputMode, PipelineId, DownloadProgress } from "../types";
import type { PromptItem, PromptTransition } from "../lib/api";
import {
  checkModelStatus,
  downloadPipelineModels,
  renderServerBurn,
  startSam3MaskJob,
  getSam3MaskJob,
  uploadAsset,
} from "../lib/api";
import { base64ToBlob } from "../lib/mp4p-api";
import { toast } from "sonner";
import { useI18n } from "../i18n";

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
  const { t } = useI18n();
  const { pipelines } = usePipelines();

  const getSam3ErrorMessage = (error: unknown) => {
    const rawMessage = error instanceof Error ? error.message : "";
    const lower = rawMessage.toLowerCase();
    if (lower.includes("no masks") || lower.includes("no detections")) {
      return { rawMessage, displayMessage: t("sam3.error.noDetections") };
    }
    if (!rawMessage) {
      return {
        rawMessage: t("sam3.error.generic"),
        displayMessage: t("sam3.error.generic"),
      };
    }
    if (rawMessage === "Mask generation failed.") {
      return { rawMessage, displayMessage: t("sam3.error.generic") };
    }
    return { rawMessage, displayMessage: rawMessage };
  };

  const getPipelineDefaultMode = (_pipelineId: string): InputMode => {
    return "video";
  };

  const {
    settings,
    updateSettings,
    getDefaults,
  } = useStreamState();

  const initialMode =
    settings.inputMode || getPipelineDefaultMode(settings.pipelineId);
  const [promptItems, setPromptItems] = useState<PromptItem[]>([
    { text: getDefaultPromptForMode(initialMode), weight: 100 },
  ]);
  const [interpolationMethod] = useState<"linear" | "slerp">("linear");

  const [shouldReinitializeVideo, setShouldReinitializeVideo] = useState(false);

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
    seed: number;
  } | null>(null);
  const [isWaitingForFrames, setIsWaitingForFrames] = useState(false);
  const [burnedVideoUrl, setBurnedVideoUrl] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"upload" | "play" | "about">("upload");
  const hideBurnSourcePreview = false;
  const [uploadedVideoFile, setUploadedVideoFile] = useState<File | null>(null);
  const [idMascaraSam, setIdMascaraSam] = useState<string | null>(null);
  const modoMascaraSam: "inside" | "outside" = "inside";
  const [estadoMascaraSam, setEstadoMascaraSam] = useState<string | null>(null);
  const [rutaAssetSam, setRutaAssetSam] = useState<string | null>(null);
  const [sam3Ta3mel, setSam3Ta3mel] = useState(false);
  const [sam3Tanzil, setSam3Tanzil] = useState(false);
  const [sam3AutoListo, setSam3AutoListo] = useState(false);
  const [sam3AutoFallo, setSam3AutoFallo] = useState(false);
  const [sam3SinDetecciones, setSam3SinDetecciones] = useState(false);
  const [cajaSamPromptActiva, setCajaSamPromptActiva] = useState(false);
  const [cajaSam, setCajaSam] = useState<[number, number, number, number] | null>(
    null
  );
  const startStreamInFlightRef = useRef(false);
  const debugLog = (...args: unknown[]) => {
    console.log("[burn-debug]", ...args);
  };
  const [serverSynthedFps, setServerSynthedFps] = useState<number | null>(null);
  const serverRenderAbortRef = useRef(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [pipelineNeedsModels, setPipelineNeedsModels] = useState<string | null>(
    null
  );


  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
  } = usePipeline();

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
    onCaptureResetDone: _ => {
   
      awaitingCaptureResetRef.current = false;
      if (captureResetInFlightRef.current) {
        maybeStartRecording();
      }
    },
    onServerVideoResetDone: () => {
      awaitingServerResetRef.current = false;
      if (captureResetInFlightRef.current) {
        maybeStartRecording();
      }
    },
    onServerVideoStartReady: _ => {
      awaitingServerStartRef.current = false;
    },
    onCaptureStartReady: maskId => {
      debugLog("Capture start ready", { maskId });
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
  const pendingRecordStreamRef = useRef<MediaStream | null>(null);
  const awaitingCaptureResetRef = useRef(false);
  const awaitingServerResetRef = useRef(false);
  const awaitingServerStartRef = useRef(false);
  const pendingRecordStartRef = useRef(false);
  const captureResetInFlightRef = useRef(false);
  const serverStartUnpauseSentRef = useRef(false);

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
    onCustomVideoResolution: resolution => {
      const fitted = fitResolutionToBounds(resolution);
      setCustomVideoResolution(fitted);
      updateSettings({
        resolution: { height: fitted.height, width: fitted.width },
      });
    },
    fpsOverride: videoInputFps,
    onFrameMeta: meta => {
      sendFrameMeta(meta);
    },
  });

  const waitForSam3Models = async () => {
    setSam3Tanzil(true);
    setEstadoMascaraSam(t("sam3.status.downloading"));

    const poll = async (resolve: () => void, reject: (error: Error) => void) => {
      try {
        const status = await checkModelStatus("sam3");
        if (status.progress) {
          setEstadoMascaraSam(
            t("sam3.status.downloadingProgress", {
              percentage: status.progress.percentage.toFixed(0),
            })
          );
        }

        if (status.downloaded) {
          setSam3Tanzil(false);
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
      setEstadoMascaraSam(t("sam3.status.uploadBefore"));
      return;
    }
    if (sam3AutoListo || sam3AutoFallo) {
      setSam3AutoListo(false);
      setSam3AutoFallo(false);
    }
    setSam3SinDetecciones(false);
    const boxToUse = cajaSamPromptActiva ? cajaSam : null;
    const targetResolution = settings.resolution || videoResolution;
    const targetWidth = targetResolution?.width ?? null;
    const targetHeight = targetResolution?.height ?? null;
    if (cajaSamPromptActiva && !boxToUse) {
      setEstadoMascaraSam(t("sam3.status.drawBoxFirst"));
      return;
    }
    if (!videoResolution) {
      setEstadoMascaraSam(t("sam3.status.videoNotReady"));
      return;
    }

    setSam3Ta3mel(true);
    setEstadoMascaraSam(t("sam3.status.preparing"));
   
    try {
      const status = await checkModelStatus("sam3");
 
      if (!status.downloaded) {
        await downloadPipelineModels("sam3");
        await waitForSam3Models();
      }

      let assetPath = rutaAssetSam;
      if (!assetPath) {
        const asset = await uploadAsset(uploadedVideoFile);
        assetPath = asset.path;
        setRutaAssetSam(assetPath);
      }
      const job = await startSam3MaskJob(
        null,
        assetPath,
        "",
        boxToUse,
        null,
        targetWidth,
        targetHeight
      );
      setEstadoMascaraSam(t("sam3.status.generating"));

      const poll = async (): Promise<void> => {
        const status = await getSam3MaskJob(job.jobId);
        if (status.status === "completed" && status.result) {
       
          setIdMascaraSam(status.result.maskId);
          setEstadoMascaraSam(t("sam3.status.ready"));
          setSam3AutoListo(true);
          setSam3AutoFallo(false);
          setSam3SinDetecciones(false);
          setCajaSamPromptActiva(false);
          setCajaSam(null);
          if (status.result.inputFps && status.result.inputFps > 0) {
            setVideoInputFps(status.result.inputFps);
            await restartVideoStream({ loop: true });
          }

          if (isStreaming) {
        
            sendParameterUpdate({
              sam3_mask_id: status.result.maskId,
              sam3_mask_mode: modoMascaraSam,
            });
          }
          setSam3Ta3mel(false);
          return;
        }

        if (status.status === "failed") {
          throw new Error(status.error || t("sam3.error.generic"));
        }

        setTimeout(() => {
          poll().catch(error => {
            console.error("SAM3 mask generation failed:", error);
            const { rawMessage, displayMessage } = getSam3ErrorMessage(error);
            setEstadoMascaraSam(displayMessage);
            if (
              rawMessage.toLowerCase().includes("no masks") ||
              rawMessage.toLowerCase().includes("no detections")
            ) {
              setSam3SinDetecciones(true);
            }
            setSam3AutoFallo(true);
            toast.error(t("sam3.error.title"), {
              description: displayMessage,
            });
            setSam3Ta3mel(false);
          });
        }, 2000);
      };

      poll().catch(error => {
        console.error("SAM3 mask generation failed:", error);
        const { rawMessage, displayMessage } = getSam3ErrorMessage(error);
        setEstadoMascaraSam(displayMessage);
        if (
          rawMessage.toLowerCase().includes("no masks") ||
          rawMessage.toLowerCase().includes("no detections")
        ) {
          setSam3SinDetecciones(true);
        }
        setSam3AutoFallo(true);
        toast.error(t("sam3.error.title"), {
          description: displayMessage,
        });
        setSam3Ta3mel(false);
      });
    } catch (error) {
      console.error("SAM3 mask generation failed:", error);
      const { rawMessage, displayMessage } = getSam3ErrorMessage(error);
      setEstadoMascaraSam(displayMessage);
      if (
        rawMessage.toLowerCase().includes("no masks") ||
        rawMessage.toLowerCase().includes("no detections")
      ) {
        setSam3SinDetecciones(true);
      }
      setSam3AutoFallo(true);
      toast.error(t("sam3.error.title"), {
        description: displayMessage,
      });
      setSam3Ta3mel(false);
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

    sendParameterUpdate({
      transition,
    });
  };

  const handlePipelineIdChange = (pipelineId: PipelineId) => {
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

    setPromptItems([{ text: getDefaultPromptForMode(modeToUse), weight: 100 }]);

    const resolution =
      modeToUse === "video" && customVideoResolution
        ? customVideoResolution
        : { height: defaults.height, width: defaults.width };

    updateSettings({
      pipelineId,
      inputMode: modeToUse,
      denoisingSteps: MAX_DENOISING_STEPS,
      resolution,
      noiseScale: defaults.noiseScale,
      noiseController: defaults.noiseController,
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

      const checkDownloadProgress = async () => {
        try {
          const status = await checkModelStatus(pipelineIdToDownload);

          if (status.progress) {
            setDownloadProgress(status.progress);
          }

          if (status.downloaded) {
            setIsDownloading(false);
            setDownloadProgress(null);
            setPipelineNeedsModels(null);

            const pipelineId = pipelineIdToDownload;

            const newPipeline = pipelines?.[pipelineId];
            const currentMode =
              settings.inputMode || newPipeline?.defaultMode || "text";
            const defaults = getDefaults(pipelineId, currentMode);

            const resolution =
              currentMode === "video" && customVideoResolution
                ? customVideoResolution
                : { height: defaults.height, width: defaults.width };

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
                false,
                pending.seed,
                true
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

      setTimeout(checkDownloadProgress, 5000);
    } catch (error) {
      console.error("Error downloading models:", error);
      setIsDownloading(false);
      setDownloadProgress(null);
      setPipelineNeedsModels(null);
    }
  };

  const generateRandomSeed = () => {
    if (globalThis.crypto?.getRandomValues) {
      const value = new Uint32Array(1);
      globalThis.crypto.getRandomValues(value);
      return value[0] & 0x7fffffff;
    }
    return Math.floor(Math.random() * 2147483648);
  };

  const handleUploadVideoFile = async (file: File) => {
    setUploadedVideoFile(file);
    setVideoInputFps(null);
    setIdMascaraSam(null);
    setEstadoMascaraSam(null);
    setRutaAssetSam(null);
    setSam3AutoListo(false);
    setSam3AutoFallo(false);
    setSam3SinDetecciones(false);
    setCajaSamPromptActiva(false);
    setCajaSam(null);
    setConfirmedSynthedBlob(null);
    if (isStreaming) {
      autoUnpauseForSam3Ref.current = false;
      updateSettings({ paused: false });
      stopStream();
    }
    return handleVideoFileUpload(file);
  };

  const handleLivePromptSubmit = (prompts: PromptItem[]) => {
    sendParameterUpdate({
      prompts,
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: MAX_DENOISING_STEPS,
    });
  };

  const maybeStartRecording = useCallback(() => {
   
    if (!pendingRecordStartRef.current) {
      return;
    }
    if (awaitingCaptureResetRef.current || awaitingServerResetRef.current) {
     
      return;
    }
    const streamToRecord = pendingRecordStreamRef.current || remoteStreamRef.current;
    if (!streamToRecord) {
      return;
    }
    pendingRecordStartRef.current = false;
    pendingRecordStreamRef.current = null;
    captureResetInFlightRef.current = false;
   
    startRecording(streamToRecord);
    if (awaitingServerStartRef.current && !serverStartUnpauseSentRef.current) {
      serverStartUnpauseSentRef.current = true;
      sendParameterUpdate({ server_video_pause: false });
    
    }
  }, [startRecording, sendParameterUpdate, isStreaming]);

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
    if (uploadedVideoFile && !idMascaraSam) {
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
    idMascaraSam,
    uploadedVideoFile,
  ]);

  useEffect(() => {
    if (!isStreaming || !idMascaraSam) {
      return;
    }
   
    sendParameterUpdate({
      sam3_mask_id: idMascaraSam,
      sam3_mask_mode: modoMascaraSam,
    });
  }, [isStreaming, idMascaraSam, modoMascaraSam, sendParameterUpdate]);

  useEffect(() => {
    if (!uploadedVideoFile || !videoResolution) {
      return;
    }
    if (
      isWaitingForFrames ||
      sam3Ta3mel ||
      sam3AutoListo ||
      sam3AutoFallo
    ) {
      return;
    }
    void handleGenerateSam3Mask();
  }, [
    uploadedVideoFile,
    videoResolution,
    isWaitingForFrames,
    sam3Ta3mel,
    sam3AutoListo,
    sam3AutoFallo,
  ]);

  useEffect(() => {
    if (!idMascaraSam || !isStreaming) {
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
  }, [idMascaraSam, isStreaming, settings.paused, sendParameterUpdate, updateSettings]);

  const sam3Listo = Boolean(idMascaraSam);
  const sam3AutoPendiente =
    Boolean(uploadedVideoFile) &&
    !isWaitingForFrames &&
    !sam3Listo &&
    !sam3AutoFallo;

  const handleStartSynth = async () => {
    const promptText = promptItems[0]?.text?.trim();
    if (!promptText) {
      return;
    }

    const synthSeed = generateRandomSeed();
    updateSettings({ seed: synthSeed });

    const serverVideoEnabled = Boolean(idMascaraSam);
    setSynthLockedPrompt(promptText);
    setIsSynthCapturing(true);
    setSynthEndPending(false);
    setConfirmedSynthedBlob(null);
    setServerSynthedFps(null);
    resetRecording();

    let restartedStream: MediaStream | null = null;
    if (serverVideoEnabled) {
      serverRenderAbortRef.current = false;
      if (isStreaming) {
        stopStream();
      }
      try {
        if (!idMascaraSam) {
          throw new Error(t("burn.missingSam3Mask"));
        }
        const renderParams = {
          prompts: [{ text: promptText, weight: 100 }],
          prompt_interpolation_method: interpolationMethod,
          denoising_step_list: MAX_DENOISING_STEPS,
          noise_scale: settings.noiseScale,
          noise_controller: settings.noiseController,
          kv_cache_attention_bias: settings.kvCacheAttentionBias,
          sam3_mask_id: idMascaraSam,
          sam3_mask_mode: modoMascaraSam,
          input_mode: "video" as const,
        };
        const runRender = async () => {
          const resolution = settings.resolution || videoResolution;
          const renderResult = await renderServerBurn({
            pipelineId: settings.pipelineId,
            maskId: idMascaraSam,
            params: renderParams,
            loadParams: {
              default_lora_enabled: true,
              seed: synthSeed,
              height: resolution?.height,
              width: resolution?.width,
            },
            outputMimeType: "video/webm",
            capture_mask_reset: true,
          });
          if (serverRenderAbortRef.current) {
            return;
          }
          const outputBlob = base64ToBlob(
            renderResult.videoBase64,
            renderResult.mimeType || "video/mp4"
          );
          setConfirmedSynthedBlob(outputBlob);
          setServerSynthedFps(renderResult.fps ?? null);
        };

        await runRender();
      } catch (error) {
        console.error("Server burn render failed:", error);
        toast.error(t("burn.serverErrorTitle"), {
          description:
            error instanceof Error ? error.message : t("burn.serverErrorFallback"),
        });
      } finally {
        setIsSynthCapturing(false);
      }
      return;
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
      if (serverVideoEnabled) {
        return;
      }
      awaitingCaptureResetRef.current = true;
      awaitingServerResetRef.current = false;
      awaitingServerStartRef.current = false;
      pendingRecordStartRef.current = true;
      pendingRecordStreamRef.current = remoteStreamRef.current;
      captureResetInFlightRef.current = true;
      serverStartUnpauseSentRef.current = false;
      sendParameterUpdate({
        capture_mask_reset: true,
        capture_mask_indices: true,
      });
      maybeStartRecording();
    };

    const canReuseStream = false;

    if (!canReuseStream) {
      const synthStarted = await handleStartStream(
        settings.pipelineId,
        [{ text: promptText, weight: 100 }],
        serverVideoEnabled ? null : restartedStream,
        false,
        synthSeed,
        true
      );

      if (!synthStarted && restartedStream) {
        pendingSynthRef.current = {
          stream: restartedStream,
          prompt: promptText,
          pipelineId: settings.pipelineId,
          seed: synthSeed,
        };
      }
      return;
    }

    if (!serverVideoEnabled && restartedStream) {
      const trackReplaced = await updateVideoTrack(restartedStream);
      if (!trackReplaced) {
        const synthStarted = await handleStartStream(
          settings.pipelineId,
          [{ text: promptText, weight: 100 }],
          restartedStream,
          false,
          synthSeed,
          true
        );

        if (!synthStarted && restartedStream) {
          pendingSynthRef.current = {
            stream: restartedStream,
            prompt: promptText,
            pipelineId: settings.pipelineId,
            seed: synthSeed,
          };
        }
        return;
      }
    }

    sendParameterUpdate({
      prompts: [{ text: promptText, weight: 100 }],
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: MAX_DENOISING_STEPS,
    });
  };

  const handleCancelSynth = async () => {

    serverRenderAbortRef.current = true;
    setSynthEndPending(false);
    setIsSynthCapturing(false);
    setSynthLockedPrompt("");
    setConfirmedSynthedBlob(null);
    setServerSynthedFps(null);
    captureResetInFlightRef.current = false;
    pendingRecordStartRef.current = false;
    awaitingCaptureResetRef.current = false;
    awaitingServerResetRef.current = false;
    awaitingServerStartRef.current = false;
    serverStartUnpauseSentRef.current = false;
    pendingSynthRef.current = null;
    setIsWaitingForFrames(false);
    stopRecording();
    sendParameterUpdate({ capture_mask_indices: false });
    stopStream();
    if (idMascaraSam) {
      sendParameterUpdate({ server_video_reset: true, server_video_loop: true });
    } else {
      await restartVideoStream({ loop: true });
    }
  };

  const handleDeleteBurn = async () => {
    setConfirmedSynthedBlob(null);
    setServerSynthedFps(null);
    resetRecording();
    setSynthLockedPrompt("");
    captureResetInFlightRef.current = false;
    pendingRecordStartRef.current = false;
    awaitingCaptureResetRef.current = false;
    awaitingServerResetRef.current = false;
    awaitingServerStartRef.current = false;
    serverStartUnpauseSentRef.current = false;
    sendParameterUpdate({ capture_mask_indices: false });
    await restartVideoStream({ loop: true });
    await handleStartStream();
  };

  useEffect(() => {
    const pipeline = pipelines?.[settings.pipelineId];
    if (pipeline?.supportsPrompts === false) {
      setPromptItems([{ text: "", weight: 1.0 }]);
    }
  }, [settings.pipelineId, pipelines]);

  const onVideoPlayingCallbackRef = useRef<(() => void) | null>(null);

  const handleStartStream = async (
    overridePipelineId?: PipelineId,
    overridePrompts?: PromptItem[],
    overrideStream?: MediaStream | null,
    forcePaused?: boolean,
    overrideSeed?: number,
    forceRestart = false
  ): Promise<boolean> => {
    if (isStreaming) {
      stopStream();
      if (!forceRestart) {
        return true;
      }
    }
    if (startStreamInFlightRef.current) {
      return false;
    }
    startStreamInFlightRef.current = true;

    const pipelineIdToUse = overridePipelineId || settings.pipelineId;
   

    try {
      setIsWaitingForFrames(true);
      const pipelineInfo = pipelines?.[pipelineIdToUse];
      if (pipelineInfo?.requiresModels) {
        try {
          const status = await checkModelStatus(pipelineIdToUse);
          if (!status.downloaded) {
            void handleDownloadModels(pipelineIdToUse);
            return false;
          }
        } catch (error) {
          console.error("Error checking model status:", error);
        }
      }

      console.log(`Loading ${pipelineIdToUse} pipeline...`);

      const currentMode =
        settings.inputMode || getPipelineDefaultMode(pipelineIdToUse) || "text";

      let resolution = settings.resolution || videoResolution;

      if (resolution) {
        const { resolution: adjustedResolution, wasAdjusted } =
          adjustResolutionForPipeline(pipelineIdToUse, resolution);

        if (wasAdjusted) {
          updateSettings({ resolution: adjustedResolution });
          resolution = adjustedResolution;
        }
      }

      const currentPipeline = pipelines?.[pipelineIdToUse];
      let loadParams: Record<string, unknown> | null = null;

      if (resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
        };

        if (currentPipeline?.supportsQuantization) {
          loadParams.seed = overrideSeed ?? settings.seed ?? 42;
        }

        if (currentPipeline?.supportsVACE) {
          const vaceEnabled = settings.vaceEnabled ?? currentMode !== "video";
          loadParams.vace_enabled = vaceEnabled;

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

      loadParams = loadParams || {};
      loadParams.default_lora_enabled = true;

      const loadSuccess = await loadPipeline(
        pipelineIdToUse,
        loadParams || undefined
      );
      if (!loadSuccess) {
        console.error("Failed to load pipeline, cannot start stream");
        setIsWaitingForFrames(false);
        return false;
      }
    

      const serverVideoEnabled = Boolean(idMascaraSam);
      const needsVideoInput = !serverVideoEnabled;
      const isSpoutMode = false;

      const streamToSend =
        needsVideoInput && !isSpoutMode
          ? overrideStream || localStream || undefined
          : undefined;

      if (needsVideoInput && !isSpoutMode && !localStream) {
        console.error("Video input required but no local stream available");
        setIsWaitingForFrames(false);
        return false;
      }

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
        input_mode: currentMode,
      };

      if (pipelineInfo?.supportsPrompts !== false) {
        initialParameters.prompts = overridePrompts ?? promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.denoising_step_list = MAX_DENOISING_STEPS;
      }

      if (pipelineInfo?.supportsKvCacheBias) {
        initialParameters.kv_cache_attention_bias =
          settings.kvCacheAttentionBias ?? 1.0;
      }

      const vaceParams = getVaceParams(
        settings.refImages,
        settings.vaceContextScale
      );
      if ("vace_ref_images" in vaceParams) {
        initialParameters.vace_ref_images = vaceParams.vace_ref_images;
        initialParameters.vace_context_scale = vaceParams.vace_context_scale;
      }

      if (currentMode === "video") {
        initialParameters.noise_scale = HARD_NOISE_SCALE;
        initialParameters.noise_controller = HARD_NOISE_CONTROLLER;
      }

      if (settings.spoutSender?.enabled) {
        initialParameters.spout_sender = settings.spoutSender;
      }
      if (settings.spoutReceiver?.enabled) {
        initialParameters.spout_receiver = settings.spoutReceiver;
      }

      if (idMascaraSam) {
        initialParameters.sam3_mask_id = idMascaraSam;
        initialParameters.sam3_mask_mode = modoMascaraSam;
        initialParameters.server_video_source = "sam3";
        initialParameters.server_video_mask_id = idMascaraSam;
        initialParameters.server_video_loop = true;
      }

      if (isSynthCapturing) {
        initialParameters.capture_mask_indices = true;
        initialParameters.capture_mask_reset = true;
      }

      if (forcePaused !== undefined) {
        autoUnpauseForSam3Ref.current = Boolean(forcePaused);
        updateSettings({ paused: forcePaused });
        initialParameters.paused = forcePaused;
      } else {
        updateSettings({ paused: false });
      }

      startStream(initialParameters, streamToSend);
     

      return true;
    } catch (error) {
      console.error("Error during stream start:", error);
      setIsWaitingForFrames(false);
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
            <div className="flex relative gap-2 md:gap-4 w-full h-full max-w-[1100px] flex-col md:flex-row">
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
                  onPipelineIdChange={handlePipelineIdChange}
                  seed={settings.seed ?? 42}
                  prompts={promptItems}
                  onPromptsChange={setPromptItems}
                  onTransitionSubmit={handleTransitionSubmit}
                  onLivePromptSubmit={handleLivePromptSubmit}
                  isVideoPaused={settings.paused}
                  confirmedSynthedBlob={confirmedSynthedBlob}
                  confirmedSynthedFps={serverSynthedFps ?? recordedSynthedFps}
                  isRecordingSynthed={isRecordingSynthed}
                  isSynthCapturing={isSynthCapturing}
                  synthLockedPrompt={synthLockedPrompt}
                  onStartSynth={handleStartSynth}
                  onCancelSynth={handleCancelSynth}
                  onDeleteBurn={handleDeleteBurn}
                  onTogglePause={handleTogglePause}
                  idMascaraSam={idMascaraSam}
                  onGenerarMascara={handleGenerateSam3Mask}
                  sam3SinDetecciones={sam3SinDetecciones}
                  cajaSamPromptActiva={cajaSamPromptActiva}
                  cajaSam={cajaSam}
                  onCajaSamChange={setCajaSam}
                  onCajaSamPromptActiva={() => {
                    setCajaSamPromptActiva(true);
                    setCajaSam(null);
                    void resumeSourceVideo();
                  }}
                  onCajaSamPromptCancelar={() => {
                    setCajaSamPromptActiva(false);
                    setCajaSam(null);
                  }}
                  sam3Listo={sam3Listo}
                  estadoMascaraSam={
                    estadoMascaraSam ||
                    (sam3Tanzil ? t("sam3.status.downloading") : null)
                  }
                  sam3Ta3mel={sam3Ta3mel}
                />
              </div>

              <div className="w-full md:flex-1 h-full flex relative flex-col min-h-0">
                <div className="flex-1 relative min-h-0">
                  <VideoOutput
                    className="h-full"
                    remoteStream={remoteStream}
                    fallbackStream={isSynthCapturing ? null : localStream}
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
                    sam3Ta3mel={sam3Ta3mel}
                    sam3AutoPendiente={sam3AutoPendiente}
                    estadoMascaraSam={estadoMascaraSam}
                    isBurning={isSynthCapturing}
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
            </div>
          </div>
        </>
      ) : viewMode === "play" ? (
        <div className="flex-1 flex px-4 pb-4 min-h-0 overflow-hidden justify-center items-start">
          <div className="w-full max-w-[720px] h-full">
            <PlayPanel className="h-full" />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex px-4 pb-4 min-h-0 overflow-hidden justify-center items-start">
          <div className="w-full max-w-[720px] h-full">
            <AboutPanel className="h-full" />
          </div>
        </div>
      )}
    </div>
  );
}
