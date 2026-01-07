import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import type { DownloadProgress } from "../types";

interface VideoOutputProps {
  className?: string;
  remoteStream: MediaStream | null;
  fallbackStream?: MediaStream | null;
  burnedVideoUrl?: string | null;
  isPipelineLoading?: boolean;
  isConnecting?: boolean;
  pipelineError?: string | null;
  isPlaying?: boolean;
  isDownloading?: boolean;
  downloadProgress?: DownloadProgress | null;
  pipelineNeedsModels?: string | null;
  isWaitingForFrames?: boolean;
  sourceVideoBlocked?: boolean;
  onResumeSourceVideo?: () => void;
  sam3Ta3mel?: boolean;
  sam3AutoPendiente?: boolean;
  estadoMascaraSam?: string | null;
  onVideoPlaying?: () => void;
  isBurning?: boolean;
}

export function VideoOutput({
  className = "",
  remoteStream,
  fallbackStream = null,
  burnedVideoUrl = null,
  isPipelineLoading = false,
  isConnecting = false,
  pipelineError: _pipelineError = null,
  isPlaying: _isPlaying = true,
  isDownloading = false,
  downloadProgress = null,
  pipelineNeedsModels = null,
  isWaitingForFrames = false,
  sourceVideoBlocked = false,
  onResumeSourceVideo,
  sam3Ta3mel = false,
  sam3AutoPendiente = false,
  estadoMascaraSam = null,
  onVideoPlaying,
  isBurning = false,
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [autoplayBlocked, setAutoplayBlocked] = useState(false);
  const [needsUserPlay, setNeedsUserPlay] = useState(false);
  const isMaskLoading = sam3Ta3mel || sam3AutoPendiente;

  const attemptPlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (isBurning && !burnedVideoUrl) return;
    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise
        .then(() => {
          setAutoplayBlocked(false);
          setNeedsUserPlay(false);
        })
        .catch(error => {
          console.warn("Video autoplay failed:", error);
          setAutoplayBlocked(true);
        });
    }
  };

  const handleStartSourceVideo = () => {
    onResumeSourceVideo?.();
    attemptPlay();
  };

  useEffect(() => {
    if (!videoRef.current) return;
    setNeedsUserPlay(false);

    if (burnedVideoUrl) {
      videoRef.current.srcObject = null;
      videoRef.current.src = burnedVideoUrl;
      videoRef.current.loop = true;
      videoRef.current.muted = true;
      videoRef.current.currentTime = 0;
      videoRef.current.load();
      attemptPlay();
      return;
    }

    if (isBurning) {
      videoRef.current.src = "";
      videoRef.current.srcObject = null;
      return;
    }

    videoRef.current.src = "";
    videoRef.current.srcObject = remoteStream || fallbackStream || null;
    attemptPlay();
  }, [remoteStream, fallbackStream, burnedVideoUrl, isBurning]);

  useEffect(() => {
    if (!remoteStream && !fallbackStream && !burnedVideoUrl) {
      setNeedsUserPlay(false);
      return;
    }

    const timeout = setTimeout(() => {
      const video = videoRef.current;
      if (!video) return;
      if (video.paused || video.ended) {
        setNeedsUserPlay(true);
      }
    }, 1200);

    return () => clearTimeout(timeout);
  }, [remoteStream, fallbackStream, burnedVideoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !remoteStream) return;

    const handlePlaying = () => {
      onVideoPlaying?.();
      setAutoplayBlocked(false);
      setNeedsUserPlay(false);
    };

    const handleLoadedData = () => {
      onVideoPlaying?.();
    };

    const handlePause = () => {
      setNeedsUserPlay(true);
    };

    if (!video.paused && video.currentTime > 0 && !video.ended) {
      setTimeout(() => onVideoPlaying?.(), 0);
    }

    video.addEventListener("playing", handlePlaying);
    video.addEventListener("loadeddata", handleLoadedData);
    video.addEventListener("pause", handlePause);
    return () => {
      video.removeEventListener("playing", handlePlaying);
      video.removeEventListener("loadeddata", handleLoadedData);
      video.removeEventListener("pause", handlePause);
    };
  }, [onVideoPlaying, remoteStream]);

  const hasVideo = Boolean(
    burnedVideoUrl || (!isBurning && (remoteStream || fallbackStream))
  );

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium text-white">Video Output</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex items-center justify-center min-h-0 p-4">
        {hasVideo ? (
          <div className="relative w-full h-full flex items-center justify-center">
            <video
              ref={videoRef}
              className="max-w-full max-h-full object-contain"
              autoPlay
              muted
              playsInline
            />
            {sourceVideoBlocked && !isMaskLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/40">
                <button
                  type="button"
                  onClick={handleStartSourceVideo}
                  className="mac-frosted-button px-4 py-2 text-sm text-white"
                >
                  Start Pipeline
                </button>
                <p className="text-xs text-muted-foreground">
                  Click to start the pipeline.
                </p>
              </div>
            ) : (autoplayBlocked || needsUserPlay) && !isMaskLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/40">
                <button
                  type="button"
                  onClick={attemptPlay}
                  className="mac-frosted-button px-4 py-2 text-sm text-white"
                >
                  Tap to Play
                </button>
                <p className="text-xs text-muted-foreground">
                  Browser paused autoplay. Click to resume.
                </p>
              </div>
            ) : null}
            {isDownloading || pipelineNeedsModels ? (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="text-center text-muted-foreground text-lg">
                  <Spinner size={24} className="mx-auto mb-3" />
                  <p>Downloading models...</p>
                  {downloadProgress?.current_artifact ? (
                    <p className="mt-2 text-xs text-muted-foreground">
                      {downloadProgress.current_artifact}
                    </p>
                  ) : null}
                  {typeof downloadProgress?.percentage === "number" ? (
                    <p className="mt-1 text-xs text-muted-foreground">
                      {Math.round(downloadProgress.percentage)}%
                    </p>
                  ) : null}
                </div>
              </div>
            ) : isWaitingForFrames && !burnedVideoUrl ? (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                <div className="text-center text-muted-foreground text-lg">
                  <Spinner size={24} className="mx-auto mb-3" />
                  <p>Warming up pipeline...</p>
                </div>
              </div>
          ) : sam3Ta3mel || sam3AutoPendiente ? (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <div className="text-center text-muted-foreground text-lg">
                <Spinner size={24} className="mx-auto mb-3" />
                <p>Generating SAM3 mask...</p>
                {estadoMascaraSam ? (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {estadoMascaraSam}
                  </p>
                ) : null}
              </div>
            </div>
          ) : isBurning ? (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-center text-muted-foreground text-lg">
                <Spinner size={24} className="mx-auto mb-3" />
                <p>Burning...</p>
              </div>
            </div>
          ) : null}
          </div>
        ) : isBurning ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Burning...</p>
          </div>
        ) : isDownloading || pipelineNeedsModels ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Downloading models...</p>
            {downloadProgress?.current_artifact ? (
              <p className="mt-2 text-xs text-muted-foreground">
                {downloadProgress.current_artifact}
              </p>
            ) : null}
            {typeof downloadProgress?.percentage === "number" ? (
              <p className="mt-1 text-xs text-muted-foreground">
                {Math.round(downloadProgress.percentage)}%
              </p>
            ) : null}
          </div>
        ) : isPipelineLoading ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Loading...</p>
          </div>
        ) : isConnecting ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Connecting...</p>
          </div>
        ) : (
          <div className="relative w-full h-full flex items-center justify-center text-muted-foreground text-sm">
            Waiting for video...
          </div>
        )}
      </CardContent>
    </Card>
  );
}
