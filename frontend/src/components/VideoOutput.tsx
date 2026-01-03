import { useEffect, useRef } from "react";
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
  onVideoPlaying?: () => void;
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
  onVideoPlaying,
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!videoRef.current) return;

    if (burnedVideoUrl) {
      videoRef.current.srcObject = null;
      videoRef.current.src = burnedVideoUrl;
      videoRef.current.loop = true;
      return;
    }

    videoRef.current.src = "";
    videoRef.current.srcObject = remoteStream || fallbackStream || null;
  }, [remoteStream, fallbackStream, burnedVideoUrl]);

  // Listen for video playing event to notify parent
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !remoteStream) return;

    const handlePlaying = () => {
      onVideoPlaying?.();
    };

    // Check if video is already playing when effect runs
    // This handles cases where the video was already playing before the callback was set
    if (!video.paused && video.currentTime > 0 && !video.ended) {
      // Use setTimeout to avoid calling during render
      setTimeout(() => onVideoPlaying?.(), 0);
    }

    video.addEventListener("playing", handlePlaying);
    return () => {
      video.removeEventListener("playing", handlePlaying);
    };
  }, [onVideoPlaying, remoteStream]);

  // No manual play/pause handling in auto-loop mode.

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium text-white">Video Output</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex items-center justify-center min-h-0 p-4">
        {burnedVideoUrl || remoteStream || fallbackStream ? (
          <div className="relative w-full h-full flex items-center justify-center">
            <video
              ref={videoRef}
              className="max-w-full max-h-full object-contain"
              autoPlay
              muted
              playsInline
            />
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
            ) : isWaitingForFrames ? (
              <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                <div className="text-center text-muted-foreground text-lg">
                  <Spinner size={24} className="mx-auto mb-3" />
                  <p>Warming up pipeline...</p>
                </div>
              </div>
            ) : null}
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
