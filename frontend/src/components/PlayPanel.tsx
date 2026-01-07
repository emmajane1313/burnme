import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import {
  loadMP4P,
  restoreMP4P,
  type MP4PData,
  type MP4PMetadata,
} from "../lib/mp4p-api";

function base64ToUrl(base64: string, mimeType: string): string {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i += 1) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: mimeType });
  return URL.createObjectURL(blob);
}

export function PlayPanel({ className = "" }: { className?: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mp4pData, setMp4pData] = useState<MP4PData | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [keyData, setKeyData] = useState<MP4PMetadata["visualCipher"] | null>(
    null
  );
  const [keyBurnIndex, setKeyBurnIndex] = useState<number | null>(null);
  const [isRestoring, setIsRestoring] = useState(false);
  const [restoreError, setRestoreError] = useState<string | null>(null);

  const formatTime = (seconds: number) => {
    if (!Number.isFinite(seconds)) return "0:00";
    const clamped = Math.max(0, seconds);
    const mins = Math.floor(clamped / 60);
    const secs = Math.floor(clamped % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.volume = volume;
    videoRef.current.muted = isMuted;
  }, [volume, isMuted]);

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const handleLoad = async (file: File, burnIndex?: number | null) => {
    setIsLoading(true);
    setLoadError(null);
    try {
      console.info("MP4P load start", {
        name: file.name,
        size: file.size,
        burnIndex: burnIndex ?? null,
      });
      const fileText = await file.text();
      const parsed: MP4PData = JSON.parse(fileText);
      setMp4pData(parsed);

      const result = await loadMP4P(file, burnIndex);
      console.info("MP4P load response", {
        burnIndex: result.selectedBurnIndex ?? null,
        mimeType: result.mimeType,
        hasVideo: Boolean(result.videoBase64),
      });
      if (result.videoBase64) {
        const url = base64ToUrl(
          result.videoBase64,
          result.mimeType || "video/mp4"
        );
        setVideoUrl(prev => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
      } else {
        setVideoUrl(null);
      }
    } catch (error) {
      setLoadError(
        error instanceof Error ? error.message : "Failed to load MP4P"
      );
      setVideoUrl(null);
      console.error("MP4P load failed", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (file: File | null) => {
    if (!file) return;
    await handleLoad(file);
  };

  const handleKeyFileChange = async (file: File | null) => {
    if (!file) return;
    setRestoreError(null);
    try {
      const fileText = await file.text();
      const parsed = JSON.parse(fileText);
      const visualCipher = parsed.visualCipher ?? parsed;
      if (
        !visualCipher?.prompt ||
        !visualCipher?.params ||
        visualCipher.seed === undefined ||
        !visualCipher.keyMaterial
      ) {
        throw new Error("Invalid key file");
      }
      setKeyData(visualCipher);
      setKeyBurnIndex(
        typeof parsed.burnIndex === "number" ? parsed.burnIndex : null
      );
    } catch (error) {
      setRestoreError(
        error instanceof Error ? error.message : "Invalid key file"
      );
      setKeyData(null);
      setKeyBurnIndex(null);
    }
  };

  const handleRestore = async () => {
    if (!mp4pData || !keyData) return;
    setIsRestoring(true);
    setRestoreError(null);
    try {
      const result = await restoreMP4P(mp4pData, keyData, keyBurnIndex);
      const url = base64ToUrl(
        result.videoBase64,
        result.mimeType || "video/webm"
      );
      setVideoUrl(prev => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    } catch (error) {
      setRestoreError(
        error instanceof Error ? error.message : "Decrypt failed"
      );
    } finally {
      setIsRestoring(false);
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
    } else {
      videoRef.current.pause();
    }
  };

  const stopVideo = () => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    videoRef.current.currentTime = 0;
  };

  const restartVideo = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play();
  };

  const skipBy = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(
      0,
      Math.min(videoRef.current.currentTime + seconds, duration || 0)
    );
  };

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <CardTitle className="text-base font-medium text-white">
          Play MP4P
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto flex-1 p-4">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="file"
              accept=".mp4p"
              className="hidden"
              id="mp4p-play-upload"
              onChange={event =>
                handleFileChange(
                  event.target.files ? event.target.files[0] : null
                )
              }
            />
            <label
              htmlFor="mp4p-play-upload"
              className="mac-frosted-button px-3 py-2 text-xs inline-flex items-center justify-center cursor-pointer"
            >
              Load MP4P
            </label>
            <input
              type="file"
              accept=".json"
              onChange={event =>
                handleKeyFileChange(event.target.files?.[0] ?? null)
              }
              className="hidden"
              id="mp4p-key-upload"
            />
            <label
              htmlFor="mp4p-key-upload"
              className="mac-frosted-button px-3 py-2 text-xs inline-flex items-center justify-center cursor-pointer"
            >
              Load Key File
            </label>
          </div>
          {loadError && <div className="text-xs text-red-500">{loadError}</div>}
        </div>

        <div className="w-full aspect-video bg-black/40 rounded flex items-center justify-center overflow-hidden relative">
          {isLoading ? (
            <div className="text-center text-xs text-muted-foreground">
              <Spinner size={22} className="mx-auto mb-2" />
              Loading MP4P...
            </div>
          ) : videoUrl ? (
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full h-full object-contain"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onLoadedMetadata={event => {
                const element = event.currentTarget;
                setDuration(element.duration || 0);
              }}
              onTimeUpdate={event => {
                setCurrentTime(event.currentTarget.currentTime);
              }}
            />
          ) : (
            <div className="text-xs text-muted-foreground">
              No video loaded.
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <button
              className="mac-frosted-button px-3 py-1 text-xs"
              onClick={togglePlay}
              disabled={!videoUrl || isLoading}
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="mac-frosted-button px-3 py-1 text-xs"
              onClick={stopVideo}
              disabled={!videoUrl}
            >
              Stop
            </button>
            <button
              className="mac-frosted-button px-3 py-1 text-xs"
              onClick={restartVideo}
              disabled={!videoUrl}
            >
              Restart
            </button>
            <button
              className="mac-frosted-button px-3 py-1 text-xs"
              onClick={() => skipBy(-5)}
              disabled={!videoUrl}
            >
              -5s
            </button>
            <button
              className="mac-frosted-button px-3 py-1 text-xs"
              onClick={() => skipBy(5)}
              disabled={!videoUrl}
            >
              +5s
            </button>
          </div>

          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={currentTime}
            onChange={event => {
              if (!videoRef.current) return;
              videoRef.current.currentTime = Number(event.target.value);
            }}
            className="mac-translucent-slider"
          />
          <div className="flex items-center justify-between text-[11px] text-muted-foreground">
            <span>0:00</span>
            <span>{formatTime(duration)}</span>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-2">
              Volume
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={volume}
                onChange={event => setVolume(Number(event.target.value))}
                className="mac-translucent-slider"
              />
            </label>
            <button
              type="button"
              onClick={() => setIsMuted(prev => !prev)}
              className="mac-frosted-button px-3 py-1 text-xs"
            >
              {isMuted ? "Unmute" : "Mute"}
            </button>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs font-medium">Unlock</div>
          {restoreError && (
            <div className="text-xs text-red-500">{restoreError}</div>
          )}
          <button
            type="button"
            onClick={handleRestore}
            disabled={!mp4pData || !keyData || isRestoring}
            className="mac-frosted-button w-full px-4 py-2 text-sm disabled:opacity-50"
          >
            {isRestoring ? "Decrypting..." : "Decrypt with Key"}
          </button>
        </div>
      </CardContent>
    </Card>
  );
}
