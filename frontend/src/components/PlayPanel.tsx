import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import {
  downloadMP4P,
  loadMP4P,
  type MP4PData,
  type MP4PMetadata,
} from "../lib/mp4p-api";

type BurnVersionOption = {
  label: string;
  index: number;
};

function base64ToUrl(base64: string): string {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i += 1) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: "video/mp4" });
  return URL.createObjectURL(blob);
}

export function PlayPanel({
  className = "",
  onCreateBurnVersion,
}: {
  className?: string;
  onCreateBurnVersion?: (mp4pData: MP4PData) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mp4pFile, setMp4pFile] = useState<File | null>(null);
  const [mp4pData, setMp4pData] = useState<MP4PData | null>(null);
  const [metadata, setMetadata] = useState<MP4PMetadata | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedBurnIndex, setSelectedBurnIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [countdownText, setCountdownText] = useState<string | null>(null);
  const [showBurnedOverlay, setShowBurnedOverlay] = useState(false);

  const isExpired = metadata ? Date.now() >= metadata.expiresAt : false;

  const burnOptions = useMemo<BurnVersionOption[]>(() => {
    if (!metadata) return [];
    if (metadata.synthedVersions && metadata.synthedVersions.length > 0) {
      return metadata.synthedVersions.map((version, index) => ({
        index,
        label: version.promptsUsed?.join(", ") || `Burn ${index + 1}`,
      }));
    }
    if (metadata.promptsUsed) {
      return [{ index: 0, label: metadata.promptsUsed.join(", ") }];
    }
    return [];
  }, [metadata]);

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.volume = volume;
    videoRef.current.muted = isMuted;
  }, [volume, isMuted]);

  useEffect(() => {
    if (!metadata || !mp4pFile) return;
    if (metadata.expiresAt <= Date.now()) return;

    const timeUntilExpire = metadata.expiresAt - Date.now();
    const timer = window.setTimeout(() => {
      handleLoad(mp4pFile, selectedBurnIndex);
      setShowBurnedOverlay(true);
    }, timeUntilExpire);

    return () => window.clearTimeout(timer);
  }, [metadata, mp4pFile, selectedBurnIndex]);

  useEffect(() => {
    if (!showBurnedOverlay) return;
    const timer = window.setTimeout(() => {
      setShowBurnedOverlay(false);
    }, 4000);
    return () => window.clearTimeout(timer);
  }, [showBurnedOverlay]);

  useEffect(() => {
    if (!metadata) {
      setCountdownText(null);
      return;
    }

    const updateCountdown = () => {
      const now = Date.now();
      const diffMs = metadata.expiresAt - now;
      if (diffMs <= 0) {
        setCountdownText("Burn date passed");
        return;
      }

      const totalSeconds = Math.floor(diffMs / 1000);
      const days = Math.floor(totalSeconds / 86400);
      const hours = Math.floor((totalSeconds % 86400) / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = totalSeconds % 60;

      setCountdownText(
        `Burns in: ${days}d ${hours}h ${minutes}m ${seconds}s`
      );
    };

    updateCountdown();
    const timer = window.setInterval(updateCountdown, 1000);
    return () => window.clearInterval(timer);
  }, [metadata]);

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
      const fileText = await file.text();
      const parsed: MP4PData = JSON.parse(fileText);
      setMp4pData(parsed);

      const result = await loadMP4P(file, burnIndex);
      setMetadata(result.metadata);
      const expired = Date.now() >= result.metadata.expiresAt;
      const hasMultipleBurns = (result.metadata.synthedVersions?.length || 0) > 0;

      if (expired && result.showSynthed && burnIndex == null && hasMultipleBurns) {
        await handleLoad(file, 0);
        return;
      }

      if (result.showSynthed || !expired) {
        if (result.videoBase64) {
          const url = base64ToUrl(result.videoBase64);
          setVideoUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return url;
          });
        } else {
          setVideoUrl(null);
        }
      } else {
        // Expired but no synthed video: never show original.
        setVideoUrl(null);
      }

      if (result.showSynthed && (burnIndex != null || hasMultipleBurns)) {
        setSelectedBurnIndex(
          typeof result.selectedBurnIndex === "number"
            ? result.selectedBurnIndex
            : burnIndex ?? 0
        );
      } else {
        setSelectedBurnIndex(
          typeof result.selectedBurnIndex === "number" ? result.selectedBurnIndex : null
        );
      }
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Failed to load MP4P");
      setVideoUrl(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (file: File | null) => {
    if (!file) return;
    setMp4pFile(file);
    await handleLoad(file);
  };

  const handleSelectBurn = async (index: number) => {
    if (!mp4pFile) return;
    setSelectedBurnIndex(index);
    await handleLoad(mp4pFile, index);
  };

  const handleExport = async () => {
    if (!mp4pData || !mp4pFile) return;
    setIsExporting(true);
    try {
      const filename = mp4pFile.name.replace(/\.[^.]+$/, "");
      await downloadMP4P(mp4pData, filename);
    } finally {
      setIsExporting(false);
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
        <CardTitle className="text-base font-medium text-white">Play MP4P</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto flex-1 p-4">
        <div className="space-y-2">
          <input
            type="file"
            accept=".mp4p"
            className="hidden"
            id="mp4p-play-upload"
            onChange={(event) =>
              handleFileChange(event.target.files ? event.target.files[0] : null)
            }
          />
          <label
            htmlFor="mp4p-play-upload"
            className="mac-frosted-button px-3 py-2 text-xs inline-flex items-center justify-center cursor-pointer"
          >
            Load MP4P
          </label>
          {loadError && (
            <div className="text-xs text-red-500">{loadError}</div>
          )}
        </div>

        {countdownText && (
          <div className="text-xs">
            <div className="mac-translucent-ruby px-3 py-1 inline-flex items-center">
              {countdownText}
            </div>
          </div>
        )}

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
              onLoadedMetadata={(event) => {
                const element = event.currentTarget;
                setDuration(element.duration || 0);
              }}
              onTimeUpdate={(event) => {
                setCurrentTime(event.currentTarget.currentTime);
              }}
            />
          ) : (
            <div className="text-xs text-muted-foreground">
              {isExpired && metadata ? "No burned video available." : "No video loaded."}
            </div>
          )}
          {showBurnedOverlay && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="mac-translucent-ruby px-4 py-2 text-xs shadow-lg">
                Burned version unlocked
              </div>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <button className="mac-frosted-button px-3 py-1 text-xs" onClick={togglePlay} disabled={!videoUrl || isLoading}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button className="mac-frosted-button px-3 py-1 text-xs" onClick={stopVideo} disabled={!videoUrl}>
              Stop
            </button>
            <button className="mac-frosted-button px-3 py-1 text-xs" onClick={restartVideo} disabled={!videoUrl}>
              Restart
            </button>
            <button className="mac-frosted-button px-3 py-1 text-xs" onClick={() => skipBy(-5)} disabled={!videoUrl}>
              -5s
            </button>
            <button className="mac-frosted-button px-3 py-1 text-xs" onClick={() => skipBy(5)} disabled={!videoUrl}>
              +5s
            </button>
          </div>

          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={currentTime}
            onChange={(event) => {
              if (!videoRef.current) return;
              videoRef.current.currentTime = Number(event.target.value);
            }}
            className="mac-translucent-slider"
          />

          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-2">
              Volume
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={volume}
                onChange={(event) => setVolume(Number(event.target.value))}
                className="mac-translucent-slider"
              />
            </label>
            <button
              type="button"
              onClick={() => setIsMuted((prev) => !prev)}
              className="mac-frosted-button px-3 py-1 text-xs"
            >
              {isMuted ? "Unmute" : "Mute"}
            </button>
          </div>
        </div>

        {isExpired && burnOptions.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-medium">Burn versions</div>
            <div className="flex flex-wrap gap-2">
              {burnOptions.map((option) => (
                <button
                  key={option.index}
                  type="button"
                  onClick={() => handleSelectBurn(option.index)}
                  className={`mac-frosted-button px-3 py-1 text-xs ${
                    option.index === selectedBurnIndex ? "ring-2 ring-white/40" : ""
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {isExpired && (
          <div className="space-y-2">
            <div className="text-xs font-medium">Create burn version</div>
            <button
              type="button"
              className="mac-frosted-button px-3 py-1 text-xs"
              onClick={() => mp4pData && onCreateBurnVersion?.(mp4pData)}
              disabled={!mp4pData}
            >
              Create Burn
            </button>
          </div>
        )}

        <button
          type="button"
          onClick={handleExport}
          disabled={!mp4pData || isExporting}
          className="mac-frosted-button w-full px-4 py-2 text-sm disabled:opacity-50"
        >
          {isExporting ? "Exporting..." : "Export MP4P"}
        </button>
      </CardContent>
    </Card>
  );
}
