import { useState, useEffect, useCallback, useRef } from "react";

export type VideoSourceMode = "video" | "camera" | "spout";

interface UseVideoSourceProps {
  onStreamUpdate?: (stream: MediaStream) => Promise<boolean>;
  onStopStream?: () => void;
  shouldReinitialize?: boolean;
  enabled?: boolean;
  fpsOverride?: number | null;
  onFrameMeta?: (meta: { time: number }) => void;
  // Called when a custom video is uploaded with its detected resolution
  onCustomVideoResolution?: (resolution: {
    width: number;
    height: number;
  }) => void;
}

// Standardized FPS for both video and camera modes
export const FPS = 15;
export const MIN_FPS = 5;
export const MAX_FPS = 30;

export function useVideoSource(props?: UseVideoSourceProps) {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoFile, setSelectedVideoFile] = useState<
    string | File | null
  >(null);
  const [videoResolution, setVideoResolution] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [sourceVideoBlocked, setSourceVideoBlocked] = useState(false);

  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const activeFps =
    props?.fpsOverride && props.fpsOverride > 0
      ? props.fpsOverride
      : FPS;

  const createVideoFromSource = useCallback(
    (videoSource: string | File, loop: boolean) => {
    const video = document.createElement("video");

    if (typeof videoSource === "string") {
      video.src = videoSource;
    } else {
      video.src = URL.createObjectURL(videoSource);
    }

    video.loop = loop;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = false;
    videoElementRef.current = video;
    return video;
  }, []);

  const createVideoFileStreamFromFile = useCallback(
    (
      videoSource: string | File,
      fps: number,
      options?: { loop?: boolean; onEnded?: () => void }
    ) => {
      const loop = options?.loop ?? true;
      const video = createVideoFromSource(videoSource, loop);
      setSourceVideoBlocked(true);

      return new Promise<{
        stream: MediaStream;
        resolution: { width: number; height: number };
      }>((resolve, reject) => {
        // Add timeout to prevent hanging promises
        const timeout = setTimeout(() => {
          reject(new Error("Video loading timeout"));
        }, 10000); // 10 second timeout

        video.onloadedmetadata = () => {
          clearTimeout(timeout);
          try {
            // Detect and store video resolution
            const detectedResolution = {
              width: video.videoWidth,
              height: video.videoHeight,
            };
            setVideoResolution(detectedResolution);

            // Create canvas matching the video resolution
            const canvas = document.createElement("canvas");
            canvas.width = detectedResolution.width;
            canvas.height = detectedResolution.height;
            const ctx = canvas.getContext("2d")!;

            const drawFrame = () => {
              if (video.readyState >= 2) {
                ctx.drawImage(
                  video,
                  0,
                  0,
                  detectedResolution.width,
                  detectedResolution.height
                );
                if (props?.onFrameMeta && !video.ended) {
                  props.onFrameMeta({ time: video.currentTime });
                }
              }
            };

            // Keep frames flowing even if the source is paused.
            const intervalId = window.setInterval(() => {
              drawFrame();
            }, Math.max(10, Math.floor(1000 / fps)));
            (video as HTMLVideoElement & { __burnDrawInterval?: number })
              .__burnDrawInterval = intervalId;

            video.onplay = () => {
              setSourceVideoBlocked(false);
            };
            video.onpause = () => {
              if (!video.ended) {
                setSourceVideoBlocked(true);
              }
            };
            video.onended = () => {
              window.clearInterval(intervalId);
            };

            if (options?.onEnded) {
              video.onended = options.onEnded;
            }

            // Capture stream from canvas at original resolution
            const stream = canvas.captureStream(fps);
            resolve({ stream, resolution: detectedResolution });

            // Explicit start required; no autoplay.
            setSourceVideoBlocked(true);
          } catch (error) {
            clearTimeout(timeout);
            reject(error);
          }
        };

        video.onerror = () => {
          clearTimeout(timeout);
          reject(new Error("Failed to load video file"));
        };
      });
    },
    [createVideoFromSource]
  );

  const createVideoFileStream = useCallback(
    async (fps: number, options?: { loop?: boolean; onEnded?: () => void }) => {
      if (!selectedVideoFile) {
        throw new Error("No video selected");
      }
      const result = await createVideoFileStreamFromFile(
        selectedVideoFile,
        fps,
        options
      );
      return result.stream;
    },
    [createVideoFileStreamFromFile, selectedVideoFile]
  );


  const handleVideoFileUpload = useCallback(
    async (file: File) => {
      // Validate file size (10MB limit)
      const maxSize = 10 * 1024 * 1024; // 10MB in bytes
      if (file.size > maxSize) {
        setError("File size must be less than 10MB");
        return false;
      }

      // Validate file type
      if (!file.type.startsWith("video/")) {
        setError("Please select a video file");
        return false;
      }

      setError(null);

      // Create new stream directly with the uploaded file (avoid race condition)
      try {
        setIsInitializing(true);
        const { stream: newStream, resolution } =
          await createVideoFileStreamFromFile(file, activeFps, { loop: true });

        // Try to update WebRTC track if streaming, otherwise just switch locally
        let trackReplaced = false;
        if (props?.onStreamUpdate) {
          trackReplaced = await props.onStreamUpdate(newStream);
        }

        // If track replacement failed and we're streaming, stop the stream
        // Otherwise, just switch locally
        if (!trackReplaced && props?.onStreamUpdate && props?.onStopStream) {
          // Track replacement failed - stop stream to allow clean switch
          props.onStopStream();
        }

        // Stop current stream only after successful replacement or if not streaming
        if (localStream && (trackReplaced || !props?.onStreamUpdate)) {
          localStream.getTracks().forEach(track => track.stop());
        }

        // Update selected video file only after successful stream creation
        setSelectedVideoFile(file);
        setLocalStream(newStream);
        setIsInitializing(false);

        // Notify about custom video resolution so caller can sync output resolution
        props?.onCustomVideoResolution?.(resolution);

        return true;
      } catch (error) {
        console.error("Failed to create stream from uploaded file:", error);
        setError("Failed to load uploaded video file");
        setIsInitializing(false);
        return false;
      }
    },
    [localStream, createVideoFileStreamFromFile, props, activeFps]
  );

  const stopVideo = useCallback(() => {
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
      setLocalStream(null);
    }

    if (videoElementRef.current) {
      const video = videoElementRef.current as HTMLVideoElement & {
        __burnDrawInterval?: number;
      };
      if (video.__burnDrawInterval) {
        window.clearInterval(video.__burnDrawInterval);
      }
      if ("cancelVideoFrameCallback" in video && (video as any).__burnVfcId) {
        try {
          (video as any).cancelVideoFrameCallback((video as any).__burnVfcId);
        } catch (error) {
          // Ignore callback cancel errors.
        }
      }
      video.pause();
      videoElementRef.current = null;
    }
    setSourceVideoBlocked(false);
  }, [localStream]);

  const reinitializeVideoSource = useCallback(async () => {
    setIsInitializing(true);
    setError(null);

    if (!selectedVideoFile) {
      setIsInitializing(false);
      return;
    }

    try {
      // Stop current stream if it exists
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }

      // Create new video file stream
      const stream = await createVideoFileStream(FPS, { loop: true });
      setLocalStream(stream);
    } catch (error) {
      console.error("Failed to reinitialize video source:", error);
      setError("Failed to load video");
    } finally {
      setIsInitializing(false);
    }
  }, [localStream, createVideoFileStream, selectedVideoFile]);

  // Initialize with video mode on mount (only if enabled)
  useEffect(() => {
    if (!props?.enabled) {
      // If not enabled, stop any existing stream and clear state
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        setLocalStream(null);
      }
      if (videoElementRef.current) {
        videoElementRef.current.pause();
        videoElementRef.current = null;
      }
      return;
    }

  const initializeVideoMode = async () => {
      if (!selectedVideoFile) {
        return;
      }
      setIsInitializing(true);
      try {
        const stream = await createVideoFileStream(activeFps, { loop: true });
        setLocalStream(stream);
      } catch (error) {
        console.error("Failed to create initial video file stream:", error);
        setError("Failed to load video");
      } finally {
        setIsInitializing(false);
      }
    };

    initializeVideoMode();

    // Cleanup on unmount
    return () => {
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
      if (videoElementRef.current) {
        videoElementRef.current.pause();
      }
    };
  }, [props?.enabled, createVideoFileStream, activeFps]); // eslint-disable-line react-hooks/exhaustive-deps

  const restartVideoStream = useCallback(
    async (
      options?: { loop?: boolean; onEnded?: () => void }
    ): Promise<MediaStream | null> => {
      if (!selectedVideoFile) {
        return null;
      }

      setIsInitializing(true);
      setError(null);

      try {
        const { stream: newStream, resolution } =
          await createVideoFileStreamFromFile(selectedVideoFile, activeFps, options);

        let trackReplaced = false;
        if (props?.onStreamUpdate) {
          trackReplaced = await props.onStreamUpdate(newStream);
        }

        if (!trackReplaced && props?.onStreamUpdate && props?.onStopStream) {
          props.onStopStream();
        }

        if (localStream && (trackReplaced || !props?.onStreamUpdate)) {
          localStream.getTracks().forEach(track => track.stop());
        }

        setLocalStream(newStream);
        props?.onCustomVideoResolution?.(resolution);

        return newStream;
      } catch (error) {
        console.error("Failed to restart video stream:", error);
        setError("Failed to load video");
        return null;
      } finally {
        setIsInitializing(false);
      }
    },
    [selectedVideoFile, createVideoFileStreamFromFile, localStream, props, activeFps]
  );

  // Handle reinitialization when shouldReinitialize changes
  useEffect(() => {
    if (props?.shouldReinitialize) {
      reinitializeVideoSource();
    }
  }, [props?.shouldReinitialize, reinitializeVideoSource]);

  return {
    localStream,
    isInitializing,
    error,
    videoResolution,
    stopVideo,
    handleVideoFileUpload,
    reinitializeVideoSource,
    restartVideoStream,
    sourceVideoBlocked,
    resumeSourceVideo: async () => {
      const video = videoElementRef.current;
      if (!video) return false;
      try {
        await video.play();
        setSourceVideoBlocked(false);
        return true;
      } catch (error) {
        setSourceVideoBlocked(true);
        return false;
      }
    },
  };
}
