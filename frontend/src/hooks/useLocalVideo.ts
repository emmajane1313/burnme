import { useState, useEffect, useCallback } from "react";
import { FPS, MIN_FPS, MAX_FPS } from "./useVideoSource";

export function useLocalVideo() {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<MediaDeviceInfo | null>(
    null
  );
  const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>(
    []
  );
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const requestDeviceAccess = useCallback(async () => {
    try {
      setError(null);

      // Request camera access - this will trigger browser's native device selection
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 512, min: 256, max: 512 },
          height: { ideal: 512, min: 256, max: 512 },
          frameRate: { ideal: FPS, min: MIN_FPS, max: MAX_FPS },
        },
        audio: false,
      });

      setLocalStream(stream);

      // Get device information after successful access
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        device => device.kind === "videoinput"
      );
      setAvailableDevices(videoDevices);

      // Find the currently selected device based on the stream's video track
      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack) {
        const currentDevice = videoDevices.find(
          device => device.deviceId === videoTrack.getSettings().deviceId
        );
        if (currentDevice) {
          setSelectedDevice(currentDevice);
        }
      }

      setIsInitializing(false);
      return stream;
    } catch (error) {
      console.error("Failed to request device access:", error);
      setError(
        error instanceof Error ? error.message : "Failed to access camera"
      );
      setIsInitializing(false);
      return null;
    }
  }, []);

  const switchDevice = useCallback(
    async (deviceId: string) => {
      try {
        setError(null);

        // Stop current stream
        if (localStream) {
          localStream.getTracks().forEach(track => track.stop());
        }

        // Create new stream with selected device
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: deviceId },
            width: { ideal: 512, min: 256, max: 512 },
            height: { ideal: 512, min: 256, max: 512 },
            frameRate: { ideal: FPS, min: MIN_FPS, max: MAX_FPS },
          },
          audio: false,
        });

        setLocalStream(stream);

        // Update selected device
        const device = availableDevices.find(d => d.deviceId === deviceId);
        if (device) {
          setSelectedDevice(device);
        }

        return stream;
      } catch (error) {
        console.error("Failed to switch device:", error);
        setError(
          error instanceof Error ? error.message : "Failed to switch camera"
        );
        return null;
      }
    },
    [localStream, availableDevices]
  );

  const stopVideo = useCallback(() => {
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
      setLocalStream(null);
    }
  }, [localStream]);

  // Initialize on mount
  useEffect(() => {
    requestDeviceAccess();

    // Cleanup on unmount
    return () => {
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [requestDeviceAccess]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    localStream,
    selectedDevice,
    availableDevices,
    isInitializing,
    error,
    switchDevice,
    stopVideo,
    requestDeviceAccess,
  };
}
