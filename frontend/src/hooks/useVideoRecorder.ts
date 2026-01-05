import { useCallback, useEffect, useRef, useState } from "react";

type RecorderState = {
  isRecording: boolean;
  recordedBlob: Blob | null;
  recordedFps: number | null;
  error: string | null;
};

const MIME_TYPE_CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
];

function pickSupportedMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") {
    return undefined;
  }

  return MIME_TYPE_CANDIDATES.find((type) =>
    MediaRecorder.isTypeSupported(type)
  );
}

export function useVideoRecorder() {
  const [state, setState] = useState<RecorderState>({
    isRecording: false,
    recordedBlob: null,
    recordedFps: null,
    error: null,
  });
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const startRecording = useCallback((stream: MediaStream) => {
    if (recorderRef.current || state.isRecording) {
      return;
    }

    const trackSettings = stream.getVideoTracks()[0]?.getSettings();
    const frameRate =
      typeof trackSettings?.frameRate === "number"
        ? trackSettings.frameRate
        : null;
    const mimeType = pickSupportedMimeType();
    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined
    );

    chunksRef.current = [];
    recorderRef.current = recorder;
    setState((prev) => ({
      ...prev,
      isRecording: true,
      recordedBlob: null,
      recordedFps: frameRate,
      error: null,
    }));

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onerror = () => {
      setState((prev) => ({
        ...prev,
        isRecording: false,
        error: "Recording failed",
      }));
    };

    recorder.onstop = () => {
      const recordedBlob = new Blob(chunksRef.current, {
        type: recorder.mimeType || mimeType || "video/webm",
      });
      recorderRef.current = null;
      chunksRef.current = [];
      setState((prev) => ({
        ...prev,
        isRecording: false,
        recordedBlob,
      }));
    };

    recorder.start(1000);
  }, [state.isRecording]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return;
    }
    recorder.stop();
  }, []);

  const resetRecording = useCallback(() => {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    chunksRef.current = [];
    setState({ isRecording: false, recordedBlob: null, recordedFps: null, error: null });
  }, []);

  useEffect(() => {
    return () => {
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.stop();
      }
      recorderRef.current = null;
      chunksRef.current = [];
    };
  }, []);

  return {
    isRecording: state.isRecording,
    recordedBlob: state.recordedBlob,
    recordedFps: state.recordedFps,
    error: state.error,
    startRecording,
    stopRecording,
    resetRecording,
  };
}
