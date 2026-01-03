import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { PipelineInfo } from "../types";
import { PromptInput } from "./PromptInput";
import { TimelinePromptEditor } from "./TimelinePromptEditor";
import type { TimelinePrompt } from "./PromptTimeline";
import { Button } from "./ui/button";
import {
  loadMP4P,
  type MP4PMetadata,
  encryptVideo,
  downloadMP4P,
  addSynthedVideo,
} from "../lib/mp4p-api";

interface InputAndControlsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  isStreaming: boolean;
  isConnecting: boolean;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  pipelineId: string;
  prompts: PromptItem[];
  onPromptsChange: (prompts: PromptItem[]) => void;
  onPromptsSubmit: (prompts: PromptItem[]) => void;
  onTransitionSubmit: (transition: PromptTransition) => void;
  interpolationMethod: "linear" | "slerp";
  onInterpolationMethodChange: (method: "linear" | "slerp") => void;
  temporalInterpolationMethod: "linear" | "slerp";
  onTemporalInterpolationMethodChange: (method: "linear" | "slerp") => void;
  isLive?: boolean;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  selectedTimelinePrompt?: TimelinePrompt | null;
  onTimelinePromptUpdate?: (prompt: TimelinePrompt) => void;
  isVideoPaused?: boolean;
  isTimelinePlaying?: boolean;
  currentTime?: number;
  timelinePrompts?: TimelinePrompt[];
  transitionSteps: number;
  onTransitionStepsChange: (steps: number) => void;
  recordedSynthedBlob: Blob | null;
  confirmedSynthedBlob: Blob | null;
  isRecordingSynthed: boolean;
  isSynthCapturing: boolean;
  synthLockedPrompt: string;
  onStartSynth: () => void;
  onCancelSynth: () => void;
}

export function InputAndControlsPanel({
  className = "",
  pipelines,
  localStream,
  isInitializing,
  error,
  isStreaming,
  isConnecting,
  onVideoFileUpload,
  pipelineId,
  prompts,
  onPromptsChange,
  interpolationMethod,
  onInterpolationMethodChange,
  temporalInterpolationMethod,
  onTemporalInterpolationMethodChange,
  isLive = false,
  onLivePromptSubmit,
  selectedTimelinePrompt = null,
  onTimelinePromptUpdate,
  isVideoPaused = false,
  isTimelinePlaying: _isTimelinePlaying = false,
  currentTime: _currentTime = 0,
  timelinePrompts: _timelinePrompts = [],
  transitionSteps,
  onTransitionStepsChange,
  recordedSynthedBlob,
  confirmedSynthedBlob,
  isRecordingSynthed,
  isSynthCapturing,
  synthLockedPrompt,
  onStartSynth,
  onCancelSynth,
}: InputAndControlsPanelProps) {
  const [burnDate, setBurnDate] = useState<string>("");
  const [burnTime, setBurnTime] = useState<string>("");
  const [burnDateTimestamp, setBurnDateTimestamp] = useState<number | null>(
    null
  );
  const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
  const [isTimePickerOpen, setIsTimePickerOpen] = useState(false);
  const [timeSelectionTouched, setTimeSelectionTouched] = useState(false);
  const [calendarMonth, setCalendarMonth] = useState<Date>(() => {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), 1);
  });
  const [timeSelection, setTimeSelection] = useState<{
    hour: number | null;
    minute: number | null;
    second: number | null;
  }>({ hour: null, minute: null, second: null });
  const dateButtonRef = useRef<HTMLButtonElement>(null);
  const timeButtonRef = useRef<HTMLButtonElement>(null);
  const datePickerRef = useRef<HTMLDivElement>(null);
  const timePickerRef = useRef<HTMLDivElement>(null);
  const [mp4pMetadata, setMp4pMetadata] = useState<MP4PMetadata | null>(null);
  const [isMP4PMode, setIsMP4PMode] = useState(false);
  const [uploadedVideoFile, setUploadedVideoFile] = useState<File | null>(null);


  const videoRef = useRef<HTMLVideoElement>(null);

  const pipeline = pipelines?.[pipelineId];

  const formatBurnDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const isMP4P = file.name.endsWith(".mp4p");

    if (isMP4P) {
      try {
        const result = await loadMP4P(file);
        setMp4pMetadata(result.metadata);
        setIsMP4PMode(true);

        if (result.metadata.expiresAt) {
          setBurnDateTimestamp(result.metadata.expiresAt);
          const date = new Date(result.metadata.expiresAt);
          setBurnDate(date.toISOString().split("T")[0]);
          setBurnTime(date.toTimeString().slice(0, 5));
        }

        const videoBlob = new Blob(
          [Uint8Array.from(atob(result.videoBase64), c => c.charCodeAt(0))],
          { type: "video/mp4" }
        );
        const videoFile = new File([videoBlob], "video.mp4", {
          type: "video/mp4",
        });

        if (onVideoFileUpload) {
          await onVideoFileUpload(videoFile);
        }

        console.log(
          `Loaded MP4P file. Showing ${result.showSynthed ? "synthed" : "original"} video`
        );
      } catch (error) {
        console.error("MP4P file loading failed:", error);
      }
    } else {
      setIsMP4PMode(false);
      setMp4pMetadata(null);
      setUploadedVideoFile(file);

      if (onVideoFileUpload) {
        try {
          await onVideoFileUpload(file);
        } catch (error) {
          console.error("Video upload failed:", error);
        }
      }
    }

    // Reset the input value so the same file can be selected again
    event.target.value = "";
  };

  const handleExportMP4P = async () => {
    if (!uploadedVideoFile || !burnDateTimestamp) {
      console.error("Missing required data for MP4P export");
      return;
    }

    try {
      let mp4pData = await encryptVideo(uploadedVideoFile, burnDateTimestamp);

      if (confirmedSynthedBlob) {
        const promptTexts = synthLockedPrompt
          ? [synthLockedPrompt]
          : prompts.map(prompt => prompt.text);
        mp4pData = await addSynthedVideo(
          mp4pData,
          confirmedSynthedBlob,
          promptTexts
        );
      }

      const filename = uploadedVideoFile.name.replace(/\.[^.]+$/, "");
      await downloadMP4P(mp4pData, filename);

      console.log("MP4P file exported successfully");
    } catch (error) {
      console.error("Failed to export MP4P:", error);
    }
  };

  const todayDate = new Date().toISOString().split("T")[0];
  const nowDate = new Date();
  const nowHour = nowDate.getHours();
  const nowMinute = nowDate.getMinutes();
  const nowSecond = nowDate.getSeconds();

  const padTime = (value: number) => value.toString().padStart(2, "0");
  const todayMidnight = new Date(
    nowDate.getFullYear(),
    nowDate.getMonth(),
    nowDate.getDate()
  );

  const toIsoDate = (date: Date) => {
    const year = date.getFullYear();
    const month = padTime(date.getMonth() + 1);
    const day = padTime(date.getDate());
    return `${year}-${month}-${day}`;
  };

  const formatDateDisplay = (value: string) => {
    if (!value) return "Select date";
    const parsed = new Date(`${value}T00:00:00`);
    return parsed.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
    });
  };

  const formatTimeDisplay = (value: string) =>
    value ? value : "Select time";

  const selectedDateIsToday = burnDate === todayDate;

  const monthLabel = useMemo(() => {
    return calendarMonth.toLocaleDateString(undefined, {
      month: "long",
      year: "numeric",
    });
  }, [calendarMonth]);

  const calendarDays = useMemo(() => {
    const year = calendarMonth.getFullYear();
    const month = calendarMonth.getMonth();
    const firstDay = new Date(year, month, 1);
    const startWeekday = firstDay.getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const days: Array<{ date: Date | null; label: string }> = [];

    for (let i = 0; i < startWeekday; i += 1) {
      days.push({ date: null, label: "" });
    }

    for (let day = 1; day <= daysInMonth; day += 1) {
      days.push({ date: new Date(year, month, day), label: String(day) });
    }

    return days;
  }, [calendarMonth]);

  useEffect(() => {
    if (!burnDate) return;
    const parsed = new Date(`${burnDate}T00:00:00`);
    setCalendarMonth(new Date(parsed.getFullYear(), parsed.getMonth(), 1));
  }, [burnDate]);

  useEffect(() => {
    if (!isTimePickerOpen) return;
    if (burnTime) {
      const [hour, minute, second] = burnTime.split(":").map(Number);
      setTimeSelection({
        hour,
        minute,
        second: Number.isFinite(second) ? second : 0,
      });
    } else {
      setTimeSelection({ hour: null, minute: null, second: null });
    }
    setTimeSelectionTouched(false);
  }, [burnTime, isTimePickerOpen]);

  useEffect(() => {
    if (!burnDate || !burnTime) return;
    if (!selectedDateIsToday) return;
    const [hour, minute, second] = burnTime.split(":").map(Number);
    const totalSeconds = hour * 3600 + minute * 60 + (second || 0);
    const nowSeconds = nowHour * 3600 + nowMinute * 60 + nowSecond;
    if (totalSeconds < nowSeconds) {
      setBurnTime("");
    }
  }, [burnDate, burnTime, nowHour, nowMinute, nowSecond, selectedDateIsToday]);

  useEffect(() => {
    if (!isDatePickerOpen && !isTimePickerOpen) return;
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        isDatePickerOpen &&
        datePickerRef.current &&
        !datePickerRef.current.contains(target) &&
        dateButtonRef.current &&
        !dateButtonRef.current.contains(target)
      ) {
        setIsDatePickerOpen(false);
      }
      if (
        isTimePickerOpen &&
        timePickerRef.current &&
        !timePickerRef.current.contains(target) &&
        timeButtonRef.current &&
        !timeButtonRef.current.contains(target)
      ) {
        setIsTimePickerOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isDatePickerOpen, isTimePickerOpen]);

  useEffect(() => {
    if (
      timeSelection.hour === null ||
      timeSelection.minute === null ||
      timeSelection.second === null ||
      !timeSelectionTouched
    ) {
      return;
    }
    const nextTime = `${padTime(timeSelection.hour)}:${padTime(
      timeSelection.minute
    )}:${padTime(timeSelection.second)}`;
    setBurnTime(nextTime);
    setIsTimePickerOpen(false);
  }, [timeSelection, timeSelectionTouched]);
  useEffect(() => {
    if (!burnDate) {
      setBurnDateTimestamp(null);
      return;
    }

    const dateTime = burnTime
      ? new Date(`${burnDate}T${burnTime}`)
      : new Date(`${burnDate}T23:59:59`);

    if (dateTime.getTime() > Date.now()) {
      setBurnDateTimestamp(dateTime.getTime());
    } else {
      setBurnDateTimestamp(null);
    }
  }, [burnDate, burnTime]);

  const canStartSynth =
    !isSynthCapturing &&
    !!uploadedVideoFile &&
    !!burnDateTimestamp &&
    !!prompts[0]?.text?.trim();

  return (
    <Card className={`h-full flex flex-col y2k-panel ${className}`}>
      <CardHeader className="flex-shrink-0 py-3 px-4 y2k-panel-header">
        <CardTitle className="text-sm font-medium">Input & Controls</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 overflow-y-auto flex-1 px-4 py-3 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div>
          <h3 className="text-xs font-medium mb-1.5">Video Input</h3>
          <div className="rounded-lg flex items-center justify-center bg-muted/10 overflow-hidden relative min-h-[120px]">
            {isInitializing ? (
              <div className="text-center text-muted-foreground text-sm">
                Initializing video...
              </div>
            ) : error ? (
              <div className="text-center text-red-500 text-sm p-4">
                <p>Video error:</p>
                <p className="text-xs mt-1">{error}</p>
              </div>
            ) : localStream ? (
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                muted
                playsInline
              />
            ) : (
              onVideoFileUpload && (
                <>
                  <input
                    type="file"
                    accept="video/mp4,.mp4p"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="video-upload"
                    disabled={isStreaming || isConnecting}
                  />
                  <label
                    htmlFor="video-upload"
                    className={`text-center text-muted-foreground text-sm p-4 ${
                      isStreaming || isConnecting
                        ? "opacity-50 cursor-not-allowed"
                        : "cursor-pointer"
                    }`}
                  >
                    Upload a vid to begin
                  </label>
                </>
              )
            )}
          </div>
        </div>

        {mp4pMetadata && isMP4PMode && (
          <div>
            <h3 className="text-sm font-medium mb-2">MP4P File Info</h3>
            <div className="flex flex-col relative gap-2 p-3 bg-muted/30 rounded-md text-sm">
              <div className="flex flex-col relative">
                <span className="text-xs text-muted-foreground">ID</span>
                <span className="font-mono text-xs truncate">
                  {mp4pMetadata.id}
                </span>
              </div>
              <div className="flex flex-col relative">
                <span className="text-xs text-muted-foreground">Created</span>
                <span>{new Date(mp4pMetadata.createdAt).toLocaleString()}</span>
              </div>
              <div className="flex flex-col relative">
                <span className="text-xs text-muted-foreground">Burns On</span>
                <span>{new Date(mp4pMetadata.expiresAt).toLocaleString()}</span>
              </div>
              {mp4pMetadata.promptsUsed &&
                mp4pMetadata.promptsUsed.length > 0 && (
                  <div className="flex flex-col relative">
                    <span className="text-xs text-muted-foreground">
                      Prompt Used
                    </span>
                    <span className="text-xs">
                      {mp4pMetadata.promptsUsed.join(", ")}
                    </span>
                  </div>
                )}
              {mp4pMetadata.burned && (
                <Badge variant="destructive" className="w-fit">
                  Burned
                </Badge>
              )}
            </div>
          </div>
        )}

        {!isMP4PMode && (
          <div>
            <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
              Burn Date
            </h3>
            <div className="flex flex-col relative gap-3">
              <div className="flex flex-col relative gap-2">
                <div className="relative">
                  <button
                    ref={dateButtonRef}
                    type="button"
                    className="win98-input text-sm w-full flex items-center justify-between px-3 py-2"
                    onClick={() => {
                      setIsDatePickerOpen(open => !open);
                      setIsTimePickerOpen(false);
                    }}
                  >
                    <span>{formatDateDisplay(burnDate)}</span>
                    <span className="text-xs">▼</span>
                  </button>
                  {isDatePickerOpen && (
                    <div
                      ref={datePickerRef}
                      className="win98-popover absolute left-0 right-0 mt-2 p-3 z-50"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <button
                          type="button"
                          className="win98-button px-2 py-1 text-xs"
                          onClick={() =>
                            setCalendarMonth(
                              prev =>
                                new Date(
                                  prev.getFullYear(),
                                  prev.getMonth() - 1,
                                  1
                                )
                            )
                          }
                        >
                          ◀
                        </button>
                        <span className="text-xs font-semibold">
                          {monthLabel}
                        </span>
                        <button
                          type="button"
                          className="win98-button px-2 py-1 text-xs"
                          onClick={() =>
                            setCalendarMonth(
                              prev =>
                                new Date(
                                  prev.getFullYear(),
                                  prev.getMonth() + 1,
                                  1
                                )
                            )
                          }
                        >
                          ▶
                        </button>
                      </div>
                      <div className="grid grid-cols-7 gap-1 text-[10px] text-gray-700 mb-1">
                        {["S", "M", "T", "W", "T", "F", "S"].map(label => (
                          <div key={label} className="text-center">
                            {label}
                          </div>
                        ))}
                      </div>
                      <div className="grid grid-cols-7 gap-1">
                        {calendarDays.map((day, index) => {
                          if (!day.date) {
                            return (
                              <div
                                key={`empty-${index}`}
                                className="h-7"
                              />
                            );
                          }
                          const isoDate = toIsoDate(day.date);
                          const isSelected = isoDate === burnDate;
                          const isDisabled =
                            day.date.getTime() < todayMidnight.getTime();
                          return (
                            <button
                              key={isoDate}
                              type="button"
                              className={`win98-button h-7 text-xs ${
                                isSelected
                                  ? "bg-[#0b246a] text-white"
                                  : ""
                              }`}
                              disabled={isDisabled}
                              onClick={() => {
                                if (isDisabled) return;
                                setBurnDate(isoDate);
                                setIsDatePickerOpen(false);
                              }}
                            >
                              {day.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                <div className="relative">
                  <button
                    ref={timeButtonRef}
                    type="button"
                    className="win98-input text-sm w-full flex items-center justify-between px-3 py-2 disabled:opacity-50 tabular-nums"
                    disabled={!burnDate}
                    onClick={() => {
                      if (!burnDate) return;
                      setIsTimePickerOpen(open => !open);
                      setIsDatePickerOpen(false);
                      setTimeSelectionTouched(false);
                    }}
                  >
                    <span>{formatTimeDisplay(burnTime)}</span>
                    <span className="text-xs">▼</span>
                  </button>
                  {isTimePickerOpen && (
                    <div
                      ref={timePickerRef}
                      className="win98-popover absolute left-0 right-0 mt-2 p-3 z-50"
                    >
                      <div className="text-xs font-semibold mb-2">
                        Select time (HH:MM:SS)
                      </div>
                      <div className="text-xs mb-2 tabular-nums">
                        {timeSelection.hour !== null &&
                        timeSelection.minute !== null &&
                        timeSelection.second !== null
                          ? `${padTime(timeSelection.hour)}:${padTime(
                              timeSelection.minute
                            )}:${padTime(timeSelection.second)}`
                          : "--:--:--"}
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-[10px] text-center mb-1">
                          Hours
                        </div>
                        <div className="text-[10px] text-center mb-1">
                          Minutes
                        </div>
                        <div className="text-[10px] text-center mb-1">
                          Seconds
                        </div>
                        <div className="max-h-40 overflow-y-auto pr-1">
                          <div className="grid grid-cols-3 gap-1">
                            {Array.from({ length: 24 }, (_, hour) => {
                              const isHourDisabled =
                                selectedDateIsToday && hour < nowHour;
                              const isSelected =
                                timeSelection.hour === hour;
                              return (
                                <button
                                  key={`hour-${hour}`}
                                  type="button"
                                  className={`win98-button text-xs h-7 tabular-nums ${
                                    isSelected
                                      ? "bg-[#0b246a] text-white"
                                      : ""
                                  }`}
                                  disabled={isHourDisabled}
                                  onClick={() => {
                                    if (isHourDisabled) return;
                                    setTimeSelectionTouched(true);
                                    setTimeSelection(prev => {
                                      const minute =
                                        prev.minute !== null &&
                                        selectedDateIsToday &&
                                        hour === nowHour &&
                                        prev.minute < nowMinute
                                          ? null
                                          : prev.minute;
                                      const second =
                                        prev.second !== null &&
                                        selectedDateIsToday &&
                                        hour === nowHour &&
                                        minute !== null &&
                                        prev.minute === nowMinute &&
                                        prev.second < nowSecond
                                          ? null
                                          : prev.second;
                                      return { hour, minute, second };
                                    });
                                  }}
                                >
                                  {padTime(hour)}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                        <div className="max-h-40 overflow-y-auto pr-1">
                          <div className="grid grid-cols-3 gap-1">
                            {Array.from({ length: 60 }, (_, minute) => {
                              const hour = timeSelection.hour;
                              const minuteDisabled =
                                hour === null ||
                                (selectedDateIsToday &&
                                  hour === nowHour &&
                                  minute < nowMinute);
                              const isSelected =
                                timeSelection.minute === minute;
                              return (
                                <button
                                  key={`minute-${minute}`}
                                  type="button"
                                  className={`win98-button text-xs h-7 tabular-nums ${
                                    isSelected
                                      ? "bg-[#0b246a] text-white"
                                      : ""
                                  }`}
                                  disabled={minuteDisabled}
                                  onClick={() => {
                                    if (minuteDisabled) return;
                                    setTimeSelectionTouched(true);
                                    setTimeSelection(prev => ({
                                      hour:
                                        prev.hour ??
                                        (selectedDateIsToday ? nowHour : 0),
                                      minute,
                                      second:
                                        prev.second !== null &&
                                        prev.hour === nowHour &&
                                        minute === nowMinute &&
                                        selectedDateIsToday &&
                                        prev.second < nowSecond
                                          ? null
                                          : prev.second,
                                    }));
                                  }}
                                >
                                  {padTime(minute)}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                        <div className="max-h-40 overflow-y-auto pr-1">
                          <div className="grid grid-cols-3 gap-1">
                            {Array.from({ length: 60 }, (_, second) => {
                              const hour = timeSelection.hour;
                              const minute = timeSelection.minute;
                              const secondDisabled =
                                hour === null ||
                                minute === null ||
                                (selectedDateIsToday &&
                                  hour === nowHour &&
                                  minute === nowMinute &&
                                  second < nowSecond);
                              const isSelected =
                                timeSelection.second === second;
                              return (
                                <button
                                  key={`second-${second}`}
                                  type="button"
                                  className={`win98-button text-xs h-7 tabular-nums ${
                                    isSelected
                                      ? "bg-[#0b246a] text-white"
                                      : ""
                                  }`}
                                  disabled={secondDisabled}
                                  onClick={() => {
                                    if (secondDisabled) return;
                                    setTimeSelectionTouched(true);
                                    setTimeSelection(prev => ({
                                      hour: prev.hour,
                                      minute: prev.minute,
                                      second,
                                    }));
                                  }}
                                >
                                  {padTime(second)}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              {burnDateTimestamp && (
                <div className="flex flex-col relative p-3 bg-muted/30 rounded-md">
                  <p className="text-xs text-muted-foreground mb-1">
                    Vid will burn on:
                  </p>
                  <p className="text-sm font-medium">
                    {formatBurnDate(burnDateTimestamp)}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        <div>
          {(() => {
            // The Input can have two states: Append (default) and Edit (when a prompt is selected and the video is paused)
            const isEditMode = selectedTimelinePrompt && isVideoPaused;

            // Hide prompts section if pipeline doesn't support prompts
            if (pipeline?.supportsPrompts === false) {
              return null;
            }

            return (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">Prompt</h3>
                  {isEditMode && (
                    <Badge variant="secondary" className="text-xs">
                      Editing
                    </Badge>
                  )}
                  {isSynthCapturing && (
                    <Badge variant="secondary" className="text-xs">
                      Locked
                    </Badge>
                  )}
                </div>

                {selectedTimelinePrompt ? (
                  <TimelinePromptEditor
                    prompt={selectedTimelinePrompt}
                    onPromptUpdate={onTimelinePromptUpdate}
                    disabled={isSynthCapturing}
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    promptIndex={_timelinePrompts.findIndex(
                      p => p.id === selectedTimelinePrompt.id
                    )}
                  />
                ) : (
                  <PromptInput
                    prompts={prompts}
                    onPromptsChange={onPromptsChange}
                    disabled={isSynthCapturing}
                    interpolationMethod={interpolationMethod}
                    onInterpolationMethodChange={onInterpolationMethodChange}
                    temporalInterpolationMethod={temporalInterpolationMethod}
                    onTemporalInterpolationMethodChange={
                      onTemporalInterpolationMethodChange
                    }
                    isLive={isLive}
                    onLivePromptSubmit={onLivePromptSubmit}
                    isStreaming={isStreaming}
                    transitionSteps={transitionSteps}
                    onTransitionStepsChange={onTransitionStepsChange}
                    timelinePrompts={_timelinePrompts}
                  />
                )}
              </div>
            );
          })()}
        </div>

        {!isMP4PMode && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Synth</h3>
            <div className="flex flex-wrap items-center text-xs gap-2">
              <Button
                onClick={onStartSynth}
                disabled={!canStartSynth || isConnecting}
                size="xs"
              >
                Start Synth
              </Button>
              <Button
                onClick={onCancelSynth}
                disabled={!isSynthCapturing}
                size="xs"
                variant="destructive"
              >
                Cancel Synth
              </Button>
            </div>
            {isSynthCapturing && (
              <div className="mt-2 text-xs text-muted-foreground">
                {isRecordingSynthed ? "Recording" : "Preparing"} from start
                with: {synthLockedPrompt || "Prompt"}
              </div>
            )}
            {recordedSynthedBlob && !isSynthCapturing && (
              <div className="mt-2 text-xs text-muted-foreground">
                Synth complete.
              </div>
            )}
            {confirmedSynthedBlob && !isSynthCapturing && (
              <div className="mt-2">
                <Badge variant="secondary" className="text-xs">
                  Synth ready for export
                </Badge>
              </div>
            )}
          </div>
        )}

        {!isMP4PMode && (
          <div>
            <Button
              onClick={handleExportMP4P}
              disabled={
                !uploadedVideoFile ||
                !burnDateTimestamp ||
                !confirmedSynthedBlob ||
                isConnecting ||
                isSynthCapturing
              }
              className="w-full"
              size="sm"
            >
              Export MP4P
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
