import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import type { PromptItem, PromptTransition } from "../lib/api";
import type { PipelineInfo } from "../types";
import { Button } from "./ui/button";
import {
  encryptVideo,
  downloadMP4P,
  addSynthedVideo,
  type MP4PData,
} from "../lib/mp4p-api";

interface InputAndControlsPanelProps {
  className?: string;
  pipelines: Record<string, PipelineInfo> | null;
  localStream: MediaStream | null;
  isInitializing: boolean;
  error: string | null;
  isStreaming: boolean;
  isConnecting: boolean;
  isLoading?: boolean;
  onVideoFileUpload?: (file: File) => Promise<boolean>;
  baseMp4pData?: MP4PData | null;
  prefillVideoFile?: File | null;
  fixedBurnDateTimestamp?: number | null;
  hideLocalPreview?: boolean;
  pipelineId: string;
  prompts: PromptItem[];
  onPromptsChange: (prompts: PromptItem[]) => void;
  onTransitionSubmit: (transition: PromptTransition) => void;
  onLivePromptSubmit?: (prompts: PromptItem[]) => void;
  isVideoPaused?: boolean;
  confirmedSynthedBlob: Blob | null;
  isRecordingSynthed: boolean;
  isSynthCapturing: boolean;
  synthLockedPrompt: string;
  onStartSynth: () => void;
  onCancelSynth: () => void;
  onDeleteBurn?: () => void;
  onTogglePause?: () => void;
  sam3MaskId?: string | null;
  onSam3Generate?: () => void;
  onSam3Clear?: () => void;
  sam3Ready?: boolean;
  sam3Status?: string | null;
  isSam3Generating?: boolean;
  sourceVideoBlocked?: boolean;
}

const makePresetSvg = (label: string, start: string, end: string) => {
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'>
    <defs>
      <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
        <stop offset='0%' stop-color='${start}'/>
        <stop offset='100%' stop-color='${end}'/>
      </linearGradient>
    </defs>
    <rect width='200' height='200' fill='url(%23g)'/>
    <circle cx='100' cy='100' r='72' fill='rgba(255,255,255,0.28)'/>
    <text x='100' y='112' text-anchor='middle' font-family='Handjet, Arial, sans-serif' font-size='40' fill='#2a0a18'>${label}</text>
  </svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
};

const PROMPT_PRESETS = [
  {
    id: "chrome-pop",
    prompt:
      "A luminous silhouette composed of swirling cosmic matter, filled with dense star fields, nebula clouds, flowing fire-like energy, and liquid rainbow colors that continuously move and circulate, emitting a soft yet intense glow; the colors shift smoothly from deep blues and violets to bright reds, greens, and golds, as if powered by an internal stellar engine, with particles drifting, igniting, and dissolving in slow motion; the edges shimmer and pulse with light, slightly translucent, energy crystals, high dynamic range, cinematic lighting, ethereal, otherworldly.",
    image: makePresetSvg("CHR", "#ffd0e6", "#b7ecff"),
  },
  {
    id: "pixel-dream",
    prompt:
      "Y2K exploding chrome hearts made of swirling cosmic matter and liquid rainbow light, packed with dense star fields, nebula clouds, and fire‑like energy streams; the hearts crack open and burst with glittering particles, prismatic shards, and plasma ribbons that circulate and flare in slow motion, emitting a soft but intense glow; colors shift smoothly from deep blues and violets into hot pinks, reds, neon greens, and molten gold, as if powered by an internal stellar engine; edges shimmer and pulse with translucent crystal highlights, high dynamic range, cinematic lighting, ethereal, otherworldly, glossy early‑2000s pop aesthetic.",
    image: makePresetSvg("PX", "#ffc7d9", "#c7d7ff"),
  },
  {
    id: "angel-heat",
    prompt:
      "Y2K firecore hearts and flames in glossy chrome and molten glass, blazing with neon orange, magenta, and electric blue heat; liquid flame ribbons spiral and explode outward with sparkling particles and glitter dust, pulsing like a club‑era screensaver; the fire glows from within, shifting from deep ember reds to hot pink and golden highlights, with shimmering translucent edges, high dynamic range, cinematic lighting, early‑2000s pop futurism.",
    image: makePresetSvg("ANG", "#ffe1f0", "#ffc6b0"),
  },
  {
    id: "neon-glass",
    prompt:
      "Y2K candy‑burst hearts and bubbles made of liquid sugar glass, coated in glossy chrome and pastel neon icing; swirling rainbow syrup, star‑glitter, and gummy‑like particles float and pop in slow motion, glowing from within; colors shift through cotton‑candy pinks, turquoise, lemon yellow, and electric purple with a soft but intense glow; translucent edges shimmer like hard candy, high dynamic range, cinematic lighting, early‑2000s pop aesthetic.",
    image: makePresetSvg("NEO", "#f7c2ff", "#c2ffe4"),
  },
  {
    id: "satin-fire",
    prompt:
      "Y2K leopard‑to‑dalmation print hearts in glossy chrome and translucent vinyl, covered with high‑contrast black‑and‑white spots and hot‑pink accents; the pattern ripples and warps across the surface like liquid plastic, shimmering with micro‑glitter and metallic highlights; bold 2000s pop vibe, high dynamic range, cinematic lighting, playful, glossy, otherworldly.",
    image: makePresetSvg("SAT", "#ffd2c2", "#f6b0e0"),
  },
];

const SNAP_TRANSITION_STEPS = 0;

export function InputAndControlsPanel({
  className = "",
  pipelines,
  localStream,
  isInitializing,
  error,
  isStreaming,
  isConnecting,
  isLoading = false,
  onVideoFileUpload,
  baseMp4pData = null,
  prefillVideoFile = null,
  fixedBurnDateTimestamp = null,
  hideLocalPreview = false,
  pipelineId,
  prompts,
  onPromptsChange,
  onTransitionSubmit,
  onLivePromptSubmit,
  isVideoPaused = false,
  confirmedSynthedBlob,
  isRecordingSynthed,
  isSynthCapturing,
  synthLockedPrompt,
  onStartSynth,
  onCancelSynth,
  onDeleteBurn,
  onTogglePause,
  sam3MaskId = null,
  onSam3Generate,
  onSam3Clear,
  sam3Ready = false,
  sam3Status = null,
  isSam3Generating = false,
  sourceVideoBlocked = false,
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

  const handlePresetSelect = (preset: (typeof PROMPT_PRESETS)[number]) => {
    const nextPrompts = [{ text: preset.prompt, weight: 100 }];
    onPromptsChange(nextPrompts);
    if (!isStreaming || isLoading || isSynthCapturing) {
      return;
    }
    if (SNAP_TRANSITION_STEPS > 0) {
      onTransitionSubmit({
        target_prompts: nextPrompts,
        num_steps: SNAP_TRANSITION_STEPS,
        temporal_interpolation_method: "slerp",
      });
    } else {
      onLivePromptSubmit?.(nextPrompts);
    }
  };
  const dateButtonRef = useRef<HTMLButtonElement>(null);
  const timeButtonRef = useRef<HTMLButtonElement>(null);
  const datePickerRef = useRef<HTMLDivElement>(null);
  const timePickerRef = useRef<HTMLDivElement>(null);
  const [uploadedVideoFile, setUploadedVideoFile] = useState<File | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  useEffect(() => {
    if (prefillVideoFile) {
      setUploadedVideoFile(prefillVideoFile);
    }
  }, [prefillVideoFile]);

  useEffect(() => {
    if (!fixedBurnDateTimestamp) return;
    const date = new Date(fixedBurnDateTimestamp);
    setBurnDate(date.toISOString().split("T")[0]);
    setBurnTime(date.toTimeString().slice(0, 8));
    setBurnDateTimestamp(fixedBurnDateTimestamp);
  }, [fixedBurnDateTimestamp]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (fixedBurnDateTimestamp) return;

    setUploadedVideoFile(file);

    if (onVideoFileUpload) {
      try {
        await onVideoFileUpload(file);
      } catch (error) {
        console.error("Video upload failed:", error);
      }
    }

    // Reset the input value so the same file can be selected again
    event.target.value = "";
  };

  const handleTriggerFilePicker = () => {
    if (fixedBurnDateTimestamp) return;
    fileInputRef.current?.click();
  };

  const handleExportMP4P = async () => {
    if ((!uploadedVideoFile && !baseMp4pData) || !burnDateTimestamp) {
      console.error("Missing required data for MP4P export");
      return;
    }

    try {
      setIsExporting(true);
      let mp4pData = baseMp4pData;
      if (!mp4pData && uploadedVideoFile) {
        mp4pData = await encryptVideo(uploadedVideoFile, burnDateTimestamp);
      }
      if (!mp4pData) {
        throw new Error("Missing MP4P data");
      }

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

      const filename = uploadedVideoFile
        ? uploadedVideoFile.name.replace(/\.[^.]+$/, "")
        : `burn-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`;
      await downloadMP4P(mp4pData, filename);

      console.log("MP4P file exported successfully");
    } catch (error) {
      console.error("Failed to export MP4P:", error);
    } finally {
      setIsExporting(false);
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

  const formatTimeDisplay = (value: string) => (value ? value : "Select time");

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
    if (fixedBurnDateTimestamp) return;
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
    if (fixedBurnDateTimestamp) {
      setBurnDateTimestamp(fixedBurnDateTimestamp);
      return;
    }
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
    !!prompts[0]?.text?.trim() &&
    isStreaming &&
    !isLoading &&
    sam3Ready;

  const isBurnDateLocked = fixedBurnDateTimestamp !== null;

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0 py-3 px-4">
        <CardTitle className="text-sm font-medium text-white">
          Input & Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 overflow-y-auto flex-1 px-4 py-3 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
        <div>
          <h3 className="text-xs font-medium mb-1.5">Video Input</h3>
          <div
            className={`rounded-lg flex items-center justify-center bg-muted/10 overflow-hidden relative min-h-[120px] ${
              fixedBurnDateTimestamp ? "" : "cursor-pointer"
            }`}
            onClick={() => {
              if (localStream && !hideLocalPreview) {
                handleTriggerFilePicker();
              }
            }}
          >
            {onVideoFileUpload && !isBurnDateLocked && (
              <input
                type="file"
                accept="video/mp4"
                onChange={handleFileUpload}
                className="hidden"
                id="video-upload"
                ref={fileInputRef}
              />
            )}
            {isInitializing ? (
              <div className="text-center text-muted-foreground text-sm">
                Initializing video...
              </div>
            ) : error ? (
              <div className="text-center text-red-500 text-sm p-4">
                <p>Video error:</p>
                <p className="text-xs mt-1">{error}</p>
              </div>
            ) : localStream && !hideLocalPreview ? (
              <div className="relative w-full h-full">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  muted
                  playsInline
                />
                {sourceVideoBlocked ? (
                  <div className="absolute inset-0 bg-black/20" />
                ) : null}
                {!fixedBurnDateTimestamp ? (
                  <div className="absolute inset-x-0 bottom-2 flex justify-center pointer-events-none">
                    <span className="mac-frosted-button px-3 py-1 text-[11px] text-white opacity-80">
                      Click to change video
                    </span>
                  </div>
                ) : null}
              </div>
            ) : (
              onVideoFileUpload && (
                <>
                  <label
                    htmlFor="video-upload"
                    className={`mac-frosted-button px-4 py-3 text-sm text-center ${
                      isBurnDateLocked
                        ? "opacity-50 cursor-not-allowed"
                        : "cursor-pointer"
                    }`}
                  >
                    {isBurnDateLocked
                      ? "Burn source loaded"
                      : "Upload a vid to begin"}
                  </label>
                </>
              )
            )}
          </div>
          {pipeline?.supportsPrompts !== false && (
            <div className="flex items-center justify-center gap-2 mt-2">
              <Button
                onClick={onTogglePause}
                disabled={isSynthCapturing}
                size="xs"
                variant="secondary"
              >
                {isVideoPaused ? "Play" : "Pause"}
              </Button>
            </div>
          )}
        </div>

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
                    if (isBurnDateLocked) return;
                    setIsDatePickerOpen(open => !open);
                    setIsTimePickerOpen(false);
                  }}
                  disabled={isBurnDateLocked}
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
                          return <div key={`empty-${index}`} className="h-7" />;
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
                              isSelected ? "bg-[#0b246a] text-white" : ""
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
                  disabled={!burnDate || isBurnDateLocked}
                  onClick={() => {
                    if (!burnDate || isBurnDateLocked) return;
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
                    className="win98-popover burn-time-popover absolute left-0 right-0 mt-2 p-3 z-50"
                  >
                    <div className="text-sm font-semibold mb-2 text-[#3a0f1f]">
                      Select time (HH:MM:SS)
                    </div>
                    <div className="text-base font-semibold mb-3 tabular-nums text-[#2a0a18]">
                      {timeSelection.hour !== null &&
                      timeSelection.minute !== null &&
                      timeSelection.second !== null
                        ? `${padTime(timeSelection.hour)}:${padTime(
                            timeSelection.minute
                          )}:${padTime(timeSelection.second)}`
                        : "--:--:--"}
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="text-xs text-center mb-1 text-[#5a1d32] uppercase tracking-[0.12em]">
                        Hours
                      </div>
                      <div className="text-xs text-center mb-1 text-[#5a1d32] uppercase tracking-[0.12em]">
                        Minutes
                      </div>
                      <div className="text-xs text-center mb-1 text-[#5a1d32] uppercase tracking-[0.12em]">
                        Seconds
                      </div>
                      <div className="max-h-40 overflow-y-auto pr-1">
                        <div className="grid grid-cols-2 gap-1">
                          {Array.from({ length: 24 }, (_, hour) => {
                            const isHourDisabled =
                              selectedDateIsToday && hour < nowHour;
                            const isSelected = timeSelection.hour === hour;
                            return (
                              <button
                                key={`hour-${hour}`}
                                type="button"
                                className={`win98-button burn-time-cell tabular-nums ${
                                  isSelected ? "burn-time-cell-selected" : ""
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
                        <div className="grid grid-cols-2 gap-1">
                          {Array.from({ length: 60 }, (_, minute) => {
                            const hour = timeSelection.hour;
                            const minuteDisabled =
                              hour === null ||
                              (selectedDateIsToday &&
                                hour === nowHour &&
                                minute < nowMinute);
                            const isSelected = timeSelection.minute === minute;
                            return (
                              <button
                                key={`minute-${minute}`}
                                type="button"
                                className={`win98-button burn-time-cell tabular-nums ${
                                  isSelected ? "burn-time-cell-selected" : ""
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
                        <div className="grid grid-cols-2 gap-1">
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
                            const isSelected = timeSelection.second === second;
                            return (
                              <button
                                key={`second-${second}`}
                                type="button"
                                className={`win98-button burn-time-cell tabular-nums ${
                                  isSelected ? "burn-time-cell-selected" : ""
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

        <div>
          {pipeline?.supportsPrompts !== false && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium">Style</h3>
                {isSynthCapturing && (
                  <Badge variant="secondary" className="text-xs">
                    Locked
                  </Badge>
                )}
              </div>
              <div className="prompt-orb-grid">
                {PROMPT_PRESETS.map(preset => {
                  const isSelected = prompts[0]?.text === preset.prompt;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      className={`prompt-orb ${isSelected ? "is-selected" : ""}`}
                      onClick={() => handlePresetSelect(preset)}
                      disabled={isSynthCapturing}
                    >
                      <span className="prompt-orb-frame">
                        <img src={preset.image} alt={preset.id} />
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {onSam3Generate && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">SAM3 Mask</h3>
            <div className="text-xs text-muted-foreground">
              Requires HF access to{" "}
              <a
                href="https://huggingface.co/facebook/sam3"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                facebook/sam3
              </a>
              . Request access before generating masks.
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Button
                size="xs"
                onClick={onSam3Generate}
                disabled={isSam3Generating || isConnecting || isLoading}
              >
                {isSam3Generating ? "Generating..." : "Regenerate Mask"}
              </Button>
              {sam3MaskId && (
                <Button
                  size="xs"
                  variant="destructive"
                  onClick={onSam3Clear}
                  disabled={isSam3Generating}
                >
                  Clear Mask
                </Button>
              )}
            </div>
            {sam3Status && (
              <div className="text-xs text-muted-foreground">{sam3Status}</div>
            )}
          </div>
        )}

        <div className="space-y-2">
          <h3 className="text-sm font-medium">Burn</h3>
          <div className="flex flex-wrap items-center text-xs gap-2">
            <Button
              onClick={onStartSynth}
              disabled={!canStartSynth || isConnecting || isLoading}
              size="xs"
            >
              Start Burn
            </Button>
            <Button
              onClick={onCancelSynth}
              disabled={!isSynthCapturing || isLoading}
              size="xs"
              variant="destructive"
            >
              Cancel Burn
            </Button>
            {confirmedSynthedBlob && !isSynthCapturing && (
              <Button onClick={onDeleteBurn} size="xs" variant="destructive">
                Delete Burn
              </Button>
            )}
          </div>
          {isSynthCapturing && (
            <div className="mt-2 text-xs text-muted-foreground">
              {isRecordingSynthed ? "Recording" : "Preparing"} from start with:{" "}
              {synthLockedPrompt || "Prompt"}
            </div>
          )}
        </div>

        <div>
          <Button
            onClick={handleExportMP4P}
            disabled={
              (!uploadedVideoFile && !baseMp4pData) ||
              !burnDateTimestamp ||
              !confirmedSynthedBlob ||
              isConnecting ||
              isSynthCapturing ||
              isExporting
            }
            className="w-full"
            size="sm"
          >
            {isExporting ? "Exporting..." : "Export MP4P"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
