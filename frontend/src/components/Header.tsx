interface HeaderProps {
  className?: string;
  mode?: "upload" | "play";
  onModeChange?: (mode: "upload" | "play") => void;
}

export function Header({
  className = "",
  mode = "upload",
  onModeChange,
}: HeaderProps) {
  return (
    <header className={`w-full bg-transparent px-6 py-4 ${className}`}>
      <div className="flex flex-col items-center justify-center gap-3">
        <img
          draggable={false}
          src="/assets/images/burnme.gif"
          alt="Burn Me While I'm Hot"
          className="h-8 object-contain"
        />
        <div className="flex flex-wrap items-center justify-center gap-2">
          <button
            type="button"
            className={`win98-button px-3 py-1 text-xs ${
              mode === "upload" ? "bg-[#0b246a] text-white" : ""
            }`}
            onClick={() => onModeChange?.("upload")}
          >
            Upload
          </button>
          <button
            type="button"
            className={`win98-button px-3 py-1 text-xs ${
              mode === "play" ? "bg-[#0b246a] text-white" : ""
            }`}
            onClick={() => onModeChange?.("play")}
          >
            Play
          </button>
        </div>
      </div>
    </header>
  );
}
