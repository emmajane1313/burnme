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
          <label
            onClick={() => onModeChange?.("upload")}
            className={`mac-frosted-button px-2 py-1 text-sm text-center ${
              mode === "upload"
                ? "opacity-50 cursor-not-allowed"
                : "cursor-pointer"
            }`}
          >
            Upload
          </label>

          <label
            onClick={() => onModeChange?.("play")}
            className={`mac-frosted-button px-2 py-1 text-sm text-center ${
              mode === "play"
                ? "opacity-50 cursor-not-allowed"
                : "cursor-pointer"
            }`}
          >
            Play
          </label>
        </div>
      </div>
    </header>
  );
}
