import { useI18n } from "../i18n";

interface HeaderProps {
  className?: string;
  mode?: "upload" | "play" | "about";
  onModeChange?: (mode: "upload" | "play" | "about") => void;
}

export function Header({
  className = "",
  mode = "upload",
  onModeChange,
}: HeaderProps) {
  const { t } = useI18n();

  return (
    <header className={`w-full bg-transparent px-6 py-4 ${className}`}>
      <div className="flex flex-col items-center justify-center gap-3">
        <img
          draggable={false}
          src="/assets/images/burnme.gif"
          alt={t("header.logoAlt")}
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
            {t("nav.upload")}
          </label>

          <label
            onClick={() => onModeChange?.("play")}
            className={`mac-frosted-button px-2 py-1 text-sm text-center ${
              mode === "play"
                ? "opacity-50 cursor-not-allowed"
                : "cursor-pointer"
            }`}
          >
            {t("nav.play")}
          </label>
          <label
            onClick={() => onModeChange?.("about")}
            className={`mac-frosted-button px-2 py-1 text-sm text-center ${
              mode === "about"
                ? "opacity-50 cursor-not-allowed"
                : "cursor-pointer"
            }`}
          >
            {t("nav.about")}
          </label>
        </div>
      </div>
    </header>
  );
}
