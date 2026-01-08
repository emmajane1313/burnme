import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { useI18n, type Language } from "../i18n";

export function AboutPanel({ className = "" }: { className?: string }) {
  const { t, language, setLanguage } = useI18n();

  const handleLanguageChange = (value: string) => {
    if (value === "en-AU" || value === "ar-EG") {
      setLanguage(value as Language);
    }
  };

  return (
    <Card className={`h-full flex flex-col mac-translucent-ruby ${className}`}>
      <CardHeader className="flex-shrink-0">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base font-medium text-white">
            {t("about.title")}
          </CardTitle>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => handleLanguageChange("en-AU")}
              className={`mac-frosted-button px-2 py-1 text-xs text-center ${
                language === "en-AU"
                  ? "opacity-50 cursor-not-allowed"
                  : "cursor-pointer"
              }`}
            >
              {t("language.australian")}
            </button>
            <button
              type="button"
              onClick={() => handleLanguageChange("ar-EG")}
              className={`mac-frosted-button px-2 py-1 text-xs text-center ${
                language === "ar-EG"
                  ? "opacity-50 cursor-not-allowed"
                  : "cursor-pointer"
              }`}
            >
              {t("language.masri")}
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 overflow-y-auto flex-1 p-4 text-sm text-white/90">
        <div className="whitespace-pre-line">{t("about.body")}</div>
      </CardContent>
    </Card>
  );
}
