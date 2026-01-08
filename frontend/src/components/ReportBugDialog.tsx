import { useState } from "react";
import { Bug, Copy, ExternalLink, Check } from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { fetchCurrentLogs } from "../lib/api";
import { useI18n } from "../i18n";

interface ReportBugDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ReportBugDialog({ open, onClose }: ReportBugDialogProps) {
  const { t } = useI18n();
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCopyLogs = async () => {
    try {
      setIsLoadingLogs(true);
      setError(null);
      setCopySuccess(false);

      const logsText = await fetchCurrentLogs();

      await navigator.clipboard.writeText(logsText);

      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 3000);
    } catch (err) {
      console.error("Failed to copy logs:", err);
      setError(
        err instanceof Error ? err.message : t("bug.error.fetchCopy")
      );
    } finally {
      setIsLoadingLogs(false);
    }
  };

  const handleCreateBug = () => {
    window.open(
      "https://github.com/daydreamlive/scope/issues/new?template=bug-report.yml",
      "_blank"
    );
  };

  const handleOpenChange = (isOpen: boolean) => {
    if (!isOpen) {
      setCopySuccess(false);
      setError(null);
      onClose();
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bug className="h-5 w-5" />
            {t("bug.title")}
          </DialogTitle>
          <DialogDescription className="mt-3">
            {t("bug.description")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
      
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-semibold">
                1
              </div>
              <h4 className="font-semibold text-sm">{t("bug.copyTitle")}</h4>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              {t("bug.copyHint")}
            </p>
            <div className="pl-8">
              <Button
                onClick={handleCopyLogs}
                disabled={isLoadingLogs}
                variant="outline"
                className="gap-2"
              >
                {copySuccess ? (
                  <>
                    <Check className="h-4 w-4" />
                    {t("bug.copied")}
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    {isLoadingLogs ? t("bug.loading") : t("bug.copyTitle")}
                  </>
                )}
              </Button>
            </div>
            {error && <p className="text-sm text-destructive pl-8">{error}</p>}
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-semibold">
                2
              </div>
              <h4 className="font-semibold text-sm">{t("bug.createTitle")}</h4>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              {t("bug.createHint")}
            </p>
            <div className="pl-8">
              <Button onClick={handleCreateBug} className="gap-2">
                <ExternalLink className="h-4 w-4" />
                {t("bug.createButton")}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
