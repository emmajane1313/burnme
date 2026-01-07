import { Textarea } from "../ui/textarea";
import { Button } from "../ui/button";
import { X } from "lucide-react";
import type { PromptItem } from "../../lib/api";

interface PromptFieldProps {
  prompt: PromptItem;
  index: number;
  placeholder: string;
  showRemove: boolean;
  onSubmit?: () => void;
  onTextChange: (index: number, text: string) => void;
  onRemove: (index: number) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
  disabled?: boolean;
}

export function PromptField({
  prompt,
  index,
  placeholder,
  showRemove,
  onSubmit,
  onTextChange,
  onRemove,
  onKeyDown,
  disabled = false,
}: PromptFieldProps) {

  return (
    <>
      <Textarea
        placeholder={placeholder}
        value={prompt.text}
        onChange={e => onTextChange(index, e.target.value)}
        onKeyDown={e => {
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            onSubmit?.();
            return;
          }
          onKeyDown?.(e);
        }}
        disabled={disabled}
        rows={3}
        className={`flex-1 resize-none bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground  disabled:opacity-50 disabled:cursor-not-allowed min-h-[80px]`}
      />
      {showRemove && (
        <Button
          onClick={() => onRemove(index)}
          disabled={disabled}
          size="sm"
          variant="ghost"
          className="rounded-full w-8 h-8 p-0"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
    </>
  );
}
