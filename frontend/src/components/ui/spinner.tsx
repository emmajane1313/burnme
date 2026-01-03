import { Loader2, LoaderCircle, LoaderPinwheel } from "lucide-react";
import { cn } from "../../lib/utils";

export interface SpinnerProps {
  size?: number;
  className?: string;
  variant?: "default" | "circle" | "pinwheel";
}

export function Spinner({
  size = 16,
  className,
  variant = "default",
}: SpinnerProps) {
  const iconProps = {
    size,
    className: cn("animate-spin", className),
  };

  switch (variant) {
    case "circle":
      return <LoaderCircle {...iconProps} />;
    case "pinwheel":
      return <LoaderPinwheel {...iconProps} />;
    case "default":
    default:
      return <Loader2 {...iconProps} />;
  }
}
