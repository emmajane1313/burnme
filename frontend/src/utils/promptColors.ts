// Color palette for prompt boxes
export const PROMPT_COLORS = [
  "#FF6B6B",
  "#4ECDC4",
  "#45B7D1",
  "#96CEB4",
  "#FFEAA7",
  "#DDA0DD",
  "#98D8C8",
  "#F7DC6F",
  "#BB8FCE",
  "#85C1E9",
  "#F8C471",
  "#82E0AA",
  "#F1948A",
  "#85C1E9",
  "#D7BDE2",
];

// Generate random color for prompt boxes
export const generateRandomColor = (excludeColors: string[] = []): string => {
  const availableColors = PROMPT_COLORS.filter(
    color => !excludeColors.includes(color)
  );

  if (availableColors.length === 0) {
    return PROMPT_COLORS[Math.floor(Math.random() * PROMPT_COLORS.length)];
  }

  return availableColors[Math.floor(Math.random() * availableColors.length)];
};

export const PROMPT: string =
  "A luminous silhouette composed of swirling cosmic matter, filled with dense star fields, nebula clouds, flowing fire-like energy, and liquid rainbow colors that continuously move and circulate, emitting a soft yet intense glow; the colors shift smoothly from deep blues and violets to bright reds, greens, and golds, as if powered by an internal stellar engine, with particles drifting, igniting, and dissolving in slow motion; the edges shimmer and pulse with light, slightly translucent, energy crystals, high dynamic range, cinematic lighting, ethereal, otherworldly.";
