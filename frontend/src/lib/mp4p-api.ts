export interface MP4PMetadata {
  id: string;
  expiresAt: number;
  salt: string;
  iv: string;
  authTag: string;
  createdAt: number;
  burned: boolean;
  burnedAt?: number;
  daydreamApiKey?: string;
  synthedSalt?: string;
  synthedIv?: string;
  synthedAuthTag?: string;
  promptsUsed?: string[];
}

export interface MP4PData {
  metadata: MP4PMetadata;
  encryptedVideo: string;
  encryptedSynthedVideo?: string;
  signature: string;
}

export interface LoadMP4PResponse {
  success: boolean;
  showSynthed: boolean;
  videoBase64: string;
  metadata: MP4PMetadata;
}

const API_BASE_URL =
  typeof window !== "undefined" ? window.location.origin : "http://localhost:8000";

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function base64ToBlob(base64: string, mimeType: string): Blob {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

export async function encryptVideo(
  videoFile: File,
  expiresAt: number
): Promise<MP4PData> {
  const videoBase64 = await fileToBase64(videoFile);

  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/encrypt`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      videoBase64,
      expiresAt,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to encrypt video: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error("Encryption failed");
  }

  return result.data;
}

export async function loadMP4P(mp4pFile: File): Promise<LoadMP4PResponse> {
  const fileContent = await mp4pFile.text();
  const mp4pData: MP4PData = JSON.parse(fileContent);

  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/load`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to load MP4P: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error("Failed to load MP4P file");
  }

  return result;
}

export async function addSynthedVideo(
  mp4pData: MP4PData,
  synthedBlob: Blob,
  prompts: string[]
): Promise<MP4PData> {
  const mimeType = synthedBlob.type || "video/webm";
  const extension = mimeType.includes("mp4") ? "mp4" : "webm";
  const synthedBase64 = await fileToBase64(
    new File([synthedBlob], `synthed.${extension}`, { type: mimeType })
  );

  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/add-synthed`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
      synthedVideoBase64: synthedBase64,
      promptsUsed: prompts,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to add synthed video: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error("Failed to add synthed video to MP4P");
  }

  return result.data;
}

export async function downloadMP4P(
  mp4pData: MP4PData,
  filename: string
): Promise<void> {
  const jsonString = JSON.stringify(mp4pData, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".mp4p") ? filename : `${filename}.mp4p`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function loadMP4PVideoToStream(
  videoBase64: string
): Promise<MediaStream> {
  return new Promise((resolve, reject) => {
    const videoBlob = base64ToBlob(videoBase64, "video/mp4");
    const videoUrl = URL.createObjectURL(videoBlob);

    const videoElement = document.createElement("video");
    videoElement.src = videoUrl;
    videoElement.muted = true;
    videoElement.loop = true;
    videoElement.playsInline = true;

    videoElement.onloadedmetadata = () => {
      videoElement.play().then(() => {
        const stream = (videoElement as any).captureStream
          ? (videoElement as any).captureStream()
          : (videoElement as any).mozCaptureStream();

        if (!stream) {
          reject(new Error("Failed to capture video stream"));
          return;
        }

        resolve(stream);
      });
    };

    videoElement.onerror = () => {
      reject(new Error("Failed to load video"));
    };
  });
}
