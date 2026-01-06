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
  synthedMimeType?: string;
  promptsUsed?: string[];
  synthedVersions?: Array<{
    createdAt: number;
    promptsUsed?: string[];
    synthedSalt: string;
    synthedIv: string;
    synthedAuthTag: string;
    synthedMimeType?: string;
  }>;
  visualCipher?: {
    version: number;
    pipelineId: string;
    pipelineVersionHash?: string;
    prompt: string;
    params: Record<string, unknown>;
    seed: number;
    maskMode: string;
    maskResolution: { width: number; height: number };
    frameCount: number;
    fps: number;
  };
}

export interface MP4PData {
  metadata: MP4PMetadata;
  encryptedVideo: string;
  encryptedSynthedVideo?: string;
  encryptedSynthedVideos?: string[];
  encryptedMaskFrames?: string[];
  maskFrameIndexMap?: number[];
  maskPayloadCodec?: string;
  signature: string;
}

export interface LoadMP4PResponse {
  success: boolean;
  showSynthed: boolean;
  videoBase64: string;
  mimeType?: string;
  metadata: MP4PMetadata;
  selectedBurnIndex?: number | null;
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

export async function blobToBase64(
  blob: Blob,
  filename: string
): Promise<string> {
  const file = new File([blob], filename, { type: blob.type });
  return fileToBase64(file);
}

export function base64ToBlob(base64: string, mimeType: string): Blob {
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

export async function createMP4P(videoId?: string): Promise<MP4PData> {
  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/create`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ videoId }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create MP4P: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error("Failed to create MP4P file");
  }

  return result.data;
}
export async function loadMP4P(
  mp4pFile: File,
  burnIndex?: number | null
): Promise<LoadMP4PResponse> {
  const fileContent = await mp4pFile.text();
  const mp4pData: MP4PData = JSON.parse(fileContent);

  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/load`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
      burnIndex,
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

export async function restoreMP4P(
  mp4pData: MP4PData,
  visualCipher: MP4PMetadata["visualCipher"],
  burnIndex?: number | null
): Promise<{ videoBase64: string; mimeType?: string }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/restore`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
      visualCipher,
      burnIndex,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to restore MP4P: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || "Failed to restore MP4P");
  }

  return { videoBase64: result.videoBase64, mimeType: result.mimeType };
}

export async function addSynthedVideo(
  mp4pData: MP4PData,
  synthedBlob: Blob,
  prompts: string[],
  synthedMimeType?: string,
  visualCipher?: MP4PMetadata["visualCipher"],
  encryptedMaskFrames?: string[],
  maskFrameIndexMap?: number[],
  maskPayloadCodec?: string
): Promise<MP4PData> {
  const mimeType = synthedBlob.type || "video/webm";
  const extension = mimeType.includes("mp4") ? "mp4" : "webm";
  const synthedBase64 = await blobToBase64(
    synthedBlob,
    `synthed.${extension}`
  );

  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/add-synthed`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
      synthedVideoBase64: synthedBase64,
      synthedMimeType: synthedMimeType ?? mimeType,
      promptsUsed: prompts,
      visualCipher,
      encryptedMaskFrames,
      maskFrameIndexMap,
      maskPayloadCodec,
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

export async function addSynthedVideoBase64(
  mp4pData: MP4PData,
  synthedVideoBase64: string,
  prompts: string[],
  synthedMimeType?: string,
  visualCipher?: MP4PMetadata["visualCipher"],
  encryptedMaskFrames?: string[],
  maskFrameIndexMap?: number[],
  maskPayloadCodec?: string
): Promise<MP4PData> {
  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/add-synthed`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
      synthedVideoBase64,
      synthedMimeType,
      promptsUsed: prompts,
      visualCipher,
      encryptedMaskFrames,
      maskFrameIndexMap,
      maskPayloadCodec,
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

export async function generateVisualCipherPayload(
  mp4pData: MP4PData,
  synthedVideoBase64: string,
  synthedMimeType: string | undefined,
  originalVideoBase64: string | undefined,
  maskId: string,
  prompt: string,
  params: Record<string, unknown>,
  seed: number,
  pipelineId: string,
  maskMode = "inside",
  synthedFps?: number | null
): Promise<{
  visualCipher: MP4PMetadata["visualCipher"];
  encryptedMaskFrames: string[];
  maskFrameIndexMap: number[];
  maskPayloadCodec: string;
  compositedVideoBase64?: string;
}> {
  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/visual-cipher`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mp4pData,
      synthedVideoBase64,
      synthedMimeType,
      originalVideoBase64,
      maskId,
      prompt,
      params,
      seed,
      pipelineId,
      maskMode,
      synthedFps,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to generate visual cipher: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || "Failed to generate visual cipher");
  }

  return {
    visualCipher: result.visualCipher,
    encryptedMaskFrames: result.encryptedMaskFrames,
    maskFrameIndexMap: result.maskFrameIndexMap,
    maskPayloadCodec: result.maskPayloadCodec,
    compositedVideoBase64: result.compositedVideoBase64,
  };
}

export async function decryptMP4P(
  mp4pData: MP4PData
): Promise<{ videoBase64: string; metadata: MP4PMetadata }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/mp4p/decrypt`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ mp4pData }),
  });

  if (!response.ok) {
    throw new Error(`Failed to decrypt MP4P: ${response.statusText}`);
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error("Failed to decrypt MP4P file");
  }

  return { videoBase64: result.videoBase64, metadata: result.metadata };
}

export async function downloadMP4P(
  mp4pData: MP4PData,
  filename: string
): Promise<void> {
  const jsonString = JSON.stringify(mp4pData, null, 2);
  // Use a generic binary MIME type so browsers keep the .mp4p extension.
  const blob = new Blob([jsonString], { type: "application/octet-stream" });
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
