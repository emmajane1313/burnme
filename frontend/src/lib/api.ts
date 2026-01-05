import type { IceServersResponse, ModelStatusResponse } from "../types";

const API_BASE =
  (import.meta as ImportMeta).env?.VITE_API_BASE ?? "";
const API_ROOT = API_BASE.replace(/\/$/, "");
const apiUrl = (path: string) => `${API_ROOT}${path}`;

export interface PromptItem {
  text: string;
  weight: number;
}

export interface PromptTransition {
  target_prompts: PromptItem[];
  num_steps?: number; // Default: 4
  temporal_interpolation_method?: "linear" | "slerp"; // Default: linear
}

export interface WebRTCOfferRequest {
  sdp?: string;
  type?: string;
  initialParameters?: {
    prompts?: string[] | PromptItem[];
    prompt_interpolation_method?: "linear" | "slerp";
    transition?: PromptTransition;
    denoising_step_list?: number[];
    noise_scale?: number;
    noise_controller?: boolean;
    manage_cache?: boolean;
    kv_cache_attention_bias?: number;
    vace_ref_images?: string[];
    vace_context_scale?: number;
    sam3_mask_id?: string | null;
    sam3_mask_mode?: "inside" | "outside";
  };
}

export interface PipelineLoadParams {
  // Base interface for pipeline load parameters
  [key: string]: unknown;
}

// Generic load params - accepts any key-value pairs based on pipeline config
export type PipelineLoadParamsGeneric = Record<string, unknown>;

export interface PipelineLoadRequest {
  pipeline_id?: string;
  load_params?: PipelineLoadParamsGeneric | null;
}

export interface PipelineStatusResponse {
  status: "not_loaded" | "loading" | "loaded" | "error";
  pipeline_id?: string;
  load_params?: Record<string, unknown>;
  // Optional list of loaded LoRA adapters, provided by backend when available.
  loaded_lora_adapters?: { path: string; scale: number }[];
  error?: string;
}

export interface Sam3MaskResponse {
  success: boolean;
  maskId: string;
  frameCount: number;
  height: number;
  width: number;
  inputFps?: number | null;
  sam3Fps?: number | null;
  error?: string;
}

export const getIceServers = async (): Promise<IceServersResponse> => {
  const response = await fetch(apiUrl("/api/v1/webrtc/ice-servers"), {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Get ICE servers failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export interface WebRTCOfferResponse {
  sdp: string;
  type: string;
  sessionId: string;
}

export const sendWebRTCOffer = async (
  data: WebRTCOfferRequest
): Promise<WebRTCOfferResponse> => {
  const response = await fetch(apiUrl("/api/v1/webrtc/offer"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `WebRTC offer failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const sendIceCandidates = async (
  sessionId: string,
  candidates: RTCIceCandidate | RTCIceCandidate[]
): Promise<void> => {
  const candidateArray = Array.isArray(candidates) ? candidates : [candidates];

  const response = await fetch(
    apiUrl(`/api/v1/webrtc/offer/${sessionId}`),
    {
      method: "PATCH",
      // TODO: Use Content-Type 'application/trickle-ice-sdpfrag'
      // once backend supports it
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidates: candidateArray.map(c => ({
          candidate: c.candidate,
          sdpMid: c.sdpMid,
          sdpMLineIndex: c.sdpMLineIndex,
        })),
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Send ICE candidate failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }
};

export const loadPipeline = async (
  data: PipelineLoadRequest = {}
): Promise<{ message: string }> => {
  const response = await fetch(apiUrl("/api/v1/pipeline/load"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Pipeline load failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const getPipelineStatus = async (): Promise<PipelineStatusResponse> => {
  const response = await fetch(apiUrl("/api/v1/pipeline/status"), {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    signal: AbortSignal.timeout(30000), // 30 second timeout per request
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Pipeline status failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const checkModelStatus = async (
  pipelineId: string
): Promise<ModelStatusResponse> => {
  const response = await fetch(
    apiUrl(`/api/v1/models/status?pipeline_id=${pipelineId}`),
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Model status check failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const downloadPipelineModels = async (
  pipelineId: string
): Promise<{ message: string }> => {
  const response = await fetch(apiUrl("/api/v1/models/download"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pipeline_id: pipelineId }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Model download failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const generateSam3Mask = async (
  videoBase64: string,
  prompt: string,
  box?: [number, number, number, number] | null,
  inputFps?: number | null
): Promise<Sam3MaskResponse> => {
  const response = await fetch(apiUrl("/api/v1/sam3/mask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ videoBase64, prompt, box, input_fps: inputFps }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `SAM3 mask generation failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  if (!result.success) {
    throw new Error(result.error || "SAM3 mask generation failed");
  }

  return result;
};

export const startSam3MaskJob = async (
  videoBase64: string,
  prompt: string,
  box?: [number, number, number, number] | null,
  inputFps?: number | null
): Promise<{ jobId: string }> => {
  const response = await fetch(apiUrl("/api/v1/sam3/mask/start"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ videoBase64, prompt, box, input_fps: inputFps }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `SAM3 mask start failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const rawText = await response.text();
  let result: any;
  try {
    result = JSON.parse(rawText);
  } catch (error) {
    throw new Error(
      `SAM3 mask start failed: Invalid JSON response: ${rawText.slice(0, 200)}`
    );
  }
  if (!result.success || !result.jobId) {
    throw new Error(result.error || "SAM3 mask start failed");
  }

  return { jobId: result.jobId };
};

export const getSam3MaskJob = async (
  jobId: string
): Promise<{
  status: string;
  error?: string | null;
  result?: Sam3MaskResponse | null;
}> => {
  const response = await fetch(apiUrl(`/api/v1/sam3/mask/status/${jobId}`), {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `SAM3 mask status failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const rawText = await response.text();
  let result: any;
  try {
    result = JSON.parse(rawText);
  } catch (error) {
    throw new Error(
      `SAM3 mask status failed: Invalid JSON response: ${rawText.slice(0, 200)}`
    );
  }
  if (!result.success) {
    throw new Error(result.error || "SAM3 mask status failed");
  }

  return {
    status: result.status,
    error: result.error,
    result: result.result,
  };
};

export interface HardwareInfoResponse {
  vram_gb: number | null;
  spout_available: boolean;
}

export const getHardwareInfo = async (): Promise<HardwareInfoResponse> => {
  const response = await fetch(apiUrl("/api/v1/hardware/info"), {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Hardware info failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const fetchCurrentLogs = async (): Promise<string> => {
  const response = await fetch(apiUrl("/api/v1/logs/current"), {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Fetch logs failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const logsText = await response.text();
  return logsText;
};

export interface LoRAFileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
}

export interface LoRAFilesResponse {
  lora_files: LoRAFileInfo[];
}

export const listLoRAFiles = async (): Promise<LoRAFilesResponse> => {
  const response = await fetch(apiUrl("/api/v1/lora/list"), {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `List LoRA files failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export interface AssetFileInfo {
  name: string;
  path: string;
  size_mb: number;
  folder?: string | null;
  type: string; // "image" or "video"
  created_at: number; // Unix timestamp
}

export interface AssetsResponse {
  assets: AssetFileInfo[];
}

export const listAssets = async (
  type?: "image" | "video"
): Promise<AssetsResponse> => {
  const url = apiUrl(
    type ? `/api/v1/assets?type=${type}` : "/api/v1/assets"
  );
  const response = await fetch(url, {
    method: "GET",
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `List assets failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const uploadAsset = async (file: File): Promise<AssetFileInfo> => {
  const fileContent = await file.arrayBuffer();
  const filename = encodeURIComponent(file.name);

  const response = await fetch(
    apiUrl(`/api/v1/assets?filename=${filename}`),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
      },
      body: fileContent,
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Upload asset failed: ${response.status} ${response.statusText}: ${errorText}`
    );
  }

  const result = await response.json();
  return result;
};

export const getAssetUrl = (assetPath: string): string => {
  // The backend returns full absolute paths, but we need to extract the relative path
  // from the assets directory for the serving endpoint
  // Example: C:\Users\...\assets\myimage.png -> myimage.png
  // or: C:\Users\...\assets\subfolder\myimage.png -> subfolder/myimage.png

  const pathParts = assetPath.split(/[/\\]/);
  const assetsIndex = pathParts.findIndex(
    part => part === "assets" || part === ".burnmewhileimhot"
  );

  if (assetsIndex >= 0 && assetsIndex < pathParts.length - 1) {
    // Find the assets directory and take everything after it
    const assetsPos = pathParts.findIndex(part => part === "assets");
    if (assetsPos >= 0) {
      const relativePath = pathParts.slice(assetsPos + 1).join("/");
      return apiUrl(`/api/v1/assets/${relativePath}`);
    }
  }

  // Fallback: just use the filename
  const filename = pathParts[pathParts.length - 1];
  return apiUrl(`/api/v1/assets/${encodeURIComponent(filename)}`);
};

// Pipeline schema types - matches output of get_schema_with_metadata()
export interface PipelineSchemaProperty {
  type?: string;
  default?: unknown;
  description?: string;
  // JSON Schema fields
  minimum?: number;
  maximum?: number;
  items?: unknown;
  anyOf?: unknown[];
}

export interface PipelineConfigSchema {
  type: string;
  properties: Record<string, PipelineSchemaProperty>;
  required?: string[];
  title?: string;
}

// Mode-specific default overrides
export interface ModeDefaults {
  height?: number;
  width?: number;
  denoising_steps?: number[];
  noise_scale?: number | null;
  noise_controller?: boolean | null;
}

export interface PipelineSchemaInfo {
  id: string;
  name: string;
  description: string;
  version: string;
  docs_url: string | null;
  estimated_vram_gb: number | null;
  requires_models: boolean;
  supports_lora: boolean;
  supports_vace: boolean;
  // Pipeline config schema
  config_schema: PipelineConfigSchema;
  // Mode support - comes from config class
  supported_modes: ("text" | "video")[];
  default_mode: "text" | "video";
  // Prompt and temporal interpolation support
  supports_prompts: boolean;
  default_temporal_interpolation_method: "linear" | "slerp";
  default_temporal_interpolation_steps: number;
  // Mode-specific default overrides (optional)
  mode_defaults?: Record<"text" | "video", ModeDefaults>;
  // UI capabilities
  supports_cache_management: boolean;
  supports_kv_cache_bias: boolean;
  supports_quantization: boolean;
  min_dimension: number;
  recommended_quantization_vram_threshold: number | null;
  modified: boolean;
}

export interface PipelineSchemasResponse {
  pipelines: Record<string, PipelineSchemaInfo>;
}

export const getPipelineSchemas =
  async (): Promise<PipelineSchemasResponse> => {
    const response = await fetch(apiUrl("/api/v1/pipelines/schemas"), {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Get pipeline schemas failed: ${response.status} ${response.statusText}: ${errorText}`
      );
    }

    const result = await response.json();
    return result;
  };
