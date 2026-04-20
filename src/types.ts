// ─── Device ───

export type DeviceBackend = 'webgpu' | 'wasm' | 'cpu';

export interface GPUInfo {
  vendor: string;
  architecture: string;
  description: string;
  /** Estimated VRAM in bytes (0 if unknown) */
  vram: number;
  /** Human-readable VRAM (e.g. '8.0 GB') */
  vramFormatted: string;
}

export interface DeviceInfo {
  backend: DeviceBackend;
  gpu: GPUInfo | null;
  /** Recommended dtype based on available VRAM */
  recommendedDtype: QuantizationType;
}

// ─── Quantization ───

export type QuantizationType =
  | 'fp32'
  | 'fp16'
  | 'q8'
  | 'q4'
  | 'q4f16'
  | 'q1'
  | 'int8'
  | 'uint8';

// ─── Pipeline Tasks ───

export type PipelineTask =
  | 'text-generation'
  | 'text-classification'
  | 'image-classification'
  | 'object-detection'
  | 'automatic-speech-recognition'
  | 'text-to-speech'
  | 'translation'
  | 'summarization'
  | 'feature-extraction'
  | 'image-to-text'
  | 'zero-shot-classification'
  | 'fill-mask'
  | 'question-answering'
  | 'token-classification'
  | 'depth-estimation'
  | 'image-segmentation';

// ─── Model Configuration ───

export interface ModelConfig {
  task: PipelineTask;
  modelId: string;
  /** Quantization level. Auto-selected if omitted. */
  dtype?: QuantizationType;
  /** Device preference. Falls back automatically. */
  device?: DeviceBackend;
  /** Revision / branch on HuggingFace Hub */
  revision?: string;
}

// ─── Progress ───

export interface ProgressEvent {
  status: 'downloading' | 'loading' | 'compiling' | 'warming';
  /** File being downloaded */
  file?: string;
  /** Bytes loaded so far */
  loaded: number;
  /** Total bytes */
  total: number;
  /** 0-100 */
  percent: number;
}

export type ProgressCallback = (event: ProgressEvent) => void;

// ─── Streaming (Text Generation) ───

export interface TokenEvent {
  /** The decoded text token */
  token: string;
  /** Tokens per second (0 until second token) */
  tps: number;
  /** Total tokens generated so far */
  numTokens: number;
  /** Milliseconds from request to first token */
  timeToFirstToken: number;
}

// ─── Chat Messages ───

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

// ─── Text Generation ───

export interface TextGenerationOptions {
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  doSample?: boolean;
  repetitionPenalty?: number;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
}

export interface TextGenerationResult {
  text: string;
  numTokens: number;
  tps: number;
  timeToFirstToken: number;
  totalTime: number;
}

// ─── Classification ───

export interface ClassificationResult {
  label: string;
  score: number;
}

// ─── Object Detection ───

export interface DetectionResult {
  label: string;
  score: number;
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
}

// ─── Speech Recognition ───

export interface TranscriptionResult {
  text: string;
  chunks?: Array<{
    text: string;
    timestamp: [number, number];
  }>;
  language?: string;
}

// ─── Embeddings ───

export interface EmbeddingResult {
  embedding: Float32Array;
  dimensions: number;
}

// ─── Image to Text ───

export interface CaptionResult {
  text: string;
}

// ─── Translation / Summarization ───

export interface TextResult {
  text: string;
}

// ─── Cache ───

export type CacheBackend = 'cache-api' | 'opfs' | 'indexeddb';

export interface CachedModel {
  modelId: string;
  /** Raw byte count */
  sizeBytes: number;
  /** Human-readable size (e.g. '412.0 MB') */
  size: string;
  lastAccessed: Date;
}

// ─── Worker Messages ───

export type WorkerCommand =
  | { type: 'check' }
  | { type: 'load'; config: ModelConfig }
  | { type: 'run'; id: string; task: PipelineTask; input: unknown; options?: unknown }
  | { type: 'interrupt' }
  | { type: 'reset' }
  | { type: 'dispose'; modelKey?: string }
  | { type: 'cache-status'; modelId: string }
  | { type: 'cache-clear'; modelId?: string };

export type WorkerResponse =
  | { type: 'device-info'; data: DeviceInfo }
  | { type: 'progress'; data: ProgressEvent }
  | { type: 'ready'; modelKey: string }
  | { type: 'token'; id: string; data: TokenEvent }
  | { type: 'result'; id: string; data: unknown }
  | { type: 'error'; id: string; data: string }
  | { type: 'device-lost'; reason: string }
  | { type: 'device-recovered' }
  | { type: 'cache-info'; data: { cached: boolean; size?: number } };

// ─── Client Events ───

export interface ModelClientEvents {
  progress: ProgressEvent;
  ready: { modelKey: string };
  token: TokenEvent;
  error: { message: string; id?: string };
  'device-lost': { reason: string };
  'device-recovered': void;
}
