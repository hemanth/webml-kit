/**
 * webml-utils — Browser ML Made Easy
 *
 * Framework-agnostic utilities for loading and running ML models
 * in the browser via WebGPU/WASM, powered by @huggingface/transformers.
 *
 * @example
 * ```ts
 * import { ModelClient, detectDevice, isCached } from 'webml-utils';
 *
 * // Check device
 * const device = await detectDevice();
 * console.log(device.backend); // 'webgpu'
 *
 * // Check cache
 * if (await isCached('onnx-community/Bonsai-1.7B-ONNX')) {
 *   console.log('Model already downloaded!');
 * }
 *
 * // Create client
 * const client = new ModelClient(
 *   new URL('./model-worker.js', import.meta.url)
 * );
 *
 * // Load model
 * await client.load({
 *   task: 'text-generation',
 *   modelId: 'onnx-community/Bonsai-1.7B-ONNX',
 *   dtype: 'q4',
 *   onProgress: ({ percent }) => console.log(`${percent}%`),
 * });
 *
 * // Stream tokens
 * for await (const { token, tps } of client.stream('Hello!')) {
 *   process.stdout.write(token);
 * }
 * ```
 *
 * @packageDocumentation
 */

// ─── Core ───
export { ModelClient } from './model-client.js';
export type { LoadOptions, ClientEventType } from './model-client.js';

// ─── Device Detection ───
export {
  detectDevice,
  checkWebGPU,
  checkWASM,
  getGPUAdapter,
  getGPUInfo,
  recommendDtype,
  canRun,
  parseSize,
  formatSize,
} from './device.js';

// ─── Cache ───
export {
  getCacheBackend,
  isCached,
  getCacheSize,
  listCachedModels,
  clearCache,
} from './cache.js';

// ─── Streaming ───
export { TokenStream, collectStream } from './streaming.js';

// ─── GPU Recovery ───
export { GPURecovery } from './gpu-recovery.js';
export type { RecoveryState, GPURecoveryEvents } from './gpu-recovery.js';

// ─── Pipelines ───
export {
  PIPELINE_REGISTRY,
  getPipelineDefaults,
  supportsStreaming,
} from './pipelines/index.js';
export type { PipelineDefaults } from './pipelines/index.js';

// ─── Types ───
export type {
  DeviceBackend,
  DeviceInfo,
  GPUInfo,
  QuantizationType,
  PipelineTask,
  ModelConfig,
  ProgressEvent,
  ProgressCallback,
  TokenEvent,
  ChatMessage,
  TextGenerationOptions,
  TextGenerationResult,
  ClassificationResult,
  DetectionResult,
  TranscriptionResult,
  EmbeddingResult,
  CaptionResult,
  TextResult,
  CacheBackend,
  CachedModel,
} from './types.js';
