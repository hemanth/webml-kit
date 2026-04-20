/**
 * Main-thread client for communicating with the ML worker.
 *
 * Wraps the Web Worker's postMessage protocol with a promise-based API,
 * AsyncIterable token streaming, and typed events.
 *
 * ```ts
 * import { ModelClient } from 'webml-utils';
 *
 * const client = new ModelClient();
 *
 * // Check device capabilities
 * const device = await client.detect();
 * console.log(device.backend); // 'webgpu'
 *
 * // Load a model
 * await client.load({
 *   task: 'text-generation',
 *   modelId: 'onnx-community/Bonsai-1.7B-ONNX',
 *   dtype: 'q4',
 *   onProgress: ({ percent }) => console.log(`${percent}%`),
 * });
 *
 * // Stream text generation
 * for await (const { token, tps } of client.stream('Hello!')) {
 *   process.stdout.write(token);
 * }
 *
 * // One-shot inference
 * const labels = await client.run('image-classification', imageBlob);
 * ```
 */

import type {
  DeviceInfo,
  ModelConfig,
  PipelineTask,
  ProgressCallback,
  TextGenerationOptions,
  TextGenerationResult,
  TokenEvent,
  WorkerCommand,
  WorkerResponse,
  ChatMessage,
} from './types.js';

import { TokenStream, collectStream } from './streaming.js';
import { GPURecovery } from './gpu-recovery.js';

// ─── Types ───

export interface LoadOptions extends ModelConfig {
  onProgress?: ProgressCallback;
}

export type ClientEventType =
  | 'progress'
  | 'ready'
  | 'error'
  | 'device-lost'
  | 'device-recovered';

type Listener = (data: unknown) => void;

// ─── Client ───

export class ModelClient {
  private worker: Worker | null = null;
  private workerUrl: string | URL | null;
  private listeners = new Map<ClientEventType, Set<Listener>>();
  private pendingRequests = new Map<
    string,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
      stream?: TokenStream;
    }
  >();
  private requestCounter = 0;
  private deviceInfo: DeviceInfo | null = null;
  private loadedModels = new Set<string>();
  private progressCallback: ProgressCallback | null = null;
  private gpuRecovery: GPURecovery;

  /**
   * Create a new ModelClient.
   *
   * @param workerUrl - URL to the model-worker.js file. If omitted,
   *   creates a Blob URL from the bundled worker (requires bundler support).
   */
  constructor(workerUrl?: string | URL) {
    this.workerUrl = workerUrl ?? null;
    this.gpuRecovery = new GPURecovery();

    this.gpuRecovery.on('lost', ({ reason }) => {
      this.emit('device-lost', { reason });
    });
    this.gpuRecovery.on('recovered', () => {
      this.emit('device-recovered', {});
    });
  }

  // ─── Worker Lifecycle ───

  private getWorker(): Worker {
    if (!this.worker) {
      if (this.workerUrl) {
        this.worker = new Worker(this.workerUrl, { type: 'module' });
      } else {
        // Inline worker via Blob URL
        // Users should provide their own worker URL or use a bundler
        // that handles `new Worker(new URL('./model-worker.js', import.meta.url))`
        throw new Error(
          'No worker URL provided. Pass the URL to your model-worker.js file, ' +
          'e.g. new ModelClient(new URL("webml-utils/worker", import.meta.url))',
        );
      }

      this.worker.addEventListener('message', this.handleMessage.bind(this));
      this.worker.addEventListener('error', (e) => {
        this.emit('error', { message: e.message });
      });
    }
    return this.worker;
  }

  private send(cmd: WorkerCommand): void {
    this.getWorker().postMessage(cmd);
  }

  private nextId(): string {
    return `req_${++this.requestCounter}_${Date.now()}`;
  }

  // ─── Message Handler ───

  private handleMessage(e: MessageEvent<WorkerResponse>): void {
    const msg = e.data;

    switch (msg.type) {
      case 'device-info': {
        this.deviceInfo = msg.data;
        const pending = this.pendingRequests.get('detect');
        if (pending) {
          pending.resolve(msg.data);
          this.pendingRequests.delete('detect');
        }
        break;
      }

      case 'progress': {
        this.progressCallback?.(msg.data);
        this.emit('progress', msg.data);
        break;
      }

      case 'ready': {
        this.loadedModels.add(msg.modelKey);
        this.emit('ready', { modelKey: msg.modelKey });
        const pending = this.pendingRequests.get('load');
        if (pending) {
          pending.resolve(undefined);
          this.pendingRequests.delete('load');
        }
        break;
      }

      case 'token': {
        const pending = this.pendingRequests.get(msg.id);
        if (pending?.stream) {
          pending.stream.push(msg.data);
        }
        break;
      }

      case 'result': {
        const pending = this.pendingRequests.get(msg.id);
        if (pending) {
          if (pending.stream) {
            pending.stream.end();
          }
          pending.resolve(msg.data);
          this.pendingRequests.delete(msg.id);
        }
        break;
      }

      case 'error': {
        const pending = this.pendingRequests.get(msg.id);
        if (pending) {
          if (pending.stream) {
            pending.stream.abort(new Error(msg.data));
          }
          pending.reject(new Error(msg.data));
          this.pendingRequests.delete(msg.id);
        }
        this.emit('error', { message: msg.data, id: msg.id });
        break;
      }

      case 'device-lost': {
        this.emit('device-lost', { reason: msg.reason });
        break;
      }

      case 'device-recovered': {
        this.emit('device-recovered', {});
        break;
      }
    }
  }

  // ─── Events ───

  /** Register an event listener. */
  on(event: ClientEventType, listener: Listener): this {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);
    return this;
  }

  /** Remove an event listener. */
  off(event: ClientEventType, listener: Listener): this {
    this.listeners.get(event)?.delete(listener);
    return this;
  }

  private emit(event: ClientEventType, data: unknown): void {
    this.listeners.get(event)?.forEach(fn => fn(data));
  }

  // ─── Public API ───

  /**
   * Detect the best available compute backend.
   *
   * ```ts
   * const info = await client.detect();
   * console.log(info.backend);         // 'webgpu'
   * console.log(info.gpu?.vendor);      // 'apple'
   * console.log(info.recommendedDtype); // 'q4'
   * ```
   */
  async detect(): Promise<DeviceInfo> {
    if (this.deviceInfo) return this.deviceInfo;

    return new Promise((resolve, reject) => {
      this.pendingRequests.set('detect', { resolve: resolve as (v: unknown) => void, reject });
      this.send({ type: 'check' });
    });
  }

  /**
   * Load a model pipeline.
   *
   * ```ts
   * await client.load({
   *   task: 'text-generation',
   *   modelId: 'onnx-community/Bonsai-1.7B-ONNX',
   *   dtype: 'q4',
   *   onProgress: ({ percent }) => updateUI(percent),
   * });
   * ```
   */
  async load(options: LoadOptions): Promise<void> {
    this.progressCallback = options.onProgress ?? null;

    const config: ModelConfig = {
      task: options.task,
      modelId: options.modelId,
      dtype: options.dtype,
      device: options.device,
      revision: options.revision,
    };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set('load', { resolve: resolve as (v: unknown) => void, reject });
      this.send({ type: 'load', config });
    });
  }

  /**
   * Run one-shot inference for any pipeline task.
   *
   * ```ts
   * // Image classification
   * const labels = await client.run('image-classification', imageUrl);
   *
   * // Speech recognition
   * const { text } = await client.run('automatic-speech-recognition', audioBlob);
   *
   * // Embeddings
   * const vectors = await client.run('feature-extraction', 'Hello world');
   * ```
   */
  async run<T = unknown>(
    task: PipelineTask,
    input: unknown,
    options?: Record<string, unknown>,
  ): Promise<T> {
    const id = this.nextId();

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.send({ type: 'run', id, task, input, options });
    });
  }

  /**
   * Generate text with streaming tokens.
   *
   * Returns an `AsyncIterable<TokenEvent>` that yields tokens as they're generated.
   *
   * ```ts
   * for await (const { token, tps } of client.stream('Tell me a joke')) {
   *   process.stdout.write(token);
   * }
   * ```
   */
  stream(
    input: string | ChatMessage[],
    options?: TextGenerationOptions,
  ): TokenStream {
    const id = this.nextId();
    const stream = new TokenStream(options?.signal);

    this.pendingRequests.set(id, {
      resolve: () => {}, // Result comes through the stream
      reject: (err) => stream.abort(err),
      stream,
    });

    this.send({
      type: 'run',
      id,
      task: 'text-generation',
      input,
      options: options as Record<string, unknown> | undefined,
    });

    return stream;
  }

  /**
   * Generate text and wait for the complete result.
   *
   * ```ts
   * const { text, tps, numTokens } = await client.generate('Hello!');
   * ```
   */
  async generate(
    input: string | ChatMessage[],
    options?: TextGenerationOptions,
  ): Promise<TextGenerationResult> {
    const tokenStream = this.stream(input, options);
    const collected = await collectStream(tokenStream);
    return {
      ...collected,
      totalTime: 0, // Will be filled from worker result
    };
  }

  /**
   * Interrupt an ongoing text generation.
   */
  interrupt(): void {
    this.send({ type: 'interrupt' });
  }

  /**
   * Reset the KV cache (start a new conversation).
   */
  reset(): void {
    this.send({ type: 'reset' });
  }

  /**
   * Dispose a loaded model and free memory.
   *
   * @param modelKey - Specific model key (task::modelId), or omit to dispose all.
   */
  dispose(modelKey?: string): void {
    this.send({ type: 'dispose', modelKey });
    if (modelKey) {
      this.loadedModels.delete(modelKey);
    } else {
      this.loadedModels.clear();
    }
  }

  /**
   * Check if a model is currently loaded.
   */
  isLoaded(task: PipelineTask, modelId: string): boolean {
    return this.loadedModels.has(`${task}::${modelId}`);
  }

  /**
   * Terminate the worker completely.
   */
  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
    this.loadedModels.clear();
    this.pendingRequests.clear();
    this.deviceInfo = null;
  }
}
