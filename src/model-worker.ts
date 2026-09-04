/**
 * Generic Web Worker for running ML pipelines.
 *
 * This file runs in a Web Worker thread. It handles:
 * - WebGPU device detection + WASM fallback
 * - Pipeline instantiation (singleton, keyed by task+model)
 * - Progress reporting during model download
 * - Text generation with streaming + KV cache
 * - One-shot inference for classification, detection, ASR, etc.
 * - GPU device-lost recovery
 * - Memory cleanup
 *
 * Communication is via structured `postMessage` using WorkerCommand/WorkerResponse types.
 */

import {
  pipeline as hfPipeline,
  TextStreamer,
  DynamicCache,
  InterruptableStoppingCriteria,
} from '@huggingface/transformers';

import { createOnnxPipeline, type OnnxPipelineInstance } from './onnx-pipeline.js';

import type {
  WorkerCommand,
  WorkerResponse,
  ModelConfig,
  PipelineTask,
  DeviceInfo,
  ProgressEvent,
  ProgressCallback,
  TokenEvent,
} from './types.js';

// ─── State ───

type PipelineInstance = Awaited<ReturnType<typeof hfPipeline>> | OnnxPipelineInstance;

const instances = new Map<string, Promise<PipelineInstance>>();
const kvCaches = new Map<string, InstanceType<typeof DynamicCache>>();
const stoppingCriteria = new InterruptableStoppingCriteria();

function modelKey(task: PipelineTask, modelId: string): string {
  return `${task}::${modelId}`;
}

// ─── Helpers ───

function send(msg: WorkerResponse): void {
  self.postMessage(msg);
}

// ─── Device Check ───

async function checkDevice(): Promise<void> {
  let backend: DeviceInfo['backend'] = 'cpu';
  let gpu: DeviceInfo['gpu'] = null;

  try {
    if ('gpu' in navigator) {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (adapter) {
        backend = 'webgpu';
        const info = adapter.info;
        const vram = Number(adapter.limits?.maxBufferSize ?? 0);
        const fmtVram = vram >= 1024 ** 3
          ? `${(vram / 1024 ** 3).toFixed(1)} GB`
          : `${(vram / 1024 ** 2).toFixed(1)} MB`;
        gpu = {
          vendor: info?.vendor ?? 'unknown',
          architecture: info?.architecture ?? 'unknown',
          description: info?.description ?? 'unknown',
          vram,
          vramFormatted: fmtVram,
        };
      }
    }
  } catch {
    // WebGPU not available
  }

  if (backend === 'cpu' && typeof WebAssembly !== 'undefined') {
    backend = 'wasm';
  }

  // Simple VRAM-based dtype recommendation
  const vram = gpu?.vram ?? 0;
  const recommendedDtype = vram >= 8e9 ? 'fp16' : vram >= 4e9 ? 'q8' : 'q4';

  send({ type: 'device-info', data: { backend, gpu, recommendedDtype } });
}

// ─── Load Pipeline ───

async function loadPipeline(config: ModelConfig): Promise<void> {
  const key = modelKey(config.task, config.modelId);

  // Dispose previous KV cache if switching models for same task
  if (kvCaches.has(key)) {
    kvCaches.get(key)?.dispose?.();
    kvCaches.delete(key);
  }

  send({
    type: 'progress',
    data: { status: 'downloading', loaded: 0, total: 0, percent: 0 },
  });

  // Create singleton pipeline
  if (!instances.has(key)) {
    const isExplicitOnnx =
      config.modelType === 'onnx' ||
      config.modelId.endsWith('.onnx') ||
      config.task === 'raw-onnx' ||
      config.task === 'custom';

    const onProgress = (info: ProgressEvent | Record<string, unknown>) => {
      const loaded = Number((info as Record<string, unknown>).loaded ?? 0);
      const total = Number((info as Record<string, unknown>).total ?? 1);
      const percent = (info as Record<string, unknown>).percent != null
        ? Number((info as Record<string, unknown>).percent)
        : (total > 0 ? Math.round((loaded / total) * 100) : 0);
      send({
        type: 'progress',
        data: {
          status: ((info as Record<string, unknown>).status as ProgressEvent['status']) ?? 'downloading',
          file: String((info as Record<string, unknown>).file ?? ''),
          loaded,
          total,
          percent,
        },
      });
    };

    const loadFn = async (): Promise<PipelineInstance> => {
      if (isExplicitOnnx) {
        return createOnnxPipeline(config, onProgress as ProgressCallback);
      }

      if (config.modelType === 'transformers') {
        return (await hfPipeline(config.task as any, config.modelId, {
          device: config.device ?? 'webgpu',
          dtype: config.dtype ?? 'q4',
          revision: config.revision,
          progress_callback: onProgress,
        })) as PipelineInstance;
      }

      // 'auto' mode: try Hugging Face pipeline first, fallback to ONNX if config is missing
      try {
        return (await hfPipeline(config.task as any, config.modelId, {
          device: config.device ?? 'webgpu',
          dtype: config.dtype ?? 'q4',
          revision: config.revision,
          progress_callback: onProgress,
        })) as PipelineInstance;
      } catch (hfErr) {
        const errMsg = hfErr instanceof Error ? hfErr.message : String(hfErr);
        if (
          errMsg.includes('config.json') ||
          errMsg.includes('Could not locate file') ||
          errMsg.includes('Unsupported model type')
        ) {
          return createOnnxPipeline(config, onProgress as ProgressCallback);
        }
        throw hfErr;
      }
    };

    instances.set(key, loadFn());
  }

  try {
    const generator = await instances.get(key)!;

    // Warm up with dummy inference (compiles WebGPU shaders)
    send({
      type: 'progress',
      data: { status: 'compiling', loaded: 0, total: 0, percent: 100 },
    });

    if (config.task === 'text-generation' && 'tokenizer' in generator) {
      const tokenizer = (generator as unknown as Record<string, unknown>).tokenizer;
      const model = (generator as unknown as Record<string, unknown>).model;
      if (tokenizer && model && typeof (tokenizer as Function) === 'function') {
        const inputs = (tokenizer as Function)('a');
        await (model as { generate: Function }).generate({
          ...inputs,
          max_new_tokens: 1,
        });
      }
    }

    send({ type: 'ready', modelKey: key });
  } catch (e) {
    instances.delete(key);
    send({
      type: 'error',
      id: 'load',
      data: e instanceof Error ? e.message : String(e),
    });
  }
}

// ─── Run Inference ───

async function runInference(
  id: string,
  task: PipelineTask,
  input: unknown,
  options?: Record<string, unknown>,
): Promise<void> {
  // Find the loaded pipeline for this task
  let pipelineInstance: PipelineInstance | null = null;

  for (const [key, promise] of instances) {
    if (key.startsWith(`${task}::`)) {
      pipelineInstance = await promise;
      break;
    }
  }

  if (!pipelineInstance) {
    send({ type: 'error', id, data: `No pipeline loaded for task: ${task}` });
    return;
  }

  try {
    if (task === 'text-generation') {
      await runTextGeneration(id, pipelineInstance, input, options);
    } else {
      // One-shot inference for all other tasks
      const result = await (pipelineInstance as Function)(input, options);
      send({ type: 'result', id, data: result });
    }
  } catch (e) {
    send({
      type: 'error',
      id,
      data: e instanceof Error ? e.message : String(e),
    });
  }
}

// ─── Text Generation (Streaming) ───

async function runTextGeneration(
  id: string,
  generator: PipelineInstance,
  messages: unknown,
  options?: Record<string, unknown>,
): Promise<void> {
  let startTime: number | null = null;
  let numTokens = 0;
  let tps = 0;
  let firstTokenTime = 0;

  const genObj = generator as unknown as {
    tokenizer: { decode: Function };
    model: { generate: Function };
  } & Function;

  if (typeof genObj !== 'function' && !('tokenizer' in genObj)) {
    send({ type: 'error', id, data: 'Invalid text-generation pipeline' });
    return;
  }

  const streamer = new TextStreamer(
    (genObj as { tokenizer: unknown }).tokenizer as ConstructorParameters<typeof TextStreamer>[0],
    {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (output: string) => {
        const event: TokenEvent = {
          token: output,
          tps,
          numTokens,
          timeToFirstToken: firstTokenTime,
        };
        send({ type: 'token', id, data: event });
      },
      token_callback_function: () => {
        const now = performance.now();
        if (startTime === null) {
          startTime = now;
          firstTokenTime = now; // Will be relative to request start
        }
        numTokens++;
        if (numTokens > 1 && startTime !== null) {
          tps = (numTokens / (now - startTime)) * 1000;
        }
      },
    },
  );

  // Get or create KV cache for this task+model
  const key = [...instances.keys()].find(k => k.startsWith('text-generation::'));
  if (key && !kvCaches.has(key)) {
    kvCaches.set(key, new DynamicCache());
  }
  const pastKeyValues = key ? kvCaches.get(key) : undefined;

  stoppingCriteria.reset();

  const genOptions: Record<string, unknown> = {
    max_new_tokens: (options?.maxNewTokens as number) ?? 1024,
    do_sample: (options?.doSample as boolean) ?? false,
    temperature: options?.temperature,
    top_p: options?.topP,
    top_k: options?.topK,
    repetition_penalty: options?.repetitionPenalty,
    streamer,
    stopping_criteria: stoppingCriteria,
    ...(pastKeyValues ? { past_key_values: pastKeyValues } : {}),
  };

  // Remove undefined keys
  for (const k of Object.keys(genOptions)) {
    if (genOptions[k] === undefined) delete genOptions[k];
  }

  try {
    const output = await (genObj as Function)(messages, genOptions);

    // Extract final text
    let finalText = '';
    if (Array.isArray(output) && output[0]?.generated_text) {
      const genText = output[0].generated_text;
      finalText = Array.isArray(genText)
        ? genText[genText.length - 1]?.content ?? ''
        : genText;
    }

    send({
      type: 'result',
      id,
      data: {
        text: finalText,
        numTokens,
        tps,
        timeToFirstToken: firstTokenTime,
        totalTime: startTime ? performance.now() - startTime : 0,
      },
    });
  } catch (e) {
    send({
      type: 'error',
      id,
      data: e instanceof Error ? e.message : String(e),
    });
  }
}

// ─── Cleanup ───

function disposeModel(targetKey?: string): void {
  if (targetKey) {
    const p = instances.get(targetKey);
    p?.then(inst => (inst as unknown as { dispose?: () => void })?.dispose?.());
    instances.delete(targetKey);
    if (kvCaches.has(targetKey)) {
      kvCaches.get(targetKey)?.dispose?.();
      kvCaches.delete(targetKey);
    }
  } else {
    for (const p of instances.values()) {
      p.then(inst => (inst as unknown as { dispose?: () => void })?.dispose?.());
    }
    instances.clear();
    for (const cache of kvCaches.values()) {
      cache?.dispose?.();
    }
    kvCaches.clear();
  }
}

function resetKVCache(): void {
  stoppingCriteria.reset();
  for (const cache of kvCaches.values()) {
    cache?.dispose?.();
  }
  kvCaches.clear();
}

// ─── Message Handler ───

self.addEventListener('message', async (e: MessageEvent<WorkerCommand>) => {
  const cmd = e.data;

  switch (cmd.type) {
    case 'check':
      await checkDevice();
      break;

    case 'load':
      await loadPipeline(cmd.config);
      break;

    case 'run':
      await runInference(
        cmd.id,
        cmd.task,
        cmd.input,
        cmd.options as Record<string, unknown> | undefined,
      );
      break;

    case 'interrupt':
      stoppingCriteria.interrupt();
      break;

    case 'reset':
      resetKVCache();
      break;

    case 'dispose':
      disposeModel(cmd.modelKey);
      break;

    default:
      send({
        type: 'error',
        id: 'unknown',
        data: `Unknown command: ${(cmd as { type: string }).type}`,
      });
  }
});
