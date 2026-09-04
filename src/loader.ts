/**
 * Functional, zero-ceremony model loader for webml-kit.
 *
 * ```ts
 * import webml from 'webml-kit';
 *
 * // Speech recognition
 * const asr = await webml('onnx-community/whisper-tiny.en');
 * const { text } = await asr.transcribe(audioBlob);
 *
 * // Live mic listening
 * const mic = await asr.listen({ onTranscript: (t) => console.log(t) });
 *
 * // Text generation
 * const llm = await webml('onnx-community/Llama-3.2-1B-Instruct-ONNX');
 * for await (const { token } of llm.stream('Tell me a joke')) {
 *   process.stdout.write(token);
 * }
 *
 * // Callable invocation for any model
 * const classifier = await webml('Xenova/vit-base-patch16-224');
 * const labels = await classifier(imageFile);
 * ```
 */

import { ModelClient } from './model-client.js';
import { getModelInfo } from './hub.js';
import { coerceAudio, listenMic, type MicListener } from './inputs.js';
import type {
  PipelineTask,
  QuantizationType,
  DeviceBackend,
  TranscriptionResult,
  ClassificationResult,
  DetectionResult,
  EmbeddingResult,
  TextGenerationOptions,
  TextGenerationResult,
  ChatMessage,
  ProgressCallback,
} from './types.js';
import { TokenStream } from './streaming.js';

export interface WebMLOptions {
  /** Override pipeline task (auto-detected from HF metadata if omitted) */
  task?: PipelineTask;
  /** Quantization level (q4, q8, fp16, fp32, etc.) */
  dtype?: QuantizationType;
  /** Device backend preference ('webgpu' | 'wasm' | 'cpu') */
  device?: DeviceBackend;
  /** Hugging Face branch or revision */
  revision?: string;
  /** Custom worker script URL */
  workerUrl?: string | URL;
  /** Download / compilation progress callback */
  onProgress?: ProgressCallback;
}

export interface LoadedModel {
  /** The active pipeline task */
  task: PipelineTask;
  /** The model identifier or URL */
  modelId: string;
  /** The underlying ModelClient instance */
  client: ModelClient;

  /** Call the model on inputs directly */
  (input: unknown, options?: Record<string, unknown>): Promise<unknown>;

  /** Run inference */
  run<T = unknown>(input: unknown, options?: Record<string, unknown>): Promise<T>;
  /** Free memory and terminate worker */
  dispose(): void;

  /** Stream tokens for text generation */
  stream(input: string | ChatMessage[], options?: TextGenerationOptions): TokenStream;
  /** Generate full text completion */
  generate(input: string | ChatMessage[], options?: TextGenerationOptions): Promise<TextGenerationResult>;
  /** Transcribe audio (accepts Float32Array, Blob, File, URL, AudioBuffer) */
  transcribe(input: unknown, options?: Record<string, unknown>): Promise<TranscriptionResult>;
  /** Classify image */
  classify(input: unknown, options?: Record<string, unknown>): Promise<ClassificationResult[]>;
  /** Extract vector embeddings */
  embed(input: string | string[], options?: Record<string, unknown>): Promise<EmbeddingResult>;
  /** Detect objects in image */
  detect(input: unknown, options?: Record<string, unknown>): Promise<DetectionResult[]>;

  /** Listen to live microphone audio in browser and transcribe on the fly */
  listen(options: { onTranscript: (text: string) => void; intervalSeconds?: number }): Promise<MicListener>;
}

/**
 * Automatically infers the pipeline task from modelId, file extensions, or Hugging Face metadata.
 */
export async function inferTask(modelId: string): Promise<PipelineTask> {
  const lower = modelId.toLowerCase();

  // Fast heuristic matching
  if (
    lower.includes('whisper') ||
    lower.includes('asr') ||
    lower.includes('speech') ||
    lower.includes('sushrota') ||
    lower.includes('parakeet')
  ) {
    return 'automatic-speech-recognition';
  }
  if (
    lower.includes('llama') ||
    lower.includes('qwen') ||
    lower.includes('gpt') ||
    lower.includes('mistral') ||
    lower.includes('phi') ||
    lower.includes('gemma') ||
    lower.includes('bonsai')
  ) {
    return 'text-generation';
  }
  if (lower.includes('detr') || lower.includes('yolo')) {
    return 'object-detection';
  }
  if (lower.includes('vit') || lower.includes('resnet') || lower.includes('mobilenet')) {
    return 'image-classification';
  }
  if (lower.includes('minilm') || lower.includes('bge') || lower.includes('embed')) {
    return 'feature-extraction';
  }
  if (lower.endsWith('.onnx')) {
    return 'raw-onnx';
  }

  // Fallback to HF Hub lookup
  try {
    const info = await getModelInfo(modelId);
    if (info.task && info.task !== 'unknown') {
      return info.task as PipelineTask;
    }
    const tags = info.tags || [];
    if (tags.includes('automatic-speech-recognition') || tags.includes('audio')) {
      return 'automatic-speech-recognition';
    }
    if (tags.includes('text-generation')) {
      return 'text-generation';
    }
    if (tags.includes('image-classification')) {
      return 'image-classification';
    }
  } catch {
    // Non-fatal
  }

  return 'text-generation';
}

/**
 * Load any ML model for in-browser execution with zero ceremony.
 *
 * @param modelId - Hugging Face model ID (e.g. 'onnx-community/whisper-tiny.en') or URL
 * @param options - Configuration options
 * @returns Ready-to-use callable model instance
 */
export async function webml(
  modelId: string,
  options: WebMLOptions = {},
): Promise<LoadedModel> {
  const task = options.task ?? (await inferTask(modelId));

  const client = new ModelClient(options.workerUrl);
  await client.load({
    task,
    modelId,
    dtype: options.dtype,
    device: options.device,
    revision: options.revision,
    onProgress: options.onProgress,
  });

  const runner = async (input: unknown, runOptions?: Record<string, unknown>) => {
    if (task === 'automatic-speech-recognition') {
      const pcm = await coerceAudio(input);
      return client.run('automatic-speech-recognition', pcm, runOptions);
    }
    return client.run(task, input, runOptions);
  };

  const model = Object.assign(runner, {
    task,
    modelId,
    client,

    run: <T = unknown>(input: unknown, runOptions?: Record<string, unknown>): Promise<T> => {
      return runner(input, runOptions) as Promise<T>;
    },

    dispose: () => {
      client.dispose();
      client.terminate();
    },

    stream: (input: string | ChatMessage[], genOptions?: TextGenerationOptions) => {
      return client.stream(input, genOptions);
    },

    generate: (input: string | ChatMessage[], genOptions?: TextGenerationOptions) => {
      return client.generate(input, genOptions);
    },

    transcribe: async (input: unknown, runOptions?: Record<string, unknown>) => {
      const pcm = await coerceAudio(input);
      return client.run<TranscriptionResult>('automatic-speech-recognition', pcm, runOptions);
    },

    classify: (input: unknown, runOptions?: Record<string, unknown>) => {
      return client.run<ClassificationResult[]>('image-classification', input, runOptions);
    },

    embed: (input: string | string[], runOptions?: Record<string, unknown>) => {
      return client.run<EmbeddingResult>('feature-extraction', input, runOptions);
    },

    detect: (input: unknown, runOptions?: Record<string, unknown>) => {
      return client.run<DetectionResult[]>('object-detection', input, runOptions);
    },

    listen: async (listenOptions: { onTranscript: (text: string) => void; intervalSeconds?: number }) => {
      return listenMic(
        async (pcm) => {
          try {
            const res = await client.run<TranscriptionResult>('automatic-speech-recognition', pcm);
            if (res?.text?.trim()) {
              listenOptions.onTranscript(res.text.trim());
            }
          } catch (err) {
            console.warn('Transcription error during mic listen:', err);
          }
        },
        { intervalSeconds: listenOptions.intervalSeconds },
      );
    },
  });

  return model as unknown as LoadedModel;
}

export default webml;
