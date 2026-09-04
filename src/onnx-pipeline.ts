/**
 * Direct ONNX Runtime Web pipeline runner.
 *
 * Supports standalone and custom ONNX models that do not use Hugging Face
 * Transformers architecture (such as NeMo Conformer-CTC ASR models,
 * standalone vision models, or raw ONNX models).
 */

import * as ort from 'onnxruntime-web';
import type { ModelConfig, ProgressCallback } from './types.js';

// Configure ONNX Runtime WebAssembly environment
ort.env.wasm.simd = true;
if (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) {
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency, 4);
}

export interface OnnxPipelineInstance {
  (input: unknown, options?: Record<string, unknown>): Promise<unknown>;
  sessions: ort.InferenceSession[];
  backend: 'webgpu' | 'wasm';
  dispose: () => void;
}

interface RepoFile {
  rfilename: string;
}

// ─── Asset Resolution Helpers ───

const HF_BASE = 'https://huggingface.co';

async function fetchRepoFiles(modelId: string): Promise<string[]> {
  try {
    const res = await fetch(`https://huggingface.co/api/models/${modelId}`);
    if (!res.ok) return [];
    const data = await res.json();
    return ((data.siblings ?? []) as RepoFile[]).map(s => s.rfilename);
  } catch {
    return [];
  }
}

function resolveAssetUrl(modelId: string, filename: string, revision = 'main'): string {
  if (filename.startsWith('http://') || filename.startsWith('https://') || filename.startsWith('/') || filename.startsWith('./')) {
    return filename;
  }
  return `${HF_BASE}/${modelId}/resolve/${revision}/${filename}`;
}

async function downloadAsset(
  url: string,
  fileName: string,
  onProgress?: ProgressCallback,
): Promise<ArrayBuffer> {
  if (
    typeof process !== 'undefined' &&
    process.versions?.node &&
    !url.startsWith('http://') &&
    !url.startsWith('https://')
  ) {
    const fs = await import('node:fs/promises');
    const path = url.startsWith('file://') ? new URL(url) : url;
    const buf = await fs.readFile(path);
    onProgress?.({
      status: 'downloading',
      file: fileName,
      loaded: buf.byteLength,
      total: buf.byteLength,
      percent: 100,
    });
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${fileName} from ${url}: ${response.status} ${response.statusText}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body || total === 0) {
    const buffer = await response.arrayBuffer();
    onProgress?.({
      status: 'downloading',
      file: fileName,
      loaded: buffer.byteLength,
      total: buffer.byteLength,
      percent: 100,
    });
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.length;
      const percent = total > 0 ? Math.round((loaded / total) * 100) : 0;
      onProgress?.({
        status: 'downloading',
        file: fileName,
        loaded,
        total,
        percent,
      });
    }
  }

  const merged = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged.buffer;
}

// ─── Session Management with Fallback ───

async function createSession(
  modelBufferOrPath: ArrayBuffer | Uint8Array | string,
  preferredBackend: 'webgpu' | 'wasm' | 'cpu' = 'webgpu',
): Promise<{ session: ort.InferenceSession; backend: 'webgpu' | 'wasm' }> {
  const model = modelBufferOrPath instanceof ArrayBuffer
    ? new Uint8Array(modelBufferOrPath)
    : modelBufferOrPath;

  if (preferredBackend === 'webgpu') {
    try {
      const session = await ort.InferenceSession.create(model as any, {
        executionProviders: ['webgpu'],
      });
      return { session, backend: 'webgpu' };
    } catch (err) {
      console.warn('WebGPU session creation failed, falling back to wasm:', err);
    }
  }

  const session = await ort.InferenceSession.create(model as any, {
    executionProviders: ['wasm'],
  });
  return { session, backend: 'wasm' };
}

// ─── Audio Helpers ───

export function decodeWavToFloat32(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  // Check 'RIFF' and 'WAVE'
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (riff !== 'RIFF' || wave !== 'WAVE') {
    // If not a valid WAV header, treat as raw Float32Array
    return new Float32Array(buffer);
  }

  let offset = 12;
  let audioFormat = 1;
  let numChannels = 1;
  let bitsPerSample = 16;
  let dataOffset = 0;
  let dataLength = 0;

  while (offset < view.byteLength - 8) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3)
    );
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === 'fmt ') {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataLength = chunkSize;
      break;
    }
    offset += 8 + chunkSize;
  }

  if (dataOffset === 0) {
    return new Float32Array(buffer);
  }

  if (audioFormat === 1 && bitsPerSample === 16) {
    // 16-bit PCM
    const numSamples = Math.floor(dataLength / (2 * numChannels));
    const pcm = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        const idx = dataOffset + (i * numChannels + c) * 2;
        if (idx + 1 < view.byteLength) {
          sum += view.getInt16(idx, true) / 32768.0;
        }
      }
      pcm[i] = sum / numChannels;
    }
    return pcm;
  }

  if (audioFormat === 3 && bitsPerSample === 32) {
    // 32-bit Float
    const numSamples = Math.floor(dataLength / (4 * numChannels));
    const pcm = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        const idx = dataOffset + (i * numChannels + c) * 4;
        if (idx + 3 < view.byteLength) {
          sum += view.getFloat32(idx, true);
        }
      }
      pcm[i] = sum / numChannels;
    }
    return pcm;
  }

  return new Float32Array(buffer.slice(dataOffset, dataOffset + dataLength));
}

export function toFloat32Array(input: unknown): Float32Array {
  if (input instanceof Float32Array) return input;
  if (Array.isArray(input)) return new Float32Array(input);
  if (input instanceof ArrayBuffer) return decodeWavToFloat32(input);
  if (ArrayBuffer.isView(input)) return new Float32Array(input.buffer, input.byteOffset, input.byteLength / 4);
  if (typeof input === 'object' && input !== null && 'audio' in input) {
    return toFloat32Array((input as { audio: unknown }).audio);
  }
  throw new Error(`Unsupported audio input format: ${typeof input}`);
}

// ─── CTC Greedy Decoding ───

export function decodeCTC(
  logprobs: Float32Array,
  timeSteps: number,
  numClasses: number,
  vocab: Record<number | string, string> | string[],
): string {
  let prevClass = 0;
  let text = '';

  for (let t = 0; t < timeSteps; t++) {
    const offset = t * numClasses;
    let maxVal = -Infinity;
    let argmax = 0;

    for (let c = 0; c < numClasses; c++) {
      const val = logprobs[offset + c];
      if (val > maxVal) {
        maxVal = val;
        argmax = c;
      }
    }

    if (argmax === 0) {
      prevClass = 0;
      continue;
    }

    if (argmax !== prevClass) {
      const token = (vocab as Record<number, string>)[argmax] ?? '';
      text += token;
      prevClass = argmax;
    }
  }

  // Replace SentencePiece whitespace marker ' ' (U+2581) with normal space
  return text.replace(/\u2581/g, ' ').replace(/\s+/g, ' ').trim();
}

// ─── Main Pipeline Factory ───

export async function createOnnxPipeline(
  config: ModelConfig,
  onProgress?: ProgressCallback,
): Promise<OnnxPipelineInstance> {
  const preferredDevice = config.device ?? 'webgpu';
  const revision = config.revision ?? 'main';

  // 1. Discover model files
  let modelFile = config.modelFile;
  let preprocessorFile = config.preprocessorFile;
  let vocabFile = config.vocabFile;

  if (!modelFile && !config.modelId.endsWith('.onnx')) {
    const repoFiles = await fetchRepoFiles(config.modelId);
    if (repoFiles.length > 0) {
      if (!modelFile) {
        // Look for acoustic / CTC / model ONNX
        modelFile = repoFiles.find(f => f.endsWith('.onnx') && !f.includes('preprocess'))
          ?? repoFiles.find(f => f.endsWith('.onnx'));
      }
      if (!preprocessorFile) {
        preprocessorFile = repoFiles.find(f => f.endsWith('.onnx') && f.includes('preprocess'));
      }
      if (!vocabFile) {
        vocabFile = repoFiles.find(f => f.endsWith('.json') && (f.includes('vocab') || f.includes('tokens')));
      }
    }
  }

  if (!modelFile) {
    modelFile = config.modelId.endsWith('.onnx') ? config.modelId : 'model.onnx';
  }

  const modelUrl = resolveAssetUrl(config.modelId, modelFile, revision);
  const modelBuffer = await downloadAsset(modelUrl, modelFile, onProgress);

  // Preprocessor (optional companion model for ASR/audio)
  let prepSession: ort.InferenceSession | null = null;
  if (preprocessorFile) {
    const prepUrl = resolveAssetUrl(config.modelId, preprocessorFile, revision);
    const prepBuffer = await downloadAsset(prepUrl, preprocessorFile, onProgress);
    const { session } = await createSession(prepBuffer, preferredDevice);
    prepSession = session;
  }

  // Vocab (optional mapping for CTC decoding)
  let vocab: Record<number, string> | string[] = {};
  if (vocabFile) {
    const vocabUrl = resolveAssetUrl(config.modelId, vocabFile, revision);
    try {
      if (
        typeof process !== 'undefined' &&
        process.versions?.node &&
        !vocabUrl.startsWith('http://') &&
        !vocabUrl.startsWith('https://')
      ) {
        const fs = await import('node:fs/promises');
        const path = vocabUrl.startsWith('file://') ? new URL(vocabUrl) : vocabUrl;
        const text = await fs.readFile(path, 'utf-8');
        vocab = JSON.parse(text);
      } else {
        const res = await fetch(vocabUrl);
        if (res.ok) {
          vocab = await res.json();
        }
      }
    } catch {
      // Non-fatal if vocab fails to load
    }
  }

  // Create primary session with WebGPU -> WASM fallback
  let { session: modelSession, backend } = await createSession(modelBuffer, preferredDevice);

  // ─── ASR Pipeline Function ───
  if (config.task === 'automatic-speech-recognition') {
    const runner = async (input: unknown): Promise<{ text: string }> => {
      const pcm = toFloat32Array(input);

      const executeInference = async (sess: ort.InferenceSession): Promise<{ text: string }> => {
        let acousticSignal: ort.Tensor;
        let acousticLength: ort.Tensor;

        if (prepSession) {
          // Mel filterbank extraction
          const audioTensor = new ort.Tensor('float32', pcm, [1, pcm.length]);
          const lengthTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(pcm.length)]), [1]);
          const prepOutputs = await prepSession.run({
            audio_signal: audioTensor,
            length: lengthTensor,
          });

          acousticSignal = (prepOutputs.processed_signal ?? Object.values(prepOutputs)[0]) as ort.Tensor;
          acousticLength = (prepOutputs.processed_length ?? Object.values(prepOutputs)[1]) as ort.Tensor;
        } else {
          // Pass raw audio directly
          acousticSignal = new ort.Tensor('float32', pcm, [1, pcm.length]);
          acousticLength = new ort.Tensor('int64', BigInt64Array.from([BigInt(pcm.length)]), [1]);
        }

        // Acoustic model inference
        const feeds: Record<string, ort.Tensor> = {};
        const inputNames = sess.inputNames;
        if (inputNames.length >= 2) {
          feeds[inputNames[0]] = acousticSignal;
          feeds[inputNames[1]] = acousticLength;
        } else if (inputNames.length === 1) {
          feeds[inputNames[0]] = acousticSignal;
        }

        const outputs = await sess.run(feeds);
        const logprobsTensor = (outputs.logprobs ?? Object.values(outputs)[0]) as ort.Tensor;

        const dims = logprobsTensor.dims;
        const timeSteps = dims.length === 3 ? dims[1] : (dims.length === 2 ? dims[0] : 1);
        const numClasses = dims[dims.length - 1];
        const data = logprobsTensor.data as Float32Array;

        const text = decodeCTC(data, timeSteps, numClasses, vocab);
        return { text };
      };

      try {
        return await executeInference(modelSession);
      } catch (err) {
        // If WebGPU failed (e.g. 1D conv kernel failure), fallback to WASM transparently
        if (backend === 'webgpu') {
          console.warn('WebGPU inference failed, retrying on WASM provider:', err);
          const wasmResult = await createSession(modelBuffer, 'wasm');
          modelSession = wasmResult.session;
          backend = 'wasm';
          return await executeInference(modelSession);
        }
        throw err;
      }
    };

    const instance = runner as unknown as OnnxPipelineInstance;
    instance.sessions = prepSession ? [prepSession, modelSession] : [modelSession];
    instance.backend = backend;
    instance.dispose = () => {
      prepSession?.release();
      modelSession.release();
    };
    return instance;
  }

  // ─── Generic / Raw ONNX Pipeline Function ───
  const genericRunner = async (input: unknown): Promise<Record<string, unknown>> => {
    const feeds: Record<string, ort.Tensor> = {};
    if (typeof input === 'object' && input !== null) {
      for (const [k, v] of Object.entries(input as Record<string, unknown>)) {
        if (v instanceof ort.Tensor) {
          feeds[k] = v;
        } else if (v instanceof Float32Array) {
          feeds[k] = new ort.Tensor('float32', v, [1, v.length]);
        }
      }
    }
    const outputs = await modelSession.run(feeds);
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(outputs)) {
      result[k] = v.data;
    }
    return result;
  };

  const instance = genericRunner as unknown as OnnxPipelineInstance;
  instance.sessions = [modelSession];
  instance.backend = backend;
  instance.dispose = () => {
    modelSession.release();
  };
  return instance;
}
