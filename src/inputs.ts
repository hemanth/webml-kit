/**
 * Universal input coercion and browser media helpers.
 *
 * Automatically converts URLs, Blobs, Files, AudioBuffers, and arrays
 * into the exact formats expected by ML models.
 */

import { decodeWavToFloat32 } from './onnx-pipeline.js';

export interface ListenOptions {
  /** Target sample rate in Hz (default: 16000) */
  sampleRate?: number;
  /** Chunk interval in seconds for streaming callback (default: 3) */
  intervalSeconds?: number;
  /** Callback fired for each audio chunk */
  onChunk?: (pcm: Float32Array) => void;
}

export interface MicListener {
  stop: () => void;
}

/**
 * Coerces various audio formats (URL, File, Blob, AudioBuffer, ArrayBuffer, Array)
 * into a standardized 16kHz Float32Array PCM buffer.
 */
export async function coerceAudio(input: unknown): Promise<Float32Array> {
  if (input instanceof Float32Array) {
    return input;
  }

  if (Array.isArray(input)) {
    return new Float32Array(input);
  }

  if (input instanceof ArrayBuffer) {
    return decodeWavToFloat32(input);
  }

  if (ArrayBuffer.isView(input)) {
    if (input instanceof Uint8Array) {
      const copy = new Uint8Array(input.byteLength);
      copy.set(new Uint8Array(input.buffer, input.byteOffset, input.byteLength));
      return decodeWavToFloat32(copy.buffer as ArrayBuffer);
    }
    return new Float32Array(input.buffer as ArrayBuffer, input.byteOffset, Math.floor(input.byteLength / 4));
  }

  // Handle Blob / File
  if (typeof Blob !== 'undefined' && input instanceof Blob) {
    const buffer = await input.arrayBuffer();
    // Try decoding via browser AudioContext if available for compressed formats (mp3/ogg/m4a)
    if (typeof AudioContext !== 'undefined' || typeof (globalThis as any).webkitAudioContext !== 'undefined') {
      try {
        const AudioCtx = (globalThis as any).AudioContext || (globalThis as any).webkitAudioContext;
        const ctx = new AudioCtx({ sampleRate: 16000 });
        const audioBuf = await ctx.decodeAudioData(buffer.slice(0));
        ctx.close();
        return audioBuf.getChannelData(0);
      } catch {
        // Fallback to WAV parser
        return decodeWavToFloat32(buffer);
      }
    }
    return decodeWavToFloat32(buffer);
  }

  // Handle URL string
  if (typeof input === 'string') {
    const response = await fetch(input);
    if (!response.ok) {
      throw new Error(`Failed to fetch audio from ${input}: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    return decodeWavToFloat32(buffer);
  }

  // Handle object with .audio or .pcm property
  if (typeof input === 'object' && input !== null) {
    if ('audio' in input) {
      return coerceAudio((input as { audio: unknown }).audio);
    }
    if ('pcm' in input) {
      return coerceAudio((input as { pcm: unknown }).pcm);
    }
    // Handle Web Audio API AudioBuffer
    if ('getChannelData' in input && typeof (input as any).getChannelData === 'function') {
      return (input as any).getChannelData(0);
    }
  }

  throw new Error(`Unsupported audio input type: ${typeof input}`);
}

/**
 * Capture microphone audio from browser and invoke callback with Float32Array PCM.
 */
export async function listenMic(
  onChunk: (pcm: Float32Array) => void,
  options: ListenOptions = {},
): Promise<MicListener> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('Microphone access is only available in browser environments with getUserMedia.');
  }

  const sampleRate = options.sampleRate ?? 16000;
  const intervalSeconds = options.intervalSeconds ?? 3;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  const AudioCtx = (globalThis as any).AudioContext || (globalThis as any).webkitAudioContext;
  const audioCtx = new AudioCtx({ sampleRate });
  const source = audioCtx.createMediaStreamSource(stream);

  let pcmBuffer: number[] = [];
  const maxSamples = Math.floor(sampleRate * intervalSeconds);
  let workletNode: any = null;
  let scriptProcessor: any = null;

  const handleData = (inputData: Float32Array) => {
    for (let i = 0; i < inputData.length; i++) {
      pcmBuffer.push(inputData[i]);
    }
    if (pcmBuffer.length >= maxSamples) {
      const chunk = new Float32Array(pcmBuffer);
      pcmBuffer = [];
      onChunk(chunk);
    }
  };

  if (audioCtx.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
    const workletCode = `
      class RecorderProcessor extends AudioWorkletProcessor {
        process(inputs) {
          const input = inputs[0];
          if (input && input[0]) {
            this.port.postMessage(input[0]);
          }
          return true;
        }
      }
      registerProcessor('recorder-processor', RecorderProcessor);
    `;
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const workletUrl = URL.createObjectURL(blob);
    await audioCtx.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);

    workletNode = new AudioWorkletNode(audioCtx, 'recorder-processor');
    workletNode.port.onmessage = (e: any) => {
      handleData(e.data);
    };
    source.connect(workletNode);
  } else {
    // Fallback for legacy environments lacking AudioWorklet
    const bufferSize = 4096;
    scriptProcessor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    scriptProcessor.onaudioprocess = (e: any) => {
      handleData(e.inputBuffer.getChannelData(0));
    };
    source.connect(scriptProcessor);
    scriptProcessor.connect(audioCtx.destination);
  }

  const stop = () => {
    if (pcmBuffer.length > 0) {
      onChunk(new Float32Array(pcmBuffer));
      pcmBuffer = [];
    }
    if (workletNode) workletNode.disconnect();
    if (scriptProcessor) scriptProcessor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track: any) => track.stop());
    audioCtx.close();
  };

  return { stop };
}
