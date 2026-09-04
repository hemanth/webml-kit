import { describe, it, expect } from 'vitest';
import {
  decodeCTC,
  decodeWavToFloat32,
  toFloat32Array,
  createOnnxPipeline,
} from '../src/onnx-pipeline.js';
import * as fs from 'node:fs';
import * as path from 'node:path';

describe('ONNX Audio Helpers', () => {
  it('decodes 16-bit PCM WAV to Float32Array', () => {
    // Generate a minimal 16-bit mono 16kHz WAV buffer
    const numSamples = 100;
    const headerSize = 44;
    const buffer = new ArrayBuffer(headerSize + numSamples * 2);
    const view = new DataView(buffer);

    // RIFF chunk
    view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46); // 'RIFF'
    view.setUint32(4, 36 + numSamples * 2, true);
    view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45); // 'WAVE'

    // fmt subchunk
    view.setUint8(12, 0x66); view.setUint8(13, 0x6d); view.setUint8(14, 0x74); view.setUint8(15, 0x20); // 'fmt '
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // audio format: 1 (PCM)
    view.setUint16(22, 1, true); // num channels: 1
    view.setUint32(24, 16000, true); // sample rate: 16000
    view.setUint32(28, 32000, true); // byte rate: 32000
    view.setUint16(32, 2, true); // block align: 2
    view.setUint16(34, 16, true); // bits per sample: 16

    // data subchunk
    view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61); // 'data'
    view.setUint32(40, numSamples * 2, true);

    // Write samples (amplitude 0.5 -> ~16384)
    for (let i = 0; i < numSamples; i++) {
      view.setInt16(44 + i * 2, 16384, true);
    }

    const decoded = decodeWavToFloat32(buffer);
    expect(decoded.length).toBe(numSamples);
    expect(decoded[0]).toBeCloseTo(0.5, 2);
  });

  it('normalizes various input formats to Float32Array', () => {
    const raw = new Float32Array([0.1, -0.2, 0.3]);
    expect(toFloat32Array(raw)).toBe(raw);

    const arr = [0.1, -0.2, 0.3];
    const fromArr = toFloat32Array(arr);
    expect(fromArr).toBeInstanceOf(Float32Array);
    expect(fromArr[0]).toBeCloseTo(0.1, 4);

    const obj = { audio: new Float32Array([0.5, -0.5]) };
    expect(toFloat32Array(obj).length).toBe(2);
  });
});

describe('CTC Greedy Decoder', () => {
  it('correctly collapses repeated tokens and ignores blank (0)', () => {
    // 3 classes: 0 = blank, 1 = 'a', 2 = 'b'
    const vocab = { 0: '<blank>', 1: 'a', 2: 'b' };
    const numClasses = 3;
    const timeSteps = 6;
    const logprobs = new Float32Array(timeSteps * numClasses);

    // t=0: class 1 ('a')
    logprobs[0 * 3 + 1] = 10;
    // t=1: class 1 ('a') -> collapsed
    logprobs[1 * 3 + 1] = 10;
    // t=2: class 0 (blank)
    logprobs[2 * 3 + 0] = 10;
    // t=3: class 1 ('a') -> new 'a' because separated by blank
    logprobs[3 * 3 + 1] = 10;
    // t=4: class 2 ('b')
    logprobs[4 * 3 + 2] = 10;
    // t=5: class 0 (blank)
    logprobs[5 * 3 + 0] = 10;

    const result = decodeCTC(logprobs, timeSteps, numClasses, vocab);
    expect(result).toBe('aab');
  });

  it('replaces SentencePiece boundary marker with space', () => {
    const vocab = {
      0: '<blank>',
      1: ' ॐ',
      2: ' नमः',
      3: ' शिवाय',
    };
    const numClasses = 4;
    const timeSteps = 5;
    const logprobs = new Float32Array(timeSteps * numClasses);

    logprobs[0 * 4 + 1] = 10; // ' ॐ'
    logprobs[1 * 4 + 0] = 10; // blank
    logprobs[2 * 4 + 2] = 10; // ' नमः'
    logprobs[3 * 4 + 0] = 10; // blank
    logprobs[4 * 4 + 3] = 10; // ' शिवाय'

    const result = decodeCTC(logprobs, timeSteps, numClasses, vocab);
    expect(result).toBe('ॐ नमः शिवाय');
  });
});

describe('Local Su-śrotā ONNX ASR Inference', () => {
  const modelDir = '/Users/lika/labs/sushrota-sanskrit-asr/models';
  const hasLocalModels =
    fs.existsSync(path.join(modelDir, 'sushrota_sanskrit_ctc_int8.onnx')) &&
    fs.existsSync(path.join(modelDir, 'preprocessor.onnx')) &&
    fs.existsSync(path.join(modelDir, 'sanskrit_vocab.json'));

  it.runIf(hasLocalModels)(
    'loads local Su-śrotā ONNX model and runs ASR inference in WASM',
    async () => {
      const pipeline = await createOnnxPipeline({
        task: 'automatic-speech-recognition',
        modelId: 'sushrota-local',
        device: 'wasm',
        modelFile: path.join(modelDir, 'sushrota_sanskrit_ctc_int8.onnx'),
        preprocessorFile: path.join(modelDir, 'preprocessor.onnx'),
        vocabFile: path.join(modelDir, 'sanskrit_vocab.json'),
      });

      expect(pipeline.backend).toBe('wasm');
      expect(pipeline.sessions.length).toBe(2);

      // Generate 1 second of 16kHz audio (silence/tone)
      const pcm = new Float32Array(16000);
      for (let i = 0; i < pcm.length; i++) {
        pcm[i] = Math.sin((2 * Math.PI * 350 * i) / 16000) * 0.1;
      }

      const res = (await pipeline(pcm)) as { text: string };
      expect(res).toBeDefined();
      expect(typeof res.text).toBe('string');

      pipeline.dispose();
    },
    30000,
  );

  it(
    'resolves model assets from Hugging Face Hub for gnumanth/sushrota-sanskrit-asr-onnx',
    async () => {
      // Test asset resolution without re-downloading 178MB if we pass local files, or test fetchRepoFiles
      const res = await fetch('https://huggingface.co/api/models/gnumanth/sushrota-sanskrit-asr-onnx');
      expect(res.ok).toBe(true);
      const data = await res.json();
      const files = (data.siblings || []).map((s: { rfilename: string }) => s.rfilename);
      expect(files).toContain('sushrota_sanskrit_ctc_int8.onnx');
      expect(files).toContain('preprocessor.onnx');
      expect(files).toContain('sanskrit_vocab.json');
    },
    15000,
  );
});
