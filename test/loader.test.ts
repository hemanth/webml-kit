import { describe, it, expect } from 'vitest';
import { inferTask } from '../src/loader.js';
import { coerceAudio } from '../src/inputs.js';

describe('Task Auto-Inference', () => {
  it('correctly infers task from model identifiers', async () => {
    expect(await inferTask('gnumanth/sushrota-sanskrit-asr-onnx')).toBe('automatic-speech-recognition');
    expect(await inferTask('onnx-community/whisper-tiny.en')).toBe('automatic-speech-recognition');
    expect(await inferTask('onnx-community/Llama-3.2-1B-Instruct-ONNX')).toBe('text-generation');
    expect(await inferTask('Qwen/Qwen2.5-0.5B-Instruct')).toBe('text-generation');
    expect(await inferTask('Xenova/vit-base-patch16-224')).toBe('image-classification');
    expect(await inferTask('Xenova/detr-resnet-50')).toBe('object-detection');
    expect(await inferTask('Xenova/all-MiniLM-L6-v2')).toBe('feature-extraction');
    expect(await inferTask('./local/model.onnx')).toBe('raw-onnx');
  });
});

describe('Universal Audio Coercion', () => {
  it('coerces Float32Array directly', async () => {
    const raw = new Float32Array([0.1, 0.2, 0.3]);
    const res = await coerceAudio(raw);
    expect(res).toBe(raw);
  });

  it('coerces number[] array', async () => {
    const raw = [0.1, -0.2, 0.3];
    const res = await coerceAudio(raw);
    expect(res).toBeInstanceOf(Float32Array);
    expect(res[1]).toBeCloseTo(-0.2, 4);
  });

  it('coerces nested object with .audio or .pcm', async () => {
    const obj1 = { audio: new Float32Array([0.5, 0.6]) };
    const res1 = await coerceAudio(obj1);
    expect(res1.length).toBe(2);

    const obj2 = { pcm: [0.7, 0.8] };
    const res2 = await coerceAudio(obj2);
    expect(res2.length).toBe(2);
  });

  it('coerces Uint8Array WAV data', async () => {
    const numSamples = 50;
    const headerSize = 44;
    const buffer = new ArrayBuffer(headerSize + numSamples * 2);
    const view = new DataView(buffer);

    view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46); // 'RIFF'
    view.setUint32(4, 36 + numSamples * 2, true);
    view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45); // 'WAVE'
    view.setUint8(12, 0x66); view.setUint8(13, 0x6d); view.setUint8(14, 0x74); view.setUint8(15, 0x20); // 'fmt '
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61); // 'data'
    view.setUint32(40, numSamples * 2, true);

    for (let i = 0; i < numSamples; i++) {
      view.setInt16(44 + i * 2, 8192, true);
    }

    const uint8 = new Uint8Array(buffer);
    const res = await coerceAudio(uint8);
    expect(res.length).toBe(numSamples);
    expect(res[0]).toBeCloseTo(0.25, 2);
  });
});
