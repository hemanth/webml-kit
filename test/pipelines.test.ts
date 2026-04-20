import { describe, it, expect } from 'vitest';
import { PIPELINE_REGISTRY, getPipelineDefaults, supportsStreaming } from '../src/pipelines/index.js';
import type { PipelineTask } from '../src/types.js';

describe('pipeline registry', () => {
  it('has entries for all expected tasks', () => {
    const expectedTasks: PipelineTask[] = [
      'text-generation',
      'text-classification',
      'image-classification',
      'object-detection',
      'automatic-speech-recognition',
      'text-to-speech',
      'translation',
      'summarization',
      'feature-extraction',
      'image-to-text',
      'zero-shot-classification',
      'fill-mask',
      'question-answering',
      'token-classification',
      'depth-estimation',
      'image-segmentation',
    ];

    for (const task of expectedTasks) {
      expect(PIPELINE_REGISTRY[task]).toBeDefined();
      expect(PIPELINE_REGISTRY[task].defaultModel).toBeTruthy();
      expect(PIPELINE_REGISTRY[task].defaultDtype).toBeTruthy();
    }
  });

  it('only text-generation supports streaming', () => {
    for (const [task, config] of Object.entries(PIPELINE_REGISTRY)) {
      if (task === 'text-generation') {
        expect(config.supportsStreaming).toBe(true);
        expect(config.usesKVCache).toBe(true);
      } else {
        expect(config.supportsStreaming).toBe(false);
      }
    }
  });

  it('getPipelineDefaults returns correct config', () => {
    const defaults = getPipelineDefaults('text-generation');
    expect(defaults.defaultModel).toContain('ONNX');
    expect(defaults.supportsStreaming).toBe(true);
  });

  it('getPipelineDefaults throws on unknown task', () => {
    expect(() => getPipelineDefaults('banana-peeling' as PipelineTask)).toThrow(
      'Unknown pipeline task',
    );
  });

  it('supportsStreaming helper works', () => {
    expect(supportsStreaming('text-generation')).toBe(true);
    expect(supportsStreaming('image-classification')).toBe(false);
    expect(supportsStreaming('automatic-speech-recognition')).toBe(false);
  });
});
