/**
 * Pipeline registry — maps task names to their default configurations.
 */

import type { PipelineTask } from '../types.js';

export interface PipelineDefaults {
  /** Default model for this task */
  defaultModel: string;
  /** Default quantization */
  defaultDtype: string;
  /** Whether this task supports token streaming */
  supportsStreaming: boolean;
  /** Whether this task uses KV cache */
  usesKVCache: boolean;
}

/**
 * Registry of pipeline tasks with sensible defaults.
 * Users can override any of these when calling `client.load()`.
 */
export const PIPELINE_REGISTRY: Record<PipelineTask, PipelineDefaults> = {
  'text-generation': {
    defaultModel: 'onnx-community/Llama-3.2-1B-Instruct-ONNX',
    defaultDtype: 'q4',
    supportsStreaming: true,
    usesKVCache: true,
  },
  'text-classification': {
    defaultModel: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'image-classification': {
    defaultModel: 'Xenova/vit-base-patch16-224',
    defaultDtype: 'fp32',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'object-detection': {
    defaultModel: 'Xenova/detr-resnet-50',
    defaultDtype: 'fp32',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'automatic-speech-recognition': {
    defaultModel: 'onnx-community/whisper-tiny.en',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'text-to-speech': {
    defaultModel: 'Xenova/speecht5_tts',
    defaultDtype: 'fp32',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'translation': {
    defaultModel: 'Xenova/nllb-200-distilled-600M',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'summarization': {
    defaultModel: 'Xenova/distilbart-cnn-6-6',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'feature-extraction': {
    defaultModel: 'Xenova/all-MiniLM-L6-v2',
    defaultDtype: 'fp32',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'image-to-text': {
    defaultModel: 'Xenova/vit-gpt2-image-captioning',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'zero-shot-classification': {
    defaultModel: 'Xenova/mobilebert-uncased-mnli',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'fill-mask': {
    defaultModel: 'Xenova/bert-base-uncased',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'question-answering': {
    defaultModel: 'Xenova/distilbert-base-uncased-distilled-squad',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'token-classification': {
    defaultModel: 'Xenova/bert-base-NER',
    defaultDtype: 'q8',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'depth-estimation': {
    defaultModel: 'Xenova/depth-anything-small-hf',
    defaultDtype: 'fp32',
    supportsStreaming: false,
    usesKVCache: false,
  },
  'image-segmentation': {
    defaultModel: 'Xenova/detr-resnet-50-panoptic',
    defaultDtype: 'fp32',
    supportsStreaming: false,
    usesKVCache: false,
  },
};

/**
 * Get the pipeline defaults for a task.
 */
export function getPipelineDefaults(task: PipelineTask): PipelineDefaults {
  const defaults = PIPELINE_REGISTRY[task];
  if (!defaults) {
    throw new Error(`Unknown pipeline task: ${task}`);
  }
  return defaults;
}

/**
 * Check if a task supports real-time token streaming.
 */
export function supportsStreaming(task: PipelineTask): boolean {
  return PIPELINE_REGISTRY[task]?.supportsStreaming ?? false;
}
