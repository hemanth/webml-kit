/**
 * Search and discover WebGPU-compatible models on Hugging Face Hub.
 *
 * These functions use the public HF API — no auth token needed.
 * They only return models tagged with 'transformers.js',
 * so everything they return works with webml-kit out of the box.
 */

import {
  searchModels,
  listModelsForTask,
  trendingModels,
  listWebGPUModels,
  getModelInfo,
} from 'webml-kit';

// Search by query
const results = await searchModels({ query: 'whisper', limit: 5 });
for (const m of results) {
  console.log(`${m.modelId} — ${m.downloads} downloads — ${m.task}`);
}

// List top models for a specific task
const asrModels = await listModelsForTask('automatic-speech-recognition', 5);
console.log('top ASR models:', asrModels.map((m) => m.modelId));

// What's trending right now?
const hot = await trendingModels(5);
console.log('trending:', hot.map((m) => m.modelId));

// Only models from known WebGPU-friendly orgs (onnx-community, Xenova)
const webgpu = await listWebGPUModels({ task: 'text-generation', limit: 10 });
for (const m of webgpu) {
  console.log(`${m.modelId} — webgpu: ${m.webgpuCompatible}`);
}

// Get info about a specific model
const info = await getModelInfo('onnx-community/Bonsai-1.7B-ONNX');
console.log(info.task, info.downloads, info.tags);
