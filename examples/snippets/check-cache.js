/**
 * Check what models are already cached locally.
 *
 * Useful for deciding whether to show a download progress
 * bar or jump straight to inference.
 */

import { isCached, listCachedModels, getCacheSize, formatSize } from 'webml-utils';

// Check a specific model
const cached = await isCached('onnx-community/Bonsai-1.7B-ONNX');
console.log('bonsai cached?', cached);

// List everything in the cache
const models = await listCachedModels();
for (const m of models) {
  console.log(`  ${m.modelId}: ${m.size}`);
}

// Total storage used
const total = await getCacheSize();
console.log('total cache usage:', formatSize(total));
