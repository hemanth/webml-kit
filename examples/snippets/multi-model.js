/**
 * Load multiple models and use them independently.
 *
 * webml-utils manages a pipeline cache in the worker,
 * so different tasks share the same worker thread
 * without interfering with each other.
 */

import { ModelClient } from 'webml-utils';

const client = new ModelClient(
  new URL('webml-utils/worker', import.meta.url)
);

// Load a text model and an embedding model
console.log('loading text generation model...');
await client.load({
  task: 'text-generation',
  modelId: 'onnx-community/Bonsai-1.7B-ONNX',
  dtype: 'q4',
  onProgress: ({ percent }) => console.log(`  text-gen: ${percent}%`),
});

console.log('loading embedding model...');
await client.load({
  task: 'feature-extraction',
  modelId: 'Xenova/all-MiniLM-L6-v2',
  dtype: 'fp32',
  onProgress: ({ percent }) => console.log(`  embeddings: ${percent}%`),
});

// Use embeddings
const embedding = await client.run('feature-extraction', 'Hello world');
console.log('embedding dimensions:', embedding.length);

// Use text generation
for await (const { token } of client.stream('What is a vector embedding?')) {
  process.stdout.write(token);
}
console.log();

// Check what's loaded
console.log('text-gen loaded?', client.isLoaded('text-generation', 'onnx-community/Bonsai-1.7B-ONNX'));
console.log('embeddings loaded?', client.isLoaded('feature-extraction', 'Xenova/all-MiniLM-L6-v2'));

// Free everything
client.dispose();
console.log('cleaned up');
