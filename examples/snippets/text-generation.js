/**
 * Stream text generation with a language model.
 *
 * Shows the core workflow: create client, load model,
 * stream tokens as they're generated.
 */

import { ModelClient } from 'webml-kit';

const client = new ModelClient(
  new URL('webml-kit/worker', import.meta.url)
);

// Load model with progress
await client.load({
  task: 'text-generation',
  modelId: 'onnx-community/Bonsai-1.7B-ONNX',
  dtype: 'q4',
  onProgress: ({ percent }) => console.log(`loading: ${percent}%`),
});

// Stream tokens
let fullText = '';
for await (const { token, tps, numTokens } of client.stream('Explain quantum computing in simple terms')) {
  process.stdout.write(token);
  fullText += token;
}

console.log('\n---');
console.log('done');

// Or use generate() for one-shot
const result = await client.generate('What is WebGPU?');
console.log(result.text);
console.log(`${result.numTokens} tokens at ${result.tps.toFixed(1)} tok/s`);

// Clean up
client.dispose();
