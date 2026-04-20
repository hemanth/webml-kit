/**
 * Classify an image using a Vision Transformer.
 *
 * Pass an image URL, data URL, or Blob to get
 * ranked labels with confidence scores.
 */

import { ModelClient } from 'webml-kit';

const client = new ModelClient(
  new URL('webml-kit/worker', import.meta.url)
);

await client.load({
  task: 'image-classification',
  modelId: 'Xenova/vit-base-patch16-224',
  dtype: 'fp32',
});

// Classify from a URL
const labels = await client.run(
  'image-classification',
  'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg'
);

// labels: [{ label: 'tabby, tabby cat', score: 0.68 }, ...]
for (const { label, score } of labels.slice(0, 5)) {
  console.log(`${(score * 100).toFixed(1)}% — ${label}`);
}

client.dispose();
