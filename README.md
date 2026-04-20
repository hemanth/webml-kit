# webml-utils

> Browser ML made easy. Framework-agnostic utilities for loading and running ML models in the browser via WebGPU/WASM.

Wraps [`@huggingface/transformers`](https://huggingface.co/docs/transformers.js) and eliminates the boilerplate you copy into every project: Web Worker setup, model caching, progress reporting, token streaming, GPU recovery, and device detection.

## Install

```bash
npm install webml-utils @huggingface/transformers
```

## Quick Start

```ts
import { ModelClient } from 'webml-utils';

// 1. Create a client (point to the worker file)
const client = new ModelClient(
  new URL('webml-utils/worker', import.meta.url)
);

// 2. Check device capabilities
const device = await client.detect();
console.log(device.backend);         // 'webgpu' | 'wasm' | 'cpu'
console.log(device.gpu?.vendor);      // 'apple'
console.log(device.recommendedDtype); // 'q4'

// 3. Load a text generation model
await client.load({
  task: 'text-generation',
  modelId: 'onnx-community/Bonsai-1.7B-ONNX',
  dtype: 'q4',
  onProgress: ({ percent }) => console.log(`Loading: ${percent}%`),
});

// 4. Stream tokens
for await (const { token, tps } of client.stream('Tell me a joke')) {
  process.stdout.write(token);
}
// → "Why did the developer go broke? Because he used up all his cache."
// → 42 tokens/sec
```

## Features

### 🔍 Device Detection

Auto-detect the best compute backend and recommended quantization:

```ts
import { detectDevice, canRun } from 'webml-utils';

const info = await detectDevice();
// { backend: 'webgpu', gpu: { vendor: 'apple', vram: 8589934592 }, recommendedDtype: 'fp16' }

const { ok, reason } = await canRun(4_000_000_000); // 4GB model
// { ok: true, backend: 'webgpu' }
```

### 📦 Cache Visibility

Know if a model is already downloaded before showing "downloading…" UI:

```ts
import { isCached, listCachedModels, getCacheSize, clearCache } from 'webml-utils';

if (await isCached('onnx-community/Bonsai-1.7B-ONNX')) {
  console.log('Instant load — no download needed!');
}

const models = await listCachedModels();
// [{ modelId: 'onnx-community/Bonsai-1.7B-ONNX', sizeBytes: 412_000_000 }]

console.log(await getCacheSize()); // 412_000_000 bytes

await clearCache('onnx-community/Bonsai-1.7B-ONNX'); // Free storage
```

### 🔄 Token Streaming

Proper `AsyncIterable` with TPS and time-to-first-token:

```ts
for await (const event of client.stream('Hello!')) {
  console.log(event);
  // { token: 'World', tps: 38.5, numTokens: 12, timeToFirstToken: 145 }
}

// Or collect everything at once:
const { text, tps, numTokens } = await client.generate('Hello!');
```

### 🛡️ GPU Recovery

Auto-recover from GPU device-lost events (TDR, VRAM pressure, driver resets):

```ts
import { GPURecovery } from 'webml-utils';

const recovery = new GPURecovery({ maxRetries: 3, baseDelayMs: 1000 });
recovery.on('lost', ({ reason }) => showBanner('GPU lost: ' + reason));
recovery.on('recovered', ({ adapter }) => console.log('GPU back online'));
recovery.on('failed', () => showFallbackMessage());
```

### 🎯 All Pipeline Tasks

Every `@huggingface/transformers` task is supported:

```ts
// Image Classification
const labels = await client.run('image-classification', imageUrl);
// [{ label: 'tabby cat', score: 0.98 }]

// Speech Recognition
const { text } = await client.run('automatic-speech-recognition', audioBlob);
// "Hello, how are you?"

// Embeddings
const vectors = await client.run('feature-extraction', 'Hello world');
// Float32Array[384]

// Object Detection
const objects = await client.run('object-detection', imageBlob);
// [{ label: 'person', score: 0.95, box: { xmin, ymin, xmax, ymax } }]

// Translation
const translated = await client.run('translation', 'Hello world');

// Summarization
const summary = await client.run('summarization', longText);

// And more: text-classification, zero-shot-classification,
// image-to-text, text-to-speech, fill-mask, question-answering,
// token-classification, depth-estimation, image-segmentation
```

## API Reference

### `ModelClient`

| Method | Description |
|---|---|
| `detect()` | Detect device capabilities |
| `load(options)` | Load a model pipeline |
| `stream(input, options?)` | Stream text generation tokens |
| `generate(input, options?)` | Generate text (one-shot) |
| `run(task, input, options?)` | Run any pipeline task |
| `interrupt()` | Interrupt ongoing generation |
| `reset()` | Reset KV cache (new conversation) |
| `dispose(modelKey?)` | Free model memory |
| `isLoaded(task, modelId)` | Check if a model is loaded |
| `terminate()` | Kill the worker |
| `on(event, listener)` | Listen to events |

### Events

| Event | Data |
|---|---|
| `progress` | `{ status, loaded, total, percent }` |
| `ready` | `{ modelKey }` |
| `error` | `{ message, id? }` |
| `device-lost` | `{ reason }` |
| `device-recovered` | `void` |

### Standalone Utilities

| Function | Module |
|---|---|
| `detectDevice()` | `webml-utils` |
| `checkWebGPU()` | `webml-utils` |
| `canRun(bytes)` | `webml-utils` |
| `isCached(modelId)` | `webml-utils` |
| `listCachedModels()` | `webml-utils` |
| `clearCache(modelId?)` | `webml-utils` |
| `getCacheSize()` | `webml-utils` |

## Supported Tasks

| Task | Streaming | Default Model |
|---|---|---|
| `text-generation` | ✅ | `onnx-community/Llama-3.2-1B-Instruct-ONNX` |
| `text-classification` | — | `Xenova/distilbert-base-uncased-finetuned-sst-2-english` |
| `image-classification` | — | `Xenova/vit-base-patch16-224` |
| `object-detection` | — | `Xenova/detr-resnet-50` |
| `automatic-speech-recognition` | — | `onnx-community/whisper-tiny.en` |
| `text-to-speech` | — | `Xenova/speecht5_tts` |
| `translation` | — | `Xenova/nllb-200-distilled-600M` |
| `summarization` | — | `Xenova/distilbart-cnn-6-6` |
| `feature-extraction` | — | `Xenova/all-MiniLM-L6-v2` |
| `image-to-text` | — | `Xenova/vit-gpt2-image-captioning` |
| `zero-shot-classification` | — | `Xenova/mobilebert-uncased-mnli` |
| `fill-mask` | — | `Xenova/bert-base-uncased` |
| `question-answering` | — | `Xenova/distilbert-base-uncased-distilled-squad` |
| `token-classification` | — | `Xenova/bert-base-NER` |
| `depth-estimation` | — | `Xenova/depth-anything-small-hf` |
| `image-segmentation` | — | `Xenova/detr-resnet-50-panoptic` |

## How It Works

```
┌─────────────┐     postMessage      ┌──────────────────┐
│  Your App   │ ◄──────────────────► │   model-worker   │
│  (main)     │   WorkerCommand/     │   (Web Worker)   │
│             │   WorkerResponse     │                  │
│  ModelClient│                      │  @hf/transformers │
│  TokenStream│                      │  WebGPU / WASM    │
│  GPURecovery│                      │  Pipeline Cache   │
└─────────────┘                      └──────────────────┘
```

## Requirements

- **Browser**: Chrome 113+, Edge 113+, Safari 18+ (WASM fallback)
- **Node.js**: 18+ (WASM only, no WebGPU)
- **Peer dep**: `@huggingface/transformers` >= 4.0.0

## License

MIT © [Hemanth HM](https://h3manth.com)
