# webml-utils

Framework-agnostic utilities for loading and running ML models in the browser via WebGPU/WASM.

If you've ever built a browser-ML demo, you know the drill: copy 150 lines of Web Worker boilerplate from the last project, wire up `postMessage`, add progress reporting, handle the GPU vanishing mid-inference, and pray the model is cached so your user doesn't wait 3 minutes. Every. Single. Time.

This library does that part for you. It wraps [`@huggingface/transformers`](https://huggingface.co/docs/transformers.js) with a sane API and handles the ugly bits: device detection, model caching, token streaming, KV-cache management, and GPU recovery.

## Install

```bash
npm install webml-utils @huggingface/transformers
```

## Quick start

```ts
import { ModelClient } from 'webml-utils';

// Point to the worker file
const client = new ModelClient(
  new URL('webml-utils/worker', import.meta.url)
);

// What can this machine do?
const device = await client.detect();
console.log(device.backend);         // 'webgpu' or 'wasm' or 'cpu'
console.log(device.gpu?.vendor);      // 'apple'
console.log(device.recommendedDtype); // 'q4'

// Load a model
await client.load({
  task: 'text-generation',
  modelId: 'onnx-community/Bonsai-1.7B-ONNX',
  dtype: 'q4',
  onProgress: ({ percent }) => console.log(`Loading: ${percent}%`),
});

// Stream tokens as they're generated
for await (const { token, tps } of client.stream('Tell me a joke')) {
  process.stdout.write(token);
}
```

## What's in here

### Device detection

Figures out what your user's machine can handle and picks a reasonable quantization level:

```ts
import { detectDevice, canRun } from 'webml-utils';

const info = await detectDevice();
// { backend: 'webgpu', gpu: { vendor: 'apple', vram: 8589934592 }, recommendedDtype: 'fp16' }

const { ok, reason } = await canRun('4GB'); // Can we fit a 4GB model?
```

### Cache visibility

The worst UX in browser ML is showing "downloading 2GB..." to someone who already has the model. Now you can check:

```ts
import { isCached, listCachedModels, getCacheSize, clearCache } from 'webml-utils';

if (await isCached('onnx-community/Bonsai-1.7B-ONNX')) {
  // Skip the progress bar entirely
}

const models = await listCachedModels();
// [{ modelId: 'onnx-community/Bonsai-1.7B-ONNX', size: '412.0 MB', sizeBytes: 432013312 }]

await clearCache('onnx-community/Bonsai-1.7B-ONNX'); // Free storage on mobile
```

### Token streaming

A proper `AsyncIterable` instead of raw `postMessage` callbacks. Tracks tokens-per-second and time-to-first-token:

```ts
for await (const event of client.stream('Hello!')) {
  console.log(event);
  // { token: 'World', tps: 38.5, numTokens: 12, timeToFirstToken: 145 }
}

// Or grab everything at once:
const { text, tps, numTokens } = await client.generate('Hello!');
```

### GPU recovery

GPUs disappear. It happens — TDR resets, VRAM pressure, mobile browsers reclaiming resources. Without handling this, the user has to reload the page. This recovers automatically with backoff:

```ts
import { GPURecovery } from 'webml-utils';

const recovery = new GPURecovery({ maxRetries: 3, baseDelayMs: 1000 });
recovery.on('lost', ({ reason }) => showBanner('GPU lost: ' + reason));
recovery.on('recovered', ({ adapter }) => console.log('Back online'));
recovery.on('failed', () => showFallbackMessage());
```

### All pipeline tasks

Not just text generation. Every task `@huggingface/transformers` supports works through the same API:

```ts
// Classify an image
const labels = await client.run('image-classification', imageUrl);
// [{ label: 'tabby cat', score: 0.98 }]

// Transcribe audio
const { text } = await client.run('automatic-speech-recognition', audioBlob);

// Get embeddings
const vectors = await client.run('feature-extraction', 'Hello world');

// Detect objects
const objects = await client.run('object-detection', imageBlob);

// Translate, summarize, caption images, answer questions,
// classify text, extract entities, estimate depth, segment images
```

## API

### ModelClient

| Method | What it does |
|---|---|
| `detect()` | Returns device capabilities and recommended dtype |
| `load(options)` | Downloads and initializes a model pipeline |
| `stream(input, options?)` | Returns an async iterator of tokens |
| `generate(input, options?)` | Generates text, waits for completion |
| `run(task, input, options?)` | Runs any pipeline task |
| `interrupt()` | Stops an in-progress generation |
| `reset()` | Clears the KV cache (new conversation) |
| `dispose(modelKey?)` | Frees model memory |
| `isLoaded(task, modelId)` | Checks if a specific model is ready |
| `terminate()` | Kills the worker entirely |
| `on(event, listener)` | Listens for progress, ready, error, device-lost, device-recovered |

### Standalone functions

These work without a ModelClient — useful for pre-flight checks:

| Function | What it does |
|---|---|
| `detectDevice()` | Backend detection + GPU info + dtype recommendation |
| `checkWebGPU()` | Boolean: is WebGPU available? |
| `canRun(bytes)` | Can a model of this size fit in VRAM? |
| `isCached(modelId)` | Is this model already downloaded? |
| `listCachedModels()` | What's in the cache? |
| `clearCache(modelId?)` | Delete cached model files |
| `getCacheSize()` | Total bytes used by cached models |
| `parseSize(input)` | Convert '4GB' / '512MB' to bytes |
| `formatSize(bytes)` | Convert bytes to '4.0 GB' / '512.0 MB' |

## Supported tasks

| Task | Streaming | Default model |
|---|---|---|
| `text-generation` | yes | `onnx-community/Llama-3.2-1B-Instruct-ONNX` |
| `text-classification` | no | `Xenova/distilbert-base-uncased-finetuned-sst-2-english` |
| `image-classification` | no | `Xenova/vit-base-patch16-224` |
| `object-detection` | no | `Xenova/detr-resnet-50` |
| `automatic-speech-recognition` | no | `onnx-community/whisper-tiny.en` |
| `text-to-speech` | no | `Xenova/speecht5_tts` |
| `translation` | no | `Xenova/nllb-200-distilled-600M` |
| `summarization` | no | `Xenova/distilbart-cnn-6-6` |
| `feature-extraction` | no | `Xenova/all-MiniLM-L6-v2` |
| `image-to-text` | no | `Xenova/vit-gpt2-image-captioning` |
| `zero-shot-classification` | no | `Xenova/mobilebert-uncased-mnli` |
| `fill-mask` | no | `Xenova/bert-base-uncased` |
| `question-answering` | no | `Xenova/distilbert-base-uncased-distilled-squad` |
| `token-classification` | no | `Xenova/bert-base-NER` |
| `depth-estimation` | no | `Xenova/depth-anything-small-hf` |
| `image-segmentation` | no | `Xenova/detr-resnet-50-panoptic` |

## How it works

```
Your App (main thread)          Web Worker
--------------------------      ---------------------------
ModelClient                     model-worker.ts
  .load()    --- postMessage -->  pipeline() from @hf/transformers
  .stream()  <-- tokens --------  TextStreamer + KV cache
  .run()     <-- result --------  One-shot inference
  .interrupt() -- signal ------>  InterruptableStoppingCriteria

TokenStream (AsyncIterable)     Singleton pipeline cache
GPURecovery (auto-reconnect)    WebGPU device management
```

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (falls back to WASM on older browsers)
- Node.js 18+ (WASM only, no WebGPU)
- `@huggingface/transformers` >= 4.0.0 as a peer dependency

## License

MIT — [Hemanth HM](https://h3manth.com)
