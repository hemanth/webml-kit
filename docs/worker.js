/**
 * webml-kit Playground Worker
 * Full multi-task pipeline dispatcher with Promise caching and AutoProcessor support.
 */

import {
  pipeline,
  env,
  TextStreamer,
  InterruptableStoppingCriteria,
} from 'https://esm.sh/@huggingface/transformers@4.1.0';

// Configure transformers environment
env.allowLocalModels = false;

// Store promises of pipelines to prevent race conditions during downloading
const instances = new Map();
let stoppingCriteria = null;

// Device check
async function checkDevice() {
  let backend = 'cpu';
  let gpu = null;

  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (adapter) {
        backend = 'webgpu';
        const info = adapter.info;
        const vram = Number(adapter.limits?.maxBufferSize ?? 0);
        const fmtVram = vram >= 1024 ** 3
          ? `${(vram / 1024 ** 3).toFixed(1)} GB`
          : `${(vram / 1024 ** 2).toFixed(0)} MB`;
        gpu = {
          vendor: info?.vendor ?? 'unknown',
          architecture: info?.architecture ?? 'unknown',
          description: info?.description ?? 'unknown',
          vram,
          vramFormatted: fmtVram,
        };
      }
    }
  } catch {}

  if (backend === 'cpu' && typeof WebAssembly !== 'undefined') {
    backend = 'wasm';
  }

  const vram = gpu?.vram ?? 0;
  const recommendedDtype = vram >= 8e9 ? 'fp16' : vram >= 4e9 ? 'q8' : 'q4';

  return { backend, gpu, recommendedDtype };
}

// Progress reporting
function onProgress(event) {
  if (event.status === 'progress' || event.status === 'download') {
    const percent = event.progress != null
      ? Math.round(event.progress)
      : (event.loaded && event.total ? Math.round((event.loaded / event.total) * 100) : 0);

    self.postMessage({
      type: 'progress',
      data: {
        status: event.status,
        loaded: event.loaded ?? 0,
        total: event.total ?? 0,
        percent,
      },
    });
  } else if (event.status === 'initiate' || event.status === 'ready' || event.status === 'done') {
    self.postMessage({
      type: 'progress',
      data: {
        status: event.status,
        loaded: 0,
        total: 0,
        percent: event.status === 'ready' || event.status === 'done' ? 100 : 0,
      },
    });
  }
}

// Pipeline loader
async function loadPipeline(config) {
  const key = `${config.task}::${config.modelId}`;

  if (!instances.has(key)) {
    const promise = (async () => {
      const pipe = await pipeline(config.task, config.modelId, {
        dtype: config.dtype ?? 'q4',
        device: config.device ?? 'webgpu',
        progress_callback: onProgress,
      });

      // Special handling for SpeechT5
      if (config.task === 'text-to-speech' && config.modelId.includes('speecht5')) {
        try {
          const { AutoProcessor } = await import('https://esm.sh/@huggingface/transformers@4.1.0');
          pipe.processor = await AutoProcessor.from_pretrained(config.modelId);
        } catch (procErr) {
          console.warn('AutoProcessor load error:', procErr);
        }
      }

      // Warmup for text generation
      if (config.task === 'text-generation' && pipe.tokenizer) {
        try {
          const warmupInputs = pipe.tokenizer('warm');
          await pipe.model.generate({ ...warmupInputs, max_new_tokens: 1 });
        } catch {}
      }

      return pipe;
    })();

    instances.set(key, promise);
  }

  try {
    await instances.get(key);
    self.postMessage({ type: 'ready', modelKey: key });
  } catch (err) {
    instances.delete(key);
    self.postMessage({
      type: 'error',
      id: 'load',
      data: err.message || String(err),
    });
  }
}

// Text Generation with Streaming
async function runTextGeneration(id, generator, input, options = {}) {
  stoppingCriteria = new InterruptableStoppingCriteria();

  let messages;
  if (typeof input === 'string') {
    messages = [{ role: 'user', content: input }];
  } else if (Array.isArray(input)) {
    messages = input;
  } else {
    messages = [{ role: 'user', content: String(input) }];
  }

  const tokenizer = generator.tokenizer;
  let promptText = typeof input === 'string' ? input : '';
  if (tokenizer && typeof tokenizer.apply_chat_template === 'function') {
    try {
      promptText = tokenizer.apply_chat_template(messages, {
        tokenize: false,
        add_generation_prompt: true,
      });
    } catch {
      promptText = typeof input === 'string' ? input : JSON.stringify(input);
    }
  }

  let numTokens = 0;
  let tps = 0;
  let startTime = 0;

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      numTokens++;
      if (startTime === 0) startTime = performance.now();

      const elapsed = (performance.now() - startTime) / 1000;
      tps = elapsed > 0 ? numTokens / elapsed : 0;

      self.postMessage({
        type: 'token',
        id,
        data: {
          token,
          tps,
          numTokens,
          timeToFirstToken: numTokens === 1 ? performance.now() - startTime : 0,
        },
      });
    },
  });

  const result = await generator(promptText, {
    max_new_tokens: options.maxTokens ?? 256,
    temperature: options.temperature ?? 0.7,
    top_p: options.topP ?? 0.9,
    do_sample: true,
    streamer,
    stopping_criteria: stoppingCriteria,
  });

  self.postMessage({
    type: 'result',
    id,
    data: { text: result[0]?.generated_text ?? '', tps, numTokens },
  });
}

// Run inference dispatcher
async function runInference(id, task, input, options = {}) {
  let pipePromise = null;
  let matchedKey = '';

  for (const [key, promise] of instances) {
    if (key.startsWith(task + '::')) {
      pipePromise = promise;
      matchedKey = key;
      break;
    }
  }

  if (!pipePromise) {
    self.postMessage({
      type: 'error',
      id,
      data: `No pipeline loaded for task "${task}". Please load a model first.`,
    });
    return;
  }

  try {
    const pipe = await pipePromise;

    if (task === 'text-generation') {
      await runTextGeneration(id, pipe, input, options);
    } else if (task === 'feature-extraction') {
      const output = await pipe(input, { pooling: 'mean', normalize: true, ...options });
      const data = output.tolist ? output.tolist() : Array.from(output.data || []);
      self.postMessage({ type: 'result', id, data });
    } else if (task === 'text-to-speech') {
      let output;
      if (matchedKey.includes('speecht5')) {
        const speakerEmbeddings = options.speaker_embeddings ||
          'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin';
        output = await pipe(input, { speaker_embeddings: speakerEmbeddings, ...options });
      } else {
        output = await pipe(input, options);
      }

      if (output && output.audio) {
        self.postMessage({
          type: 'result',
          id,
          data: {
            audio: Array.from(output.audio),
            sampling_rate: output.sampling_rate || 16000,
          },
        });
      } else {
        throw new Error('Text-to-speech model did not return an audio waveform');
      }
    } else {
      const result = await pipe(input, options);
      self.postMessage({ type: 'result', id, data: result });
    }
  } catch (err) {
    self.postMessage({
      type: 'error',
      id,
      data: err.message || String(err),
    });
  }
}

// Event router
self.addEventListener('message', async (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'check': {
      const device = await checkDevice();
      self.postMessage({ type: 'device-info', data: device });
      break;
    }
    case 'load':
      await loadPipeline(msg.config);
      break;
    case 'run':
      await runInference(msg.id, msg.task, msg.input, msg.options);
      break;
    case 'interrupt':
      stoppingCriteria?.interrupt();
      break;
    case 'dispose': {
      if (msg.modelKey) {
        const p = instances.get(msg.modelKey);
        if (p) p.then(inst => inst?.dispose?.());
        instances.delete(msg.modelKey);
      } else {
        for (const [, p] of instances) {
          p.then(inst => inst?.dispose?.());
        }
        instances.clear();
      }
      break;
    }
  }
});
