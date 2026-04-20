/**
 * Worker entry point for standalone examples.
 * 
 * This pulls @huggingface/transformers from esm.sh (a CDN that serves
 * ES modules) so the examples work without a build step or npm install.
 *
 * In a real project you'd use the built dist/model-worker.js instead.
 */

import {
  pipeline,
  env,
  TextStreamer,
  InterruptableStoppingCriteria,
  AutoTokenizer,
} from 'https://esm.sh/@huggingface/transformers@4.1.0';

// Disable local model caching path (use browser Cache API)
env.allowLocalModels = false;

// ─── State ───

const pipelines = new Map();
let stoppingCriteria = null;

// ─── Device detection ───

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
        gpu = {
          vendor: info?.vendor ?? 'unknown',
          architecture: info?.architecture ?? 'unknown',
          description: info?.description ?? 'unknown',
          vram: Number(adapter.limits?.maxBufferSize ?? 0),
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

// ─── Progress callback ───

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
  } else if (event.status === 'initiate' || event.status === 'ready') {
    self.postMessage({
      type: 'progress',
      data: {
        status: event.status,
        loaded: 0,
        total: 0,
        percent: event.status === 'ready' ? 100 : 0,
      },
    });
  }
}

// ─── Load pipeline ───

async function loadPipeline(config) {
  const key = `${config.task}::${config.modelId}`;

  if (pipelines.has(key)) {
    self.postMessage({ type: 'ready', modelKey: key });
    return;
  }

  try {
    const pipelineInstance = await pipeline(config.task, config.modelId, {
      dtype: config.dtype ?? 'q4',
      device: config.device ?? 'webgpu',
      progress_callback: onProgress,
    });

    pipelines.set(key, pipelineInstance);

    // Warmup for text-generation
    if (config.task === 'text-generation' && pipelineInstance.tokenizer) {
      try {
        const warmupInputs = pipelineInstance.tokenizer('warm');
        await pipelineInstance.model.generate({
          ...warmupInputs,
          max_new_tokens: 1,
        });
      } catch {}
    }

    self.postMessage({ type: 'ready', modelKey: key });
  } catch (err) {
    self.postMessage({
      type: 'error',
      id: 'load',
      data: err.message || String(err),
    });
  }
}

// ─── Run inference ───

async function runInference(id, task, input, options = {}) {
  // Find the right pipeline
  let pipelineInstance = null;
  for (const [key, p] of pipelines) {
    if (key.startsWith(task + '::')) {
      pipelineInstance = p;
      break;
    }
  }

  if (!pipelineInstance) {
    self.postMessage({
      type: 'error',
      id,
      data: `No pipeline loaded for task "${task}"`,
    });
    return;
  }

  try {
    if (task === 'text-generation') {
      await runTextGeneration(id, pipelineInstance, input, options);
    } else {
      const result = await pipelineInstance(input, options);
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

async function runTextGeneration(id, generator, input, options) {
  stoppingCriteria = new InterruptableStoppingCriteria();

  // Build messages
  let messages;
  if (typeof input === 'string') {
    messages = [{ role: 'user', content: input }];
  } else if (Array.isArray(input)) {
    messages = input;
  } else {
    messages = [{ role: 'user', content: String(input) }];
  }

  const tokenizer = generator.tokenizer;
  const promptText = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });

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
    max_new_tokens: options.maxTokens ?? 512,
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

// ─── Message handler ───

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

    case 'reset':
      // Clear KV cache by recreating — simplified for examples
      break;

    case 'dispose': {
      if (msg.modelKey) {
        const p = pipelines.get(msg.modelKey);
        if (p?.dispose) p.dispose();
        pipelines.delete(msg.modelKey);
      } else {
        for (const [, p] of pipelines) {
          if (p?.dispose) p.dispose();
        }
        pipelines.clear();
      }
      break;
    }
  }
});
