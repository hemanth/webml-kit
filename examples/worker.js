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
    const isExplicitOnnx = config.modelType === 'onnx' || config.modelId.endsWith('.onnx') || config.task === 'raw-onnx';
    if (isExplicitOnnx) {
      const onnxInstance = await loadStandaloneOnnx(config);
      pipelines.set(key, onnxInstance);
      self.postMessage({ type: 'ready', modelKey: key });
      return;
    }

    let pipelineInstance;
    try {
      pipelineInstance = await pipeline(config.task, config.modelId, {
        dtype: config.dtype ?? 'q4',
        device: config.device ?? 'webgpu',
        progress_callback: onProgress,
      });
    } catch (hfErr) {
      const errMsg = hfErr.message || String(hfErr);
      if (errMsg.includes('config.json') || errMsg.includes('Could not locate file') || errMsg.includes('Unsupported model')) {
        pipelineInstance = await loadStandaloneOnnx(config);
      } else {
        throw hfErr;
      }
    }

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

async function loadStandaloneOnnx(config) {
  const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/esm/ort.min.js');
  ort.env.wasm.simd = true;

  let modelUrl = config.modelId;
  let prepUrl = null;
  let vocab = {};

  if (!config.modelId.endsWith('.onnx') && !config.modelId.startsWith('http')) {
    // Query Hugging Face Hub
    try {
      const res = await fetch(`https://huggingface.co/api/models/${config.modelId}`);
      if (res.ok) {
        const data = await res.json();
        const files = (data.siblings || []).map(s => s.rfilename);
        const modelFile = files.find(f => f.endsWith('.onnx') && !f.includes('preprocess')) || files.find(f => f.endsWith('.onnx'));
        const prepFile = files.find(f => f.endsWith('.onnx') && f.includes('preprocess'));
        const vocabFile = files.find(f => f.endsWith('.json') && (f.includes('vocab') || f.includes('tokens')));

        if (modelFile) modelUrl = `https://huggingface.co/${config.modelId}/resolve/main/${modelFile}`;
        if (prepFile) prepUrl = `https://huggingface.co/${config.modelId}/resolve/main/${prepFile}`;
        if (vocabFile) {
          const vRes = await fetch(`https://huggingface.co/${config.modelId}/resolve/main/${vocabFile}`);
          if (vRes.ok) vocab = await vRes.json();
        }
      }
    } catch {}
  }

  // Create session with WebGPU -> WASM fallback
  let backend = config.device === 'wasm' ? 'wasm' : 'webgpu';
  let modelSession = null;
  let prepSession = null;

  try {
    modelSession = await ort.InferenceSession.create(modelUrl, { executionProviders: [backend] });
  } catch {
    backend = 'wasm';
    modelSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
  }

  if (prepUrl) {
    try {
      prepSession = await ort.InferenceSession.create(prepUrl, { executionProviders: [backend] });
    } catch {
      prepSession = await ort.InferenceSession.create(prepUrl, { executionProviders: ['wasm'] });
    }
  }

  if (config.task === 'automatic-speech-recognition') {
    return async function runASR(input) {
      let pcm = input instanceof Float32Array ? input : new Float32Array(input.buffer || input);

      const runInference = async (sess) => {
        let signal, len;
        if (prepSession) {
          const audioTensor = new ort.Tensor('float32', pcm, [1, pcm.length]);
          const lengthTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(pcm.length)]), [1]);
          const prepOut = await prepSession.run({ audio_signal: audioTensor, length: lengthTensor });
          signal = prepOut.processed_signal || Object.values(prepOut)[0];
          len = prepOut.processed_length || Object.values(prepOut)[1];
        } else {
          signal = new ort.Tensor('float32', pcm, [1, pcm.length]);
          len = new ort.Tensor('int64', BigInt64Array.from([BigInt(pcm.length)]), [1]);
        }

        const feeds = {};
        if (sess.inputNames.length >= 2) {
          feeds[sess.inputNames[0]] = signal;
          feeds[sess.inputNames[1]] = len;
        } else {
          feeds[sess.inputNames[0]] = signal;
        }

        const outputs = await sess.run(feeds);
        const logprobs = outputs.logprobs || Object.values(outputs)[0];
        const dims = logprobs.dims;
        const timeSteps = dims.length === 3 ? dims[1] : dims[0];
        const numClasses = dims[dims.length - 1];
        const data = logprobs.data;

        // Greedy CTC
        let prev = 0;
        let text = '';
        for (let t = 0; t < timeSteps; t++) {
          const off = t * numClasses;
          let maxVal = -Infinity;
          let argmax = 0;
          for (let c = 0; c < numClasses; c++) {
            if (data[off + c] > maxVal) {
              maxVal = data[off + c];
              argmax = c;
            }
          }
          if (argmax === 0) { prev = 0; continue; }
          if (argmax !== prev) {
            text += vocab[argmax] || '';
            prev = argmax;
          }
        }
        return { text: text.replace(/\u2581/g, ' ').replace(/\s+/g, ' ').trim() };
      };

      try {
        return await runInference(modelSession);
      } catch (err) {
        if (backend === 'webgpu') {
          backend = 'wasm';
          modelSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
          return await runInference(modelSession);
        }
        throw err;
      }
    };
  }

  // Generic ONNX runner
  return async function runGeneric(input) {
    const outputs = await modelSession.run(input);
    const res = {};
    for (const [k, v] of Object.entries(outputs)) res[k] = v.data;
    return res;
  };
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
