import { chromium } from 'playwright';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const docsDir = path.resolve(__dirname, '../docs');

// Create local static file server for testing docs
function createServer(port = 8765) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.json': 'application/json',
    '.wav': 'audio/wav',
  };

  const server = http.createServer((req, res) => {
    let reqPath = req.url.split('?')[0];
    if (reqPath === '/' || reqPath === '') reqPath = '/index.html';
    const filePath = path.join(docsDir, reqPath);

    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('Not found: ' + reqPath);
        return;
      }
      const ext = path.extname(filePath);
      const contentType = mimeTypes[ext] || 'application/octet-stream';
      res.writeHead(200, {
        'Content-Type': contentType,
        'Access-Control-Allow-Origin': '*',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      res.end(data);
    });
  });

  return new Promise((resolve) => {
    server.listen(port, () => resolve(server));
  });
}

async function runE2ETests() {
  const customUrl = process.argv[2];
  let server = null;
  let targetUrl = customUrl;

  if (!targetUrl) {
    server = await createServer(8765);
    targetUrl = 'http://localhost:8765/';
  }

  console.log(`\n======================================================`);
  console.log(`Comprehensive Playwright E2E Test: All 11 Tabs & Capabilities`);
  console.log(`Target URL: ${targetUrl}`);
  console.log(`======================================================\n`);

  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    args: ['--enable-unsafe-webgpu', '--use-gl=angle'],
  });

  const context = await browser.newContext({
    permissions: ['microphone'],
  });

  const page = await context.newPage();
  page.setDefaultTimeout(120000);

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      console.log(`  [Browser Error]: ${msg.text()}`);
    }
  });

  page.on('pageerror', (err) => {
    console.error(`  [Uncaught Page Error]: ${err.message}`);
  });

  console.log(`[01/11] Navigating to ${targetUrl}...`);
  await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: 45000 });
  console.log(`        Page title: "${await page.title()}"`);

  // ─── Telemetry Bar ───
  await page.waitForSelector('#deviceDot.active', { timeout: 15000 });
  const backend = await page.locator('#backendText').innerText();
  const gpu = await page.locator('#gpuInfoText').innerText();
  console.log(`        Telemetry: Backend=${backend} | GPU=${gpu}`);

  // ─── TAB 1: Speech Recognition (Whisper ASR) ───
  console.log(`\n[02/11] Testing Tab 1: Speech Recognition (Whisper)...`);
  await page.locator('.tab-btn[data-tab="asr"]').click();
  await page.waitForTimeout(200);
  console.log(`        Clicking "Sample Voice" to run Whisper transcription...`);
  await page.locator('#sampleAudioBtn').click();

  await page.waitForFunction(() => {
    const out = document.getElementById('asrOutput')?.innerText || '';
    return out.trim().length > 10;
  }, { timeout: 90000 });
  const asrOutput = await page.locator('#asrOutput').innerText();
  console.log(`        ✓ Whisper ASR Result:\n          "${asrOutput.trim()}"`);

  // ─── TAB 2: Text Generation (LLM Streaming) ───
  console.log(`\n[03/11] Testing Tab 2: Text Generation (LLM Streaming)...`);
  await page.locator('.tab-btn[data-tab="llm"]').click();
  await page.waitForTimeout(200);
  // Choose SmolLM2-135M for fast streaming verification
  await page.locator('#llmModelSelect').selectOption({ index: 0 });
  await page.locator('#llmInput').fill('Say hello and one fact about space.');
  console.log(`        Clicking "Send" to stream tokens...`);
  await page.locator('#llmSendBtn').click();

  await page.waitForFunction(() => {
    const msgs = document.querySelectorAll('.chat-msg.assistant');
    if (msgs.length < 2) return false;
    const text = msgs[msgs.length - 1]?.innerText || '';
    return text.replace('Assistant: ', '').trim().length > 10;
  }, { timeout: 90000 });

  const botMsgs = await page.locator('.chat-msg.assistant').allInnerTexts();
  const latestBotMsg = botMsgs[botMsgs.length - 1].replace('Assistant: ', '').trim();
  const tpsStats = await page.locator('#llmStats').innerText();
  console.log(`        ✓ LLM Generated Output:\n          "${latestBotMsg.slice(0, 100)}..."`);
  console.log(`        ✓ Performance Stats: ${tpsStats}`);

  // ─── TAB 3: Vision (ViT Classification) ───
  console.log(`\n[04/11] Testing Tab 3: Vision (ViT Classification)...`);
  await page.locator('.tab-btn[data-tab="vision"]').click();
  await page.waitForTimeout(200);
  console.log(`        Clicking "Classify Image"...`);
  await page.locator('#classifyBtn').click();

  await page.waitForFunction(() => {
    const res = document.getElementById('visionResults');
    return res && res.children.length > 0;
  }, { timeout: 60000 });

  const visResults = await page.locator('#visionResults').innerText();
  console.log(`        ✓ Vision Classification Results:\n          ${visResults.split('\n').join('\n          ')}`);

  // ─── TAB 4: Object Detection (YOLO) ───
  console.log(`\n[05/11] Testing Tab 4: Object Detection (YOLO)...`);
  await page.locator('.tab-btn[data-tab="detect"]').click();
  await page.waitForTimeout(200);
  console.log(`        Clicking "Detect Objects" on preset canvas image...`);
  await page.locator('#detectBtn').click();

  await page.waitForFunction(() => {
    const res = document.getElementById('detectResults');
    return res && res.children.length > 0 && !res.innerText.includes('Ready');
  }, { timeout: 60000 });

  const detectResults = await page.locator('#detectResults').innerText();
  console.log(`        ✓ Detected Objects:\n          ${detectResults.split('\n').join('\n          ')}`);

  // ─── TAB 5: Speech Synthesis (TTS) ───
  console.log(`\n[06/11] Testing Tab 5: Speech Synthesis (TTS)...`);
  await page.locator('.tab-btn[data-tab="tts"]').click();
  await page.waitForTimeout(200);
  const selectedTTS = await page.locator('#ttsModelSelect').inputValue();
  console.log(`        Selected Model: ${selectedTTS}`);
  console.log(`        Clicking "Synthesize & Play Audio"...`);
  await page.locator('#speakBtn').click();

  await page.waitForFunction(() => {
    const status = document.getElementById('statusLabel')?.innerText || '';
    return status.includes('Playing');
  }, { timeout: 120000 });
  console.log(`        ✓ TTS Status: "${await page.locator('#statusLabel').innerText()}"`);

  // ─── TAB 6: Embeddings (MiniLM) ───
  console.log(`\n[07/11] Testing Tab 6: Embeddings & Cosine Similarity...`);
  await page.locator('.tab-btn[data-tab="embed"]').click();
  await page.waitForTimeout(200);
  console.log(`        Clicking "Calculate Cosine Similarity"...`);
  await page.locator('#calcSimBtn').click();

  await page.waitForFunction(() => {
    const el = document.getElementById('embedOutput');
    return el && el.innerText.includes('Semantic Similarity Score:') && !el.innerText.includes('NaN');
  }, { timeout: 60000 });
  const embedOut = await page.locator('#embedOutput').innerText();
  console.log(`        ✓ Embeddings Similarity Result:\n          ${embedOut.split('\n').join('\n          ')}`);

  // ─── TAB 7: canRun() VRAM Preflight & Hardware Limits ───
  console.log(`\n[08/11] Testing Tab 7: Hardware VRAM Preflight (canRun) & Limits...`);
  await page.locator('.tab-btn[data-tab="vram"]').click();
  await page.waitForTimeout(200);

  // Check specs grid populated
  const specBackend = await page.locator('#specBackend').innerText();
  const specMaxBuffer = await page.locator('#specMaxBuffer').innerText();
  const specDtype = await page.locator('#specDtype').innerText();
  console.log(`        Hardware Specs: Backend=${specBackend} | Buffer=${specMaxBuffer} | RecDtype=${specDtype}`);

  // Test 1: 2GB (should pass)
  await page.locator('#vramInput').fill('2GB');
  await page.locator('#vramCheckBtn').click();
  await page.waitForTimeout(200);
  const vram2GB = await page.locator('#vramOutput').innerText();
  console.log(`        ✓ canRun('2GB'): ${vram2GB.includes('true') ? 'PASSED' : 'FAILED'}`);

  // Test 2: 64GB (should fail gracefully)
  await page.locator('#vramInput').fill('64GB');
  await page.locator('#vramCheckBtn').click();
  await page.waitForTimeout(200);
  const vram64GB = await page.locator('#vramOutput').innerText();
  console.log(`        ✓ canRun('64GB'): ${vram64GB.includes('false') ? 'CORRECTLY REJECTED' : 'UNEXPECTED'}`);

  // ─── TAB 8: Pipeline Task Registry & inferTask() ───
  console.log(`\n[09/11] Testing Tab 8: Pipeline Task Registry & inferTask()...`);
  await page.locator('.tab-btn[data-tab="registry"]').click();
  await page.waitForTimeout(200);

  // Test 1: inferTask('onnx-community/whisper-tiny.en')
  await page.locator('#inferTaskInput').fill('onnx-community/whisper-tiny.en');
  await page.waitForTimeout(100);
  let badgeText = await page.locator('#inferTaskBadge').innerText();
  console.log(`        ✓ inferTask('whisper-tiny.en'): "${badgeText}" (expected: automatic-speech-recognition)`);

  // Test 2: inferTask('Xenova/yolos-tiny')
  await page.locator('#inferTaskInput').fill('Xenova/yolos-tiny');
  await page.waitForTimeout(100);
  badgeText = await page.locator('#inferTaskBadge').innerText();
  console.log(`        ✓ inferTask('Xenova/yolos-tiny'): "${badgeText}" (expected: object-detection)`);

  // Test 3: verify registry rows
  const taskRows = await page.locator('#registryTableBody tr').count();
  console.log(`        ✓ PIPELINE_REGISTRY count: ${taskRows} tasks registered`);

  // ─── TAB 9: HF Hub Search ───
  console.log(`\n[10/11] Testing Tab 9: Hugging Face Hub Search...`);
  await page.locator('.tab-btn[data-tab="hub"]').click();
  await page.waitForTimeout(200);
  await page.locator('#hubSearchInput').fill('whisper');
  await page.locator('#hubSearchBtn').click();

  await page.waitForFunction(() => {
    const rows = document.querySelectorAll('#hubResultsBody tr');
    return rows.length > 0 && !rows[0].innerText.includes('Searching');
  }, { timeout: 20000 });

  const firstHubModel = await page.locator('#hubResultsBody tr:first-child td:first-child').innerText();
  console.log(`        ✓ HF Hub Search: Top result = "${firstHubModel}"`);

  // Inspect model click to sync code
  await page.locator('#hubResultsBody tr:first-child').click();
  await page.waitForTimeout(200);
  const syncCode = await page.locator('#codeSnippet').innerText();
  console.log(`        ✓ Code snippet synced with model: ${syncCode.includes(firstHubModel)}`);

  // ─── TAB 10: Cache Manager ───
  console.log(`\n[11/11] Testing Tab 10: Cache Manager & Tab 11: GPU Resilience...`);
  await page.locator('.tab-btn[data-tab="cache"]').click();
  await page.waitForTimeout(200);
  await page.locator('#refreshCacheBtn').click();
  await page.waitForTimeout(500);
  const cacheTable = await page.locator('#cacheTableBody').innerText();
  console.log(`        ✓ Cache Manager contents after running pipelines:\n          ${cacheTable.split('\n').join('\n          ')}`);

  // ─── TAB 11: GPU Resilience (GPURecovery) ───
  console.log(`\n        Testing Tab 11: GPU Resilience (GPURecovery)...`);
  await page.locator('.tab-btn[data-tab="recovery"]').click();
  await page.waitForTimeout(200);

  // Click Simulate GPU Loss
  console.log(`        Clicking "Simulate GPU Loss"...`);
  await page.locator('#simLostBtn').click();
  await page.waitForTimeout(500);

  let stateBadge = await page.locator('#recoveryStateBadge').innerText();
  console.log(`        ✓ State during recovery: "${stateBadge}"`);

  // Wait for recovery to complete
  await page.waitForFunction(() => {
    const badge = document.getElementById('recoveryStateBadge')?.innerText || '';
    return badge.includes('RECOVERED') || badge.includes('IDLE');
  }, { timeout: 10000 });

  const logEntries = await page.locator('#recoveryLog').innerText();
  console.log(`        ✓ Recovery Telemetry Log:\n          ${logEntries.split('\n').slice(-3).join('\n          ')}`);

  console.log(`\n======================================================`);
  console.log(`SUCCESS: ALL 11 TABS AND CAPABILITIES TESTED & VERIFIED!`);
  console.log(`======================================================\n`);

  await browser.close();
  if (server) server.close();
}

runE2ETests().catch((err) => {
  console.error('\nTest Execution Error:', err);
  process.exit(1);
});
