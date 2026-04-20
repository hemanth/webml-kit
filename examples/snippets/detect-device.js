/**
 * Detect what the user's machine can handle.
 *
 * Works without downloading any model — just probes
 * the browser's WebGPU/WASM capabilities.
 *
 * Usage: import into your app and call before showing
 * model selection UI or progress bars.
 */

import { detectDevice, canRun } from 'webml-kit';

const device = await detectDevice();

console.log('backend:', device.backend);         // 'webgpu' | 'wasm' | 'cpu'
console.log('dtype:', device.recommendedDtype);   // 'fp16' | 'q8' | 'q4'

if (device.gpu) {
  console.log('gpu vendor:', device.gpu.vendor);
  console.log('gpu arch:', device.gpu.architecture);
  console.log('vram:', device.gpu.vramFormatted);  // '8.0 GB'
}

// canRun accepts both human-readable strings and raw byte counts
const check = await canRun('4GB');
console.log('can run 4GB model?', check.ok);
if (!check.ok) console.log('reason:', check.reason);

const same = await canRun(4_000_000_000);          // same as above
console.log('same result:', same.ok);
