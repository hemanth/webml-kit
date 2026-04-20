/**
 * Detect what the user's machine can handle.
 *
 * Works without downloading any model — just probes
 * the browser's WebGPU/WASM capabilities.
 *
 * Usage: import into your app and call before showing
 * model selection UI or progress bars.
 */

import { detectDevice, canRun, formatSize } from 'webml-utils';

const device = await detectDevice();

console.log('backend:', device.backend);         // 'webgpu' | 'wasm' | 'cpu'
console.log('dtype:', device.recommendedDtype);   // 'fp16' | 'q8' | 'q4'

if (device.gpu) {
  console.log('gpu vendor:', device.gpu.vendor);
  console.log('gpu arch:', device.gpu.architecture);
  console.log('vram:', formatSize(device.gpu.vram));
}

// Can a 4GB model fit?
const check = await canRun('4GB');
console.log('can run 4GB model?', check.ok);
if (!check.ok) console.log('reason:', check.reason);

// Can a 512MB model fit?
const small = await canRun('512MB');
console.log('can run 512MB model?', small.ok);
