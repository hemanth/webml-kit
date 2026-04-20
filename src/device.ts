/**
 * WebGPU / WASM device detection and capability probing.
 *
 * Detects the best available backend, extracts GPU adapter info,
 * and recommends a quantization level based on estimated VRAM.
 */

import type { DeviceBackend, DeviceInfo, GPUInfo, QuantizationType } from './types.js';

// ─── VRAM thresholds for dtype recommendation ───

const VRAM_THRESHOLDS: Array<{ min: number; dtype: QuantizationType }> = [
  { min: 8 * 1024 ** 3, dtype: 'fp16' },   // 8 GB+  → fp16
  { min: 4 * 1024 ** 3, dtype: 'q8' },     // 4 GB+  → q8
  { min: 2 * 1024 ** 3, dtype: 'q4' },     // 2 GB+  → q4
  { min: 0,             dtype: 'q4' },      // < 2 GB → q4 (safest)
];

/**
 * Check if WebGPU is available and request an adapter.
 * Returns `null` if WebGPU is not supported.
 */
export async function getGPUAdapter(): Promise<GPUAdapter | null> {
  if (typeof navigator === 'undefined') return null;
  if (!('gpu' in navigator)) return null;

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    return adapter;
  } catch {
    return null;
  }
}

/**
 * Extract GPU info from an adapter.
 */
export async function getGPUInfo(adapter: GPUAdapter): Promise<GPUInfo> {
  // adapterInfo is available synchronously on modern browsers
  const info = adapter.info ?? (adapter as unknown as { requestAdapterInfo?: () => Promise<GPUAdapterInfo> }).requestAdapterInfo?.();
  const resolved = info instanceof Promise ? await info : info;

  return {
    vendor: resolved?.vendor ?? 'unknown',
    architecture: resolved?.architecture ?? 'unknown',
    description: resolved?.description ?? 'unknown',
    // maxBufferSize gives a rough VRAM lower-bound
    vram: Number(adapter.limits?.maxBufferSize ?? 0),
  };
}

/**
 * Recommend a quantization dtype based on available VRAM.
 */
export function recommendDtype(vram: number): QuantizationType {
  for (const { min, dtype } of VRAM_THRESHOLDS) {
    if (vram >= min) return dtype;
  }
  return 'q4';
}

/**
 * Check if WebGPU is available.
 */
export async function checkWebGPU(): Promise<boolean> {
  const adapter = await getGPUAdapter();
  return adapter !== null;
}

/**
 * Check if WebAssembly is available.
 */
export function checkWASM(): boolean {
  return typeof WebAssembly !== 'undefined';
}

/**
 * Detect the best available compute backend and return device info.
 *
 * Priority: WebGPU → WASM → CPU
 *
 * ```ts
 * const info = await detectDevice();
 * console.log(info.backend);         // 'webgpu'
 * console.log(info.recommendedDtype); // 'q4'
 * console.log(info.gpu?.vendor);      // 'apple'
 * ```
 */
export async function detectDevice(): Promise<DeviceInfo> {
  // Try WebGPU first
  const adapter = await getGPUAdapter();
  if (adapter) {
    const gpu = await getGPUInfo(adapter);
    return {
      backend: 'webgpu',
      gpu,
      recommendedDtype: recommendDtype(gpu.vram),
    };
  }

  // Fall back to WASM
  if (checkWASM()) {
    return {
      backend: 'wasm',
      gpu: null,
      recommendedDtype: 'q4',
    };
  }

  // CPU-only (rare)
  return {
    backend: 'cpu',
    gpu: null,
    recommendedDtype: 'q4',
  };
}

/**
 * Check whether a model of a given size (bytes) can likely run
 * on the current device without OOM.
 *
 * This is a heuristic — real limits depend on browser, OS, and
 * other tabs consuming VRAM.
 */
export async function canRun(estimatedModelBytes: number): Promise<{
  ok: boolean;
  backend: DeviceBackend;
  reason?: string;
}> {
  const device = await detectDevice();

  if (device.backend === 'webgpu' && device.gpu) {
    // Leave ~500MB headroom for browser internals
    const available = device.gpu.vram - 512 * 1024 ** 2;
    if (estimatedModelBytes > available) {
      return {
        ok: false,
        backend: device.backend,
        reason: `Model (~${mb(estimatedModelBytes)}) exceeds available VRAM (~${mb(available)})`,
      };
    }
  }

  return { ok: true, backend: device.backend };
}

function mb(bytes: number): string {
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}
