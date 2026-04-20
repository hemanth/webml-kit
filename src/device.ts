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

// ─── Size parsing / formatting ───

const SIZE_UNITS: Record<string, number> = {
  b: 1,
  kb: 1024,
  mb: 1024 ** 2,
  gb: 1024 ** 3,
  tb: 1024 ** 4,
};

/**
 * Parse a human-readable size string into bytes.
 *
 * Accepts formats like '4GB', '512 MB', '1.5gb', '256mb'.
 * Also accepts raw numbers (passed through as-is).
 *
 * ```ts
 * parseSize('4GB')    // 4294967296
 * parseSize('512MB')  // 536870912
 * parseSize('1.5gb')  // 1610612736
 * parseSize(1024)     // 1024
 * ```
 */
export function parseSize(input: string | number): number {
  if (typeof input === 'number') return input;

  const match = input.trim().match(/^([\d.]+)\s*(b|kb|mb|gb|tb)$/i);
  if (!match) {
    throw new Error(
      `Invalid size format: "${input}". Use something like "4GB", "512MB", or a number in bytes.`
    );
  }

  const value = parseFloat(match[1]);
  const unit = match[2].toLowerCase();
  return Math.round(value * SIZE_UNITS[unit]);
}

/**
 * Format bytes into a human-readable string.
 *
 * ```ts
 * formatSize(4294967296)  // '4 GB'
 * formatSize(536870912)   // '512 MB'
 * formatSize(1536)        // '1.5 KB'
 * ```
 */
export function formatSize(bytes: number): string {
  if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(1)} TB`;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024)      return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

// ─── canRun ───

/**
 * Check whether a model of a given size can likely run
 * on the current device without OOM.
 *
 * Accepts human-readable strings or raw byte counts:
 *
 * ```ts
 * await canRun('4GB');
 * await canRun('512MB');
 * await canRun(4_000_000_000);
 * ```
 *
 * This is a heuristic — real limits depend on browser, OS, and
 * other tabs consuming VRAM.
 */
export async function canRun(estimatedSize: string | number): Promise<{
  ok: boolean;
  backend: DeviceBackend;
  reason?: string;
}> {
  const bytes = parseSize(estimatedSize);
  const device = await detectDevice();

  if (device.backend === 'webgpu' && device.gpu) {
    // Leave ~500MB headroom for browser internals
    const available = device.gpu.vram - 512 * 1024 ** 2;
    if (bytes > available) {
      return {
        ok: false,
        backend: device.backend,
        reason: `Model (~${formatSize(bytes)}) exceeds available VRAM (~${formatSize(available)})`,
      };
    }
  }

  return { ok: true, backend: device.backend };
}
