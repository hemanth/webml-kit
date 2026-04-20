/**
 * GPU device-lost recovery handler.
 *
 * WebGPU devices can be lost due to TDR resets, VRAM exhaustion, driver
 * updates, or mobile browsers reclaiming resources. This module provides
 * automatic recovery with exponential backoff.
 */

export type RecoveryState = 'idle' | 'lost' | 'recovering' | 'recovered' | 'failed';

export interface GPURecoveryEvents {
  'state-change': RecoveryState;
  'lost': { reason: string };
  'recovered': { adapter: GPUAdapter };
  'failed': { attempts: number; lastError: string };
}

export type GPURecoveryListener<K extends keyof GPURecoveryEvents> =
  (data: GPURecoveryEvents[K]) => void;

/**
 * Monitors a GPU device for loss and attempts automatic recovery.
 *
 * ```ts
 * const recovery = new GPURecovery();
 * recovery.on('lost', ({ reason }) => showBanner('GPU lost: ' + reason));
 * recovery.on('recovered', ({ adapter }) => reinitPipeline(adapter));
 * recovery.on('failed', () => showFallbackUI());
 *
 * const device = await recovery.watchDevice(existingDevice);
 * ```
 */
export class GPURecovery {
  private state: RecoveryState = 'idle';
  private listeners = new Map<string, Set<Function>>();
  private maxRetries: number;
  private baseDelay: number;

  constructor(options?: { maxRetries?: number; baseDelayMs?: number }) {
    this.maxRetries = options?.maxRetries ?? 3;
    this.baseDelay = options?.baseDelayMs ?? 1000;
  }

  /** Current recovery state. */
  getState(): RecoveryState {
    return this.state;
  }

  /** Register a listener for recovery events. */
  on<K extends keyof GPURecoveryEvents>(
    event: K,
    listener: GPURecoveryListener<K>,
  ): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);
  }

  /** Remove a listener. */
  off<K extends keyof GPURecoveryEvents>(
    event: K,
    listener: GPURecoveryListener<K>,
  ): void {
    this.listeners.get(event)?.delete(listener);
  }

  private emit<K extends keyof GPURecoveryEvents>(
    event: K,
    data: GPURecoveryEvents[K],
  ): void {
    this.listeners.get(event)?.forEach(fn => (fn as GPURecoveryListener<K>)(data));
  }

  private setState(state: RecoveryState): void {
    this.state = state;
    this.emit('state-change', state);
  }

  /**
   * Watch a GPU device for loss. When the device is lost, automatically
   * attempt to re-acquire an adapter.
   *
   * @returns The same device (for chaining)
   */
  watchDevice(device: GPUDevice): GPUDevice {
    device.lost.then((info: GPUDeviceLostInfo) => {
      const reason = info.message || 'unknown';

      // If the device was intentionally destroyed, don't recover
      if (info.reason === 'destroyed') {
        this.setState('idle');
        return;
      }

      this.setState('lost');
      this.emit('lost', { reason });
      this.attemptRecovery(0);
    });

    this.setState('idle');
    return device;
  }

  private async attemptRecovery(attempt: number): Promise<void> {
    if (attempt >= this.maxRetries) {
      this.setState('failed');
      this.emit('failed', {
        attempts: attempt,
        lastError: 'Max retries exceeded',
      });
      return;
    }

    this.setState('recovering');

    // Exponential backoff: 1s, 2s, 4s
    const delay = this.baseDelay * 2 ** attempt;
    await new Promise(r => setTimeout(r, delay));

    try {
      if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        throw new Error('WebGPU not available');
      }

      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!adapter) {
        throw new Error('No GPU adapter available');
      }

      this.setState('recovered');
      this.emit('recovered', { adapter });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // Retry
      this.attemptRecovery(attempt + 1);
    }
  }
}
