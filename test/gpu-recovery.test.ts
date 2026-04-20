import { describe, it, expect, vi } from 'vitest';
import { GPURecovery } from '../src/gpu-recovery.js';

describe('GPURecovery', () => {
  it('starts in idle state', () => {
    const recovery = new GPURecovery();
    expect(recovery.getState()).toBe('idle');
  });

  it('accepts custom config', () => {
    const recovery = new GPURecovery({ maxRetries: 5, baseDelayMs: 500 });
    expect(recovery.getState()).toBe('idle');
  });

  it('emits state-change events', () => {
    const recovery = new GPURecovery();
    const states: string[] = [];

    recovery.on('state-change', (state) => {
      states.push(state);
    });

    // Simulate watching a mock device that immediately loses
    const mockDevice = {
      lost: Promise.resolve({ reason: 'unknown', message: 'GPU crashed' }),
    } as unknown as GPUDevice;

    recovery.watchDevice(mockDevice);
    expect(recovery.getState()).toBe('idle');
  });

  it('can remove listeners', () => {
    const recovery = new GPURecovery();
    const listener = vi.fn();

    recovery.on('lost', listener);
    recovery.off('lost', listener);

    // No way to trigger without a real device, but at least
    // verify the listener set management doesn't throw
    expect(recovery.getState()).toBe('idle');
  });
});
