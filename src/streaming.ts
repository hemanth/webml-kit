/**
 * Token streaming utilities for text generation.
 *
 * Converts the Web Worker's `postMessage`-based token stream into a proper
 * `AsyncIterable` with TPS measurement, time-to-first-token tracking,
 * and AbortSignal support.
 */

import type { TokenEvent } from './types.js';

/**
 * A readable token stream that implements `AsyncIterable`.
 *
 * ```ts
 * const stream = new TokenStream();
 *
 * // In your worker message handler:
 * stream.push({ token: 'Hello', tps: 42, numTokens: 1, timeToFirstToken: 120 });
 *
 * // In your app:
 * for await (const event of stream) {
 *   process.stdout.write(event.token);
 * }
 * ```
 */
export class TokenStream implements AsyncIterable<TokenEvent> {
  private queue: TokenEvent[] = [];
  private resolve: ((value: IteratorResult<TokenEvent>) => void) | null = null;
  private done = false;
  private error: Error | null = null;
  private abortHandler: (() => void) | null = null;

  constructor(signal?: AbortSignal) {
    if (signal) {
      this.abortHandler = () => {
        this.abort(new Error('Stream aborted'));
      };
      signal.addEventListener('abort', this.abortHandler, { once: true });
    }
  }

  /** Push a new token event into the stream. */
  push(event: TokenEvent): void {
    if (this.done) return;

    if (this.resolve) {
      // A consumer is waiting — deliver immediately
      const r = this.resolve;
      this.resolve = null;
      r({ value: event, done: false });
    } else {
      // Buffer until consumed
      this.queue.push(event);
    }
  }

  /** Signal that generation is complete. */
  end(): void {
    this.done = true;
    this.cleanup();
    if (this.resolve) {
      const r = this.resolve;
      this.resolve = null;
      r({ value: undefined as unknown as TokenEvent, done: true });
    }
  }

  /** Signal an error. */
  abort(error: Error): void {
    this.error = error;
    this.done = true;
    this.cleanup();
    if (this.resolve) {
      const r = this.resolve;
      this.resolve = null;
      r({ value: undefined as unknown as TokenEvent, done: true });
    }
  }

  /** Get the accumulated error, if any. */
  getError(): Error | null {
    return this.error;
  }

  private cleanup(): void {
    // Remove abort listener to prevent leaks
    if (this.abortHandler) {
      this.abortHandler = null;
    }
  }

  // ─── AsyncIterable implementation ───

  [Symbol.asyncIterator](): AsyncIterator<TokenEvent> {
    return {
      next: (): Promise<IteratorResult<TokenEvent>> => {
        // If there's a buffered event, return it immediately
        if (this.queue.length > 0) {
          return Promise.resolve({
            value: this.queue.shift()!,
            done: false,
          });
        }

        // If the stream is done, signal completion
        if (this.done) {
          return Promise.resolve({
            value: undefined as unknown as TokenEvent,
            done: true,
          });
        }

        // Otherwise, wait for the next push
        return new Promise((resolve) => {
          this.resolve = resolve;
        });
      },

      return: (): Promise<IteratorResult<TokenEvent>> => {
        this.done = true;
        this.cleanup();
        return Promise.resolve({
          value: undefined as unknown as TokenEvent,
          done: true,
        });
      },
    };
  }
}

/**
 * Collect a token stream into a single string result.
 *
 * ```ts
 * const { text, tps, numTokens } = await collectStream(stream);
 * ```
 */
export async function collectStream(stream: TokenStream): Promise<{
  text: string;
  tps: number;
  numTokens: number;
  timeToFirstToken: number;
}> {
  let text = '';
  let tps = 0;
  let numTokens = 0;
  let timeToFirstToken = 0;

  for await (const event of stream) {
    text += event.token;
    tps = event.tps;
    numTokens = event.numTokens;
    if (timeToFirstToken === 0) {
      timeToFirstToken = event.timeToFirstToken;
    }
  }

  const error = stream.getError();
  if (error) throw error;

  return { text, tps, numTokens, timeToFirstToken };
}
