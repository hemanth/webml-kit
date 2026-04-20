import { describe, it, expect } from 'vitest';
import { TokenStream, collectStream } from '../src/streaming.js';

describe('TokenStream', () => {
  it('yields pushed events in order', async () => {
    const stream = new TokenStream();
    const events = [
      { token: 'Hello', tps: 0, numTokens: 1, timeToFirstToken: 100 },
      { token: ' world', tps: 20, numTokens: 2, timeToFirstToken: 100 },
    ];

    // Push then end
    for (const e of events) stream.push(e);
    stream.end();

    const collected: typeof events = [];
    for await (const event of stream) {
      collected.push(event);
    }

    expect(collected).toEqual(events);
  });

  it('handles async push after iteration starts', async () => {
    const stream = new TokenStream();
    const tokens: string[] = [];

    const consumer = (async () => {
      for await (const event of stream) {
        tokens.push(event.token);
      }
    })();

    // Push tokens with delays
    stream.push({ token: 'A', tps: 0, numTokens: 1, timeToFirstToken: 50 });
    await new Promise(r => setTimeout(r, 10));
    stream.push({ token: 'B', tps: 10, numTokens: 2, timeToFirstToken: 50 });
    await new Promise(r => setTimeout(r, 10));
    stream.end();

    await consumer;
    expect(tokens).toEqual(['A', 'B']);
  });

  it('stops on abort', async () => {
    const stream = new TokenStream();
    const tokens: string[] = [];

    const consumer = (async () => {
      for await (const event of stream) {
        tokens.push(event.token);
      }
    })();

    stream.push({ token: 'A', tps: 0, numTokens: 1, timeToFirstToken: 50 });
    await new Promise(r => setTimeout(r, 10));
    stream.abort(new Error('interrupted'));

    await consumer;
    expect(tokens).toEqual(['A']);
    expect(stream.getError()?.message).toBe('interrupted');
  });

  it('works with AbortSignal', async () => {
    const controller = new AbortController();
    const stream = new TokenStream(controller.signal);
    const tokens: string[] = [];

    const consumer = (async () => {
      for await (const event of stream) {
        tokens.push(event.token);
      }
    })();

    stream.push({ token: 'A', tps: 0, numTokens: 1, timeToFirstToken: 50 });
    await new Promise(r => setTimeout(r, 10));
    controller.abort();
    // Give the abort handler time to fire
    await new Promise(r => setTimeout(r, 10));

    await consumer;
    expect(tokens).toEqual(['A']);
  });

  it('ignores pushes after end', async () => {
    const stream = new TokenStream();
    stream.push({ token: 'A', tps: 0, numTokens: 1, timeToFirstToken: 50 });
    stream.end();
    stream.push({ token: 'B', tps: 10, numTokens: 2, timeToFirstToken: 50 });

    const tokens: string[] = [];
    for await (const event of stream) {
      tokens.push(event.token);
    }

    expect(tokens).toEqual(['A']);
  });
});

describe('collectStream', () => {
  it('collects all tokens into a single string', async () => {
    const stream = new TokenStream();
    stream.push({ token: 'Hello', tps: 0, numTokens: 1, timeToFirstToken: 100 });
    stream.push({ token: ' world', tps: 30, numTokens: 2, timeToFirstToken: 100 });
    stream.push({ token: '!', tps: 35, numTokens: 3, timeToFirstToken: 100 });
    stream.end();

    const result = await collectStream(stream);
    expect(result.text).toBe('Hello world!');
    expect(result.numTokens).toBe(3);
    expect(result.tps).toBe(35);
    expect(result.timeToFirstToken).toBe(100);
  });

  it('throws on aborted stream', async () => {
    const stream = new TokenStream();
    stream.push({ token: 'A', tps: 0, numTokens: 1, timeToFirstToken: 50 });
    stream.abort(new Error('boom'));

    await expect(collectStream(stream)).rejects.toThrow('boom');
  });
});
