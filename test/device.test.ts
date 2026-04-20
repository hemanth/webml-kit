import { describe, it, expect } from 'vitest';
import { parseSize, formatSize, recommendDtype } from '../src/device.js';

describe('parseSize', () => {
  it('passes through raw numbers', () => {
    expect(parseSize(1024)).toBe(1024);
    expect(parseSize(0)).toBe(0);
  });

  it('parses GB', () => {
    expect(parseSize('4GB')).toBe(4 * 1024 ** 3);
    expect(parseSize('1.5GB')).toBe(Math.round(1.5 * 1024 ** 3));
  });

  it('parses MB', () => {
    expect(parseSize('512MB')).toBe(512 * 1024 ** 2);
    expect(parseSize('256mb')).toBe(256 * 1024 ** 2);
  });

  it('parses KB', () => {
    expect(parseSize('100KB')).toBe(100 * 1024);
  });

  it('parses TB', () => {
    expect(parseSize('1TB')).toBe(1024 ** 4);
  });

  it('parses bytes', () => {
    expect(parseSize('4096B')).toBe(4096);
  });

  it('handles spaces between number and unit', () => {
    expect(parseSize('4 GB')).toBe(4 * 1024 ** 3);
    expect(parseSize('512 MB')).toBe(512 * 1024 ** 2);
  });

  it('is case insensitive', () => {
    expect(parseSize('4gb')).toBe(parseSize('4GB'));
    expect(parseSize('4Gb')).toBe(parseSize('4GB'));
  });

  it('throws on invalid input', () => {
    expect(() => parseSize('4 gigglebytes')).toThrow('Invalid size format');
    expect(() => parseSize('big')).toThrow('Invalid size format');
    expect(() => parseSize('')).toThrow('Invalid size format');
  });
});

describe('formatSize', () => {
  it('formats TB', () => {
    expect(formatSize(1024 ** 4)).toBe('1.0 TB');
  });

  it('formats GB', () => {
    expect(formatSize(4 * 1024 ** 3)).toBe('4.0 GB');
    expect(formatSize(1.5 * 1024 ** 3)).toBe('1.5 GB');
  });

  it('formats MB', () => {
    expect(formatSize(512 * 1024 ** 2)).toBe('512.0 MB');
  });

  it('formats KB', () => {
    expect(formatSize(1024)).toBe('1.0 KB');
    expect(formatSize(1536)).toBe('1.5 KB');
  });

  it('formats bytes', () => {
    expect(formatSize(512)).toBe('512 B');
    expect(formatSize(0)).toBe('0 B');
  });

  it('roundtrips with parseSize', () => {
    const sizes = ['4GB', '512MB', '1TB', '100KB'];
    for (const s of sizes) {
      const bytes = parseSize(s);
      const formatted = formatSize(bytes);
      // Re-parse the formatted output and compare bytes
      const reparsed = parseSize(formatted.replace(' ', ''));
      expect(reparsed).toBe(bytes);
    }
  });
});

describe('recommendDtype', () => {
  it('recommends fp16 for 8GB+', () => {
    expect(recommendDtype(8 * 1024 ** 3)).toBe('fp16');
    expect(recommendDtype(16 * 1024 ** 3)).toBe('fp16');
  });

  it('recommends q8 for 4-8GB', () => {
    expect(recommendDtype(4 * 1024 ** 3)).toBe('q8');
    expect(recommendDtype(6 * 1024 ** 3)).toBe('q8');
  });

  it('recommends q4 for 2-4GB', () => {
    expect(recommendDtype(2 * 1024 ** 3)).toBe('q4');
    expect(recommendDtype(3 * 1024 ** 3)).toBe('q4');
  });

  it('recommends q4 for under 2GB', () => {
    expect(recommendDtype(1 * 1024 ** 3)).toBe('q4');
    expect(recommendDtype(0)).toBe('q4');
  });
});
