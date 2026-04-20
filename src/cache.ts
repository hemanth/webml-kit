/**
 * Model cache visibility utilities.
 *
 * Transformers.js uses the Cache API internally to persist downloaded model
 * files. This module gives users visibility into what's cached, letting them
 * skip "downloading…" UX for returning users or clear storage on mobile.
 */

import type { CachedModel, CacheBackend } from './types.js';

/** HuggingFace Transformers.js cache name prefix */
const HF_CACHE_PREFIX = 'transformers-cache';

/**
 * Detect which cache backend is available.
 * Priority: Cache API → OPFS → IndexedDB
 */
export async function getCacheBackend(): Promise<CacheBackend> {
  if (typeof caches !== 'undefined') {
    return 'cache-api';
  }

  if (typeof navigator !== 'undefined' && 'storage' in navigator) {
    try {
      const root = await navigator.storage.getDirectory();
      if (root) return 'opfs';
    } catch {
      // OPFS not available
    }
  }

  if (typeof indexedDB !== 'undefined') {
    return 'indexeddb';
  }

  return 'cache-api'; // fallback, will fail gracefully
}

/**
 * Check if a specific model is already cached locally.
 *
 * ```ts
 * if (await isCached('onnx-community/Bonsai-1.7B-ONNX')) {
 *   // Skip "downloading..." UI
 * }
 * ```
 */
export async function isCached(modelId: string): Promise<boolean> {
  if (typeof caches === 'undefined') return false;

  try {
    const keys = await caches.keys();
    const hfCaches = keys.filter(k => k.startsWith(HF_CACHE_PREFIX));

    for (const cacheName of hfCaches) {
      const cache = await caches.open(cacheName);
      const cacheKeys = await cache.keys();
      const hasModel = cacheKeys.some(req =>
        req.url.includes(encodeURIComponent(modelId)) || req.url.includes(modelId)
      );
      if (hasModel) return true;
    }
  } catch {
    // Cache API not available or permission denied
  }

  return false;
}

/**
 * Get total cache size in bytes used by downloaded models.
 */
export async function getCacheSize(): Promise<number> {
  if (typeof navigator === 'undefined' || !('storage' in navigator)) return 0;

  try {
    const estimate = await navigator.storage.estimate();
    return estimate.usage ?? 0;
  } catch {
    return 0;
  }
}

/**
 * List all cached models with metadata.
 */
export async function listCachedModels(): Promise<CachedModel[]> {
  if (typeof caches === 'undefined') return [];

  const models: CachedModel[] = [];

  try {
    const keys = await caches.keys();
    const hfCaches = keys.filter(k => k.startsWith(HF_CACHE_PREFIX));

    for (const cacheName of hfCaches) {
      const cache = await caches.open(cacheName);
      const cacheKeys = await cache.keys();

      // Group by model ID (extract from URL pattern)
      const modelUrls = new Map<string, number>();

      for (const request of cacheKeys) {
        const url = request.url;
        // HF URLs: https://huggingface.co/{org}/{model}/resolve/{rev}/{file}
        const match = url.match(/huggingface\.co\/([^/]+\/[^/]+)\//);
        if (match) {
          const id = match[1];
          const response = await cache.match(request);
          const size = response
            ? Number(response.headers.get('content-length') ?? 0)
            : 0;
          modelUrls.set(id, (modelUrls.get(id) ?? 0) + size);
        }
      }

      for (const [modelId, sizeBytes] of modelUrls) {
        models.push({
          modelId,
          sizeBytes,
          lastAccessed: new Date(), // Cache API doesn't track this
        });
      }
    }
  } catch {
    // Cache API not available
  }

  return models;
}

/**
 * Clear cached model files.
 *
 * @param modelId - Specific model to clear, or omit to clear all.
 *
 * ```ts
 * // Clear a specific model
 * await clearCache('onnx-community/Bonsai-1.7B-ONNX');
 *
 * // Clear all cached models
 * await clearCache();
 * ```
 */
export async function clearCache(modelId?: string): Promise<void> {
  if (typeof caches === 'undefined') return;

  try {
    const keys = await caches.keys();
    const hfCaches = keys.filter(k => k.startsWith(HF_CACHE_PREFIX));

    for (const cacheName of hfCaches) {
      if (!modelId) {
        // Clear entire cache
        await caches.delete(cacheName);
      } else {
        // Clear specific model
        const cache = await caches.open(cacheName);
        const cacheKeys = await cache.keys();
        for (const request of cacheKeys) {
          if (
            request.url.includes(encodeURIComponent(modelId)) ||
            request.url.includes(modelId)
          ) {
            await cache.delete(request);
          }
        }
      }
    }
  } catch {
    // Cache API not available
  }
}
