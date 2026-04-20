/**
 * Search and list WebGPU/ONNX-compatible models from the Hugging Face Hub.
 *
 * Uses the public HF Hub API — no auth token needed for public models.
 */

import type { PipelineTask } from './types.js';

// ─── Types ───

export interface HubModel {
  /** Model ID (e.g. 'onnx-community/Bonsai-1.7B-ONNX') */
  modelId: string;
  /** Pipeline task (e.g. 'text-generation') */
  task: PipelineTask | string;
  /** Author or organization */
  author: string;
  /** Number of downloads in the last month */
  downloads: number;
  /** Number of likes */
  likes: number;
  /** Last modified date */
  lastModified: string;
  /** Tags on the model (e.g. ['onnx', 'webgpu']) */
  tags: string[];
  /** Whether it's likely WebGPU-compatible (has ONNX weights) */
  webgpuCompatible: boolean;
}

export interface SearchOptions {
  /** Filter by pipeline task */
  task?: PipelineTask | string;
  /** Free-text search query */
  query?: string;
  /** Sort by: 'downloads' | 'likes' | 'modified' | 'trending' */
  sort?: 'downloads' | 'likes' | 'modified' | 'trending';
  /** Sort direction */
  direction?: 'asc' | 'desc';
  /** Max results to return (default: 20, max: 100) */
  limit?: number;
  /** Filter by author/org (e.g. 'onnx-community', 'Xenova') */
  author?: string;
}

// ─── Constants ───

const HF_API = 'https://huggingface.co/api';

/** Known orgs that publish ONNX/WebGPU-ready models */
export const WEBGPU_ORGS = [
  'onnx-community',
  'Xenova',
  'webml-community',
] as const;

// ─── Core functions ───

/**
 * Search for models compatible with transformers.js (and thus WebGPU/WASM).
 *
 * ```ts
 * import { searchModels } from 'webml-kit';
 *
 * const models = await searchModels({ task: 'text-generation', limit: 5 });
 * for (const m of models) {
 *   console.log(`${m.modelId} — ${m.downloads} downloads`);
 * }
 * ```
 */
export async function searchModels(options: SearchOptions = {}): Promise<HubModel[]> {
  const params = new URLSearchParams();

  // Only show models that work with transformers.js
  params.set('library', 'transformers.js');

  if (options.task) params.set('pipeline_tag', options.task);
  if (options.query) params.set('search', options.query);
  if (options.author) params.set('author', options.author);
  if (options.sort === 'trending') {
    params.set('sort', 'trending');
  } else if (options.sort) {
    params.set('sort', options.sort === 'modified' ? 'lastModified' : options.sort);
    params.set('direction', options.direction === 'asc' ? '1' : '-1');
  } else {
    params.set('sort', 'downloads');
    params.set('direction', '-1');
  }

  params.set('limit', String(Math.min(options.limit ?? 20, 100)));

  const url = `${HF_API}/models?${params.toString()}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HF Hub API error: ${response.status} ${response.statusText}`);
  }

  const data: HubModelRaw[] = await response.json();
  return data.map(normalizeModel);
}

/**
 * List popular models for a specific task, sorted by downloads.
 *
 * ```ts
 * import { listModelsForTask } from 'webml-kit';
 *
 * const models = await listModelsForTask('automatic-speech-recognition');
 * // [{ modelId: 'onnx-community/whisper-tiny.en', downloads: 12345, ... }]
 * ```
 */
export async function listModelsForTask(
  task: PipelineTask | string,
  limit = 10,
): Promise<HubModel[]> {
  return searchModels({ task, sort: 'downloads', limit });
}

/**
 * List trending transformers.js models right now.
 *
 * ```ts
 * import { trendingModels } from 'webml-kit';
 *
 * const hot = await trendingModels(5);
 * ```
 */
export async function trendingModels(limit = 10): Promise<HubModel[]> {
  return searchModels({ sort: 'trending', limit });
}

/**
 * List all models from known WebGPU-friendly orgs
 * (onnx-community, Xenova, webml-community).
 *
 * ```ts
 * import { listWebGPUModels } from 'webml-kit';
 *
 * const models = await listWebGPUModels({ task: 'text-generation' });
 * ```
 */
export async function listWebGPUModels(
  options: Omit<SearchOptions, 'author'> = {},
): Promise<HubModel[]> {
  const results: HubModel[] = [];

  for (const org of WEBGPU_ORGS) {
    const models = await searchModels({ ...options, author: org });
    results.push(...models);
  }

  // Deduplicate by modelId and sort by downloads
  const seen = new Set<string>();
  return results
    .filter((m) => {
      if (seen.has(m.modelId)) return false;
      seen.add(m.modelId);
      return true;
    })
    .sort((a, b) => b.downloads - a.downloads);
}

/**
 * Get detailed info about a specific model.
 *
 * ```ts
 * import { getModelInfo } from 'webml-kit';
 *
 * const info = await getModelInfo('onnx-community/Bonsai-1.7B-ONNX');
 * console.log(info.task, info.downloads, info.webgpuCompatible);
 * ```
 */
export async function getModelInfo(modelId: string): Promise<HubModel> {
  const response = await fetch(`${HF_API}/models/${modelId}`);
  if (!response.ok) {
    throw new Error(`Model not found: ${modelId} (${response.status})`);
  }

  const data: HubModelRaw = await response.json();
  return normalizeModel(data);
}

// ─── Internal ───

interface HubModelRaw {
  id?: string;
  modelId?: string;
  pipeline_tag?: string;
  author?: string;
  downloads?: number;
  likes?: number;
  lastModified?: string;
  tags?: string[];
}

function normalizeModel(raw: HubModelRaw): HubModel {
  const id = raw.modelId ?? raw.id ?? 'unknown';
  const tags = raw.tags ?? [];
  const author = raw.author ?? id.split('/')[0] ?? 'unknown';

  return {
    modelId: id,
    task: (raw.pipeline_tag ?? 'unknown') as PipelineTask | string,
    author,
    downloads: raw.downloads ?? 0,
    likes: raw.likes ?? 0,
    lastModified: raw.lastModified ?? '',
    tags,
    webgpuCompatible: tags.includes('onnx') || WEBGPU_ORGS.some((org) => id.startsWith(org)),
  };
}
