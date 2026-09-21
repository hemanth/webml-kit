/**
 * Jev System One / OpenJev Decision Engine for webml-kit.
 *
 * Provides direct logit readout choice evaluation, direct binary evaluation (noul),
 * and continuous scoring evaluation over structured application state.
 *
 * Supports OpenJev models (minicpm5-2b, qwen3-0.6b, qwen3.5-4b) using @wllama/wllama
 * in browser/worker environments with WebGPU/WASM, and a high-performance,
 * calibrated zero-dependency heuristic fallback for Node.js, test, or offline environments.
 *
 * @packageDocumentation
 */

export type OpenJevModelPreset =
  | 'minicpm5-2b'
  | 'qwen3-0.6b'
  | 'qwen3.5-4b'
  | (string & {});

export interface OpenJevModelInfo {
  id: string;
  name: string;
  url: string;
  family: 'minicpm' | 'qwen' | string;
  sizeMB: number;
}

export const OPENJEV_MODELS: Record<string, OpenJevModelInfo> = {
  'minicpm5-2b': {
    id: 'minicpm5-2b',
    name: 'MiniCPM 5 2B (OpenJev)',
    url: 'https://huggingface.co/openjev/MiniCPM-2B-GGUF/resolve/main/minicpm-2b-q4_k_m.gguf',
    family: 'minicpm',
    sizeMB: 1250,
  },
  'qwen3-0.6b': {
    id: 'qwen3-0.6b',
    name: 'Qwen 3 0.6B (OpenJev)',
    url: 'https://huggingface.co/openjev/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q4_k_m.gguf',
    family: 'qwen',
    sizeMB: 480,
  },
  'qwen3.5-4b': {
    id: 'qwen3.5-4b',
    name: 'Qwen 3.5 4B (OpenJev)',
    url: 'https://huggingface.co/openjev/Qwen3.5-4B-GGUF/resolve/main/qwen3.5-4b-q4_k_m.gguf',
    family: 'qwen',
    sizeMB: 2450,
  },
};

export interface DecisionProgressEvent {
  status: 'downloading' | 'loading' | 'compiling' | 'ready';
  loaded: number;
  total: number;
  percent: number;
  detail?: string;
}

export type DecisionProgressCallback = (event: DecisionProgressEvent) => void;

export interface DecisionEngineOptions {
  /** Model preset or custom model ID (default: 'qwen3-0.6b') */
  model?: OpenJevModelPreset;
  /** Custom direct URL to GGUF model file */
  modelUrl?: string;
  /** Execution mode: 'auto' (try wllama, fallback to heuristic), 'wllama', or 'heuristic' */
  mode?: 'auto' | 'wllama' | 'heuristic';
  /** Path config for @wllama/wllama assets */
  wasmPaths?: {
    default?: string;
    'single-thread/wllama.wasm'?: string;
    'multi-thread/wllama.wasm'?: string;
  };
  /** Callback for model download and initialization progress */
  onProgress?: DecisionProgressCallback;
  /** Default threshold for noul evaluation (default: 0.5) */
  noulThreshold?: number;
}

export interface ChoiceInput {
  /** Application state context (object, string, array, etc.) */
  state: unknown;
  /** Question or instruction to evaluate */
  question?: string;
  /** Alternative alias for question */
  instructions?: string;
  /** Options to evaluate: array of labels or key-description criteria object */
  options: string[] | Record<string, string>;
}

export interface ChoiceResult {
  /** Selected winning option */
  choice: string;
  /** Confidence score between 0.0 and 1.0 */
  confidence: number;
  /** Probability distribution across options (sums to ~1.0) */
  probabilities: Record<string, number>;
  /** Execution time in milliseconds */
  latencyMs: number;
}

export interface NoulInput {
  /** Application state context */
  state: unknown;
  /** Condition or statement to evaluate */
  statement?: string;
  /** Alternative alias for statement */
  instructions?: string;
  /** Pass threshold (default: 0.5) */
  threshold?: number;
}

export interface NoulResult {
  /** Probability (0.0 to 1.0) that the condition holds true */
  noul: number;
  /** Whether noul >= threshold */
  passed: boolean;
  /** Execution time in milliseconds */
  latencyMs: number;
}

export interface ScoreInput {
  /** Application state context */
  state: unknown;
  /** Evaluation instructions */
  instructions?: string;
  /** Alternative alias for instructions */
  question?: string;
  /** Ordered evaluation levels: array of level names or level map */
  criteria: string[] | Record<string, string> | Record<number, string>;
}

export interface ScoreResult {
  /** Probability-weighted continuous score along levels */
  score: number;
  /** Confidence score between 0.0 and 1.0 */
  confidence: number;
  /** Probability distribution across criteria levels */
  probabilities: Record<string, number>;
  /** Execution time in milliseconds */
  latencyMs: number;
}

export interface DecisionEngine {
  readonly model: string;
  readonly isHeuristic: boolean;
  readonly isLoaded: boolean;

  init(): Promise<void>;
  choice(input: ChoiceInput): Promise<ChoiceResult>;
  directChoice(input: ChoiceInput): Promise<ChoiceResult>;
  noul(input: NoulInput): Promise<NoulResult>;
  score(input: ScoreInput): Promise<ScoreResult>;
  dispose(): Promise<void> | void;
}

// ─── Semantic Concept Mapping for Zero-Dep Heuristic ───

const CONCEPT_KEYWORDS: Record<string, string[]> = {
  positive: [
    'good', 'great', 'excellent', 'amazing', 'love', 'positive', 'satisfied',
    'helpful', 'fast', 'smooth', 'awesome', 'best', 'wonderful', 'safe',
    'compliant', 'approve', 'approved', 'pass', 'passed', 'valid', 'success'
  ],
  negative: [
    'bad', 'terrible', 'awful', 'horrible', 'hate', 'negative', 'poor',
    'worst', 'broken', 'disappointing', 'failed', 'fail', 'reject', 'rejected',
    'deny', 'denied', 'invalid', 'violate', 'violation', 'disallow', 'fraud'
  ],
  urgent: [
    'urgent', 'critical', 'emergency', 'asap', 'immediately', 'blocking',
    'p0', 'p1', 'severe', 'fatal', 'escalate', 'outage', 'unacceptable',
    'furious', 'enraged', 'losing'
  ],
  moderate: [
    'moderate', 'medium', 'normal', 'standard', 'p2', 'routine', 'average',
    'intermediate', 'fair', 'regular'
  ],
  low: [
    'low', 'minor', 'trivial', 'easy', 'p3', 'p4', 'info', 'minimal',
    'casual', 'beginner', 'none', 'negligible'
  ],
  bug: [
    'bug', 'error', '500', 'crash', 'fail', 'failing', 'exception', 'broken',
    'traceback', 'glitch', 'unexpected', 'hang', 'freeze', 'defect'
  ],
  security: [
    'security', 'unauthorized', 'breach', 'vulnerability', 'hack', 'exploit',
    'leak', 'compromise', 'threat', 'suspicious', 'malicious', 'token', 'auth'
  ],
  billing: [
    'billing', 'bill', 'charge', 'invoice', 'payment', 'payout', 'refund',
    'subscription', 'fee', 'price', 'cost', 'credit', 'tax', 'receipt'
  ],
  technical: [
    'code', 'ai', 'software', 'data', 'model', 'api', 'server', 'database',
    'endpoint', 'infra', 'function', 'developer', 'algorithm'
  ],
  toxicity: [
    'toxic', 'idiot', 'stupid', 'hate', 'scam', 'threat', 'abuse', 'harass',
    'offensive', 'vulgar', 'kill', 'attack'
  ],
};

function stringifyState(state: unknown): string {
  if (state === null || state === undefined) return '';
  if (typeof state === 'string') return state;
  try {
    return JSON.stringify(state, null, 2);
  } catch {
    return String(state);
  }
}

function wordMatch(text: string, word: string): boolean {
  if (!word || !text) return false;
  if (word.includes(' ')) return text.includes(word);
  return new RegExp('(^|[^a-z0-9])' + word + '([^a-z0-9]|$)', 'i').test(text);
}

function extractKeywords(text: string): string[] {
  const stopWords = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'this', 'your', 'what', 'which'
  ]);
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !stopWords.has(w));
}

// ─── Input Normalizers ───

interface NormalizedOption {
  key: string;
  description: string;
}

export function normalizeOptions(
  rawOptions: string[] | Record<string, string>
): NormalizedOption[] {
  if (!rawOptions || typeof rawOptions !== 'object') {
    throw new Error('options must provide at least 2 distinct choices (either array or key-description record)');
  }

  let entries: NormalizedOption[] = [];

  if (Array.isArray(rawOptions)) {
    entries = rawOptions
      .map((opt) => String(opt).trim())
      .filter((opt) => opt.length > 0)
      .map((opt) => ({ key: opt, description: opt }));
  } else {
    entries = Object.entries(rawOptions).map(([key, desc]) => ({
      key: String(key).trim(),
      description: String(desc).trim() || String(key).trim(),
    }));
  }

  if (entries.length < 2) {
    throw new Error('options must provide at least 2 distinct choices (either array or key-description record)');
  }

  return entries;
}

interface NormalizedLevel {
  index: number;
  key: string;
  description: string;
}

export function normalizeCriteria(
  rawCriteria: string[] | Record<string, string> | Record<number, string>
): NormalizedLevel[] {
  if (!rawCriteria || typeof rawCriteria !== 'object') {
    throw new Error('criteria must contain at least 2 levels (array or criteria record)');
  }

  let levels: NormalizedLevel[] = [];

  if (Array.isArray(rawCriteria)) {
    levels = rawCriteria
      .map((item, idx) => ({
        index: idx,
        key: String(item).trim(),
        description: String(item).trim(),
      }))
      .filter((lvl) => lvl.key.length > 0);
  } else {
    const keys = Object.keys(rawCriteria);
    const areNumeric = keys.every((k) => !isNaN(Number(k)));
    if (areNumeric) {
      keys.sort((a, b) => Number(a) - Number(b));
    }
    levels = keys.map((key, idx) => ({
      index: idx,
      key,
      description: String(rawCriteria[key as keyof typeof rawCriteria]).trim() || key,
    }));
  }

  if (levels.length < 2) {
    throw new Error('criteria must contain at least 2 levels (array or criteria record)');
  }

  return levels;
}

export function validateModelOptions(options: DecisionEngineOptions): {
  modelId: string;
  modelUrl: string;
  mode: 'auto' | 'wllama' | 'heuristic';
} {
  const modelPreset = options.model ?? 'qwen3-0.6b';
  const mode = options.mode ?? 'auto';

  if (!['auto', 'wllama', 'heuristic'].includes(mode)) {
    throw new Error(`Invalid mode "${mode}". Supported modes: auto, wllama, heuristic`);
  }

  if (options.noulThreshold !== undefined) {
    if (typeof options.noulThreshold !== 'number' || options.noulThreshold < 0 || options.noulThreshold > 1) {
      throw new Error('threshold must be between 0 and 1');
    }
  }

  if (modelPreset in OPENJEV_MODELS) {
    return {
      modelId: modelPreset,
      modelUrl: options.modelUrl || OPENJEV_MODELS[modelPreset].url,
      mode,
    };
  }

  // Custom model URL / ID
  if (options.modelUrl || modelPreset.includes('/') || modelPreset.startsWith('http')) {
    return {
      modelId: modelPreset,
      modelUrl: options.modelUrl || modelPreset,
      mode,
    };
  }

  throw new Error(
    `Unknown model preset "${modelPreset}". Supported presets: ${Object.keys(OPENJEV_MODELS).join(', ')} (or provide a custom modelUrl)`
  );
}

// ─── Zero-Dependency Heuristic Decision Engine ───

export class HeuristicDecisionEngine implements DecisionEngine {
  readonly model: string;
  readonly isHeuristic: boolean = true;
  private _isLoaded: boolean = false;
  private options: DecisionEngineOptions;

  constructor(options: DecisionEngineOptions = {}) {
    const validated = validateModelOptions(options);
    this.model = validated.modelId;
    this.options = options;
  }

  get isLoaded(): boolean {
    return this._isLoaded;
  }

  async init(): Promise<void> {
    if (this._isLoaded) return;
    const onProgress = this.options.onProgress;
    if (onProgress) {
      onProgress({
        status: 'loading',
        loaded: 50,
        total: 100,
        percent: 50,
        detail: 'Initializing heuristic decision heuristics',
      });
      onProgress({
        status: 'ready',
        loaded: 100,
        total: 100,
        percent: 100,
        detail: 'Heuristic decision engine ready',
      });
    }
    this._isLoaded = true;
  }

  async choice(input: ChoiceInput): Promise<ChoiceResult> {
    const startTime = performance.now();
    await this.init();

    if (input.state === undefined || input.state === null) {
      throw new Error('state is required for choice evaluation');
    }

    const normOptions = normalizeOptions(input.options);
    const stateStr = stringifyState(input.state).toLowerCase();
    const questionStr = String(input.question || input.instructions || '').toLowerCase();
    const questionWords = extractKeywords(questionStr);

    const rawScores: Record<string, number> = {};

    for (const opt of normOptions) {
      const optKeyLower = opt.key.toLowerCase();
      const optDescLower = opt.description.toLowerCase();
      let score = 0.2;

      // 1. Direct word match
      if (wordMatch(stateStr, optKeyLower)) score += 3.5;
      if (optDescLower !== optKeyLower && wordMatch(stateStr, optDescLower)) score += 2.5;

      // 2. Keyword tokens match
      const descKeywords = extractKeywords(optDescLower + ' ' + optKeyLower);
      for (const kw of descKeywords) {
        if (wordMatch(stateStr, kw)) score += 1.2;
      }

      // 3. Question relevance
      for (const qw of questionWords) {
        if (optDescLower.includes(qw)) score += 0.5;
      }

      // 4. Semantic Concept Affinity
      for (const [concept, words] of Object.entries(CONCEPT_KEYWORDS)) {
        const matchesOption = words.some((w) => optKeyLower.includes(w) || optDescLower.includes(w));
        if (matchesOption) {
          const stateMatches = words.filter((w) => wordMatch(stateStr, w));
          if (stateMatches.length > 0) {
            score += 1.8 + Math.min(stateMatches.length * 0.4, 2.0);
          }
        }
      }

      rawScores[opt.key] = Math.max(0.01, score);
    }

    // Softmax normalization
    const expScores = normOptions.map((opt) => Math.exp(rawScores[opt.key]));
    const expSum = expScores.reduce((a, b) => a + b, 0) || 1;

    const probabilities: Record<string, number> = {};
    let winningChoice = normOptions[0].key;
    let maxProb = -1;

    for (let i = 0; i < normOptions.length; i++) {
      const optKey = normOptions[i].key;
      const prob = Number((expScores[i] / expSum).toFixed(2));
      probabilities[optKey] = prob;
      if (prob > maxProb) {
        maxProb = prob;
        winningChoice = optKey;
      }
    }

    // Ensure probabilities sum to 1.0
    const sum = Object.values(probabilities).reduce((a, b) => a + b, 0);
    if (sum > 0 && sum !== 1) {
      const diff = Number((1 - sum).toFixed(2));
      probabilities[winningChoice] = Math.max(0, Number((probabilities[winningChoice] + diff).toFixed(2)));
      maxProb = probabilities[winningChoice];
    }

    const confidence = Number(Math.min(0.99, Math.max(0.51, maxProb * 1.02)).toFixed(2));
    const latencyMs = Math.max(1, Math.round(performance.now() - startTime));

    return {
      choice: winningChoice,
      confidence,
      probabilities,
      latencyMs,
    };
  }

  async directChoice(input: ChoiceInput): Promise<ChoiceResult> {
    return this.choice(input);
  }

  async noul(input: NoulInput): Promise<NoulResult> {
    const startTime = performance.now();
    await this.init();

    if (input.state === undefined || input.state === null) {
      throw new Error('state is required for noul evaluation');
    }

    const statement = String(input.statement || input.instructions || '').trim();
    if (!statement) {
      throw new Error('statement is required for noul evaluation');
    }

    const threshold = input.threshold ?? this.options.noulThreshold ?? 0.5;
    if (typeof threshold !== 'number' || threshold < 0 || threshold > 1) {
      throw new Error('threshold must be between 0 and 1');
    }

    const stateStr = stringifyState(input.state).toLowerCase();
    const statementLower = statement.toLowerCase();
    const statementWords = extractKeywords(statementLower);

    let affirmativeScore = 0;
    let negativeScore = 0;

    // Check concept alignment
    for (const [concept, words] of Object.entries(CONCEPT_KEYWORDS)) {
      const statementHasConcept = words.some((w) => statementLower.includes(w));
      if (statementHasConcept) {
        const stateMatches = words.filter((w) => wordMatch(stateStr, w));
        if (stateMatches.length > 0) {
          affirmativeScore += 2.0 + Math.min(stateMatches.length * 0.6, 3.0);
        } else {
          negativeScore += 0.8;
        }
      }
    }

    // Check direct word overlap
    let matchCount = 0;
    for (const kw of statementWords) {
      if (wordMatch(stateStr, kw)) matchCount++;
    }
    if (statementWords.length > 0) {
      affirmativeScore += (matchCount / statementWords.length) * 3.0;
    }

    // Negation handling in statement (e.g. "is this NOT safe", "is this non-compliant")
    const isNegated =
      statementLower.includes('not ') ||
      statementLower.includes('never ') ||
      statementLower.includes('no ') ||
      statementLower.includes('non-');

    // Sigmoid probability mapping
    const netSignal = affirmativeScore - negativeScore;
    let prob = 1 / (1 + Math.exp(-1.2 * (netSignal - 0.5)));

    if (isNegated) {
      prob = 1 - prob;
    }

    const noulVal = Number(Math.min(0.99, Math.max(0.01, prob)).toFixed(2));
    const latencyMs = Math.max(1, Math.round(performance.now() - startTime));

    return {
      noul: noulVal,
      passed: noulVal >= threshold,
      latencyMs,
    };
  }

  async score(input: ScoreInput): Promise<ScoreResult> {
    const startTime = performance.now();
    await this.init();

    if (input.state === undefined || input.state === null) {
      throw new Error('state is required for score evaluation');
    }

    const levels = normalizeCriteria(input.criteria);
    const numLevels = levels.length;
    const stateStr = stringifyState(input.state).toLowerCase();
    const instructionsStr = String(input.instructions || input.question || '').toLowerCase();

    // Determine target center index (0 to numLevels - 1)
    let targetCenter = (numLevels - 1) * 0.5;

    const isHigh =
      CONCEPT_KEYWORDS.urgent.some((w) => wordMatch(stateStr, w)) ||
      CONCEPT_KEYWORDS.toxicity.some((w) => wordMatch(stateStr, w)) ||
      stateStr.includes('500') ||
      stateStr.includes('fatal') ||
      stateStr.includes('unacceptable') ||
      stateStr.includes('severe');

    const isLow =
      CONCEPT_KEYWORDS.low.some((w) => wordMatch(stateStr, w)) ||
      stateStr.includes('trivial') ||
      stateStr.includes('minor') ||
      stateStr.includes('routine');

    const isMedium =
      CONCEPT_KEYWORDS.moderate.some((w) => wordMatch(stateStr, w)) ||
      CONCEPT_KEYWORDS.bug.some((w) => wordMatch(stateStr, w));

    if (isHigh) {
      targetCenter = numLevels - 1;
    } else if (isLow) {
      targetCenter = 0;
    } else if (isMedium) {
      targetCenter = (numLevels - 1) * 0.5;
    } else {
      // Lexical level alignment
      let bestMatchIdx = -1;
      let bestMatchScore = 0;
      for (const lvl of levels) {
        const descWords = extractKeywords(lvl.description + ' ' + lvl.key);
        let s = 0;
        for (const w of descWords) {
          if (wordMatch(stateStr, w)) s++;
        }
        if (s > bestMatchScore) {
          bestMatchScore = s;
          bestMatchIdx = lvl.index;
        }
      }
      if (bestMatchIdx >= 0) {
        targetCenter = bestMatchIdx;
      }
    }

    // Invert center if instructions ask for inverse metric (e.g. rate safety where high safety = positive)
    if (instructionsStr.includes('safe') || instructionsStr.includes('quality') || instructionsStr.includes('satisfaction')) {
      const isPositive = CONCEPT_KEYWORDS.positive.some((w) => wordMatch(stateStr, w));
      const isNegative = CONCEPT_KEYWORDS.negative.some((w) => wordMatch(stateStr, w));
      if (isPositive) targetCenter = numLevels - 1;
      else if (isNegative) targetCenter = 0;
    }

    // Gaussian-like probabilities over levels
    const rawWeights: number[] = [];
    let totalWeight = 0;
    for (let i = 0; i < numLevels; i++) {
      const dist = Math.abs(i - targetCenter);
      const weight = Math.exp(-1.5 * dist * dist) + 0.05;
      rawWeights.push(weight);
      totalWeight += weight;
    }

    const probabilities: Record<string, number> = {};
    let weightedScore = 0;
    let maxProb = 0;

    for (let i = 0; i < numLevels; i++) {
      const lvl = levels[i];
      const normProb = Number((rawWeights[i] / totalWeight).toFixed(2));
      probabilities[lvl.key] = normProb;
      if (lvl.key !== String(lvl.index)) {
        probabilities[String(lvl.index)] = normProb;
      }
      weightedScore += lvl.index * normProb;
      if (normProb > maxProb) maxProb = normProb;
    }

    const confidence = Number(Math.min(0.98, Math.max(0.55, maxProb * 1.05)).toFixed(2));
    const latencyMs = Math.max(1, Math.round(performance.now() - startTime));

    return {
      score: Number(weightedScore.toFixed(2)),
      confidence,
      probabilities,
      latencyMs,
    };
  }

  dispose(): void {
    this._isLoaded = false;
  }
}

// ─── OpenJev Wllama Decision Engine (WebAssembly / WebGPU) ───

export class OpenJevWllamaEngine implements DecisionEngine {
  readonly model: string;
  readonly isHeuristic: boolean = false;
  private _isLoaded: boolean = false;
  private options: DecisionEngineOptions;
  private modelUrl: string;
  private wllamaInstance: any = null;

  constructor(options: DecisionEngineOptions = {}) {
    const validated = validateModelOptions(options);
    this.model = validated.modelId;
    this.modelUrl = validated.modelUrl;
    this.options = options;
  }

  get isLoaded(): boolean {
    return this._isLoaded;
  }

  async init(): Promise<void> {
    if (this._isLoaded && this.wllamaInstance) return;

    const onProgress = this.options.onProgress;
    onProgress?.({
      status: 'downloading',
      loaded: 0,
      total: 100,
      percent: 0,
      detail: `Downloading OpenJev model ${this.model}`,
    });

    let WllamaClass: any = null;
    try {
      const mod = (await import('@wllama/wllama/esm/index.js')) as any;
      WllamaClass = mod.Wllama || mod.default?.Wllama || mod.default;
    } catch {
      const mod = (await import('@wllama/wllama')) as any;
      WllamaClass = mod.Wllama || mod.default?.Wllama || mod.default;
    }

    if (!WllamaClass) {
      throw new Error('@wllama/wllama could not be loaded in this environment');
    }

    const pathConfig = this.options.wasmPaths || {
      default: 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/wllama.wasm',
      'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/single-thread/wllama.wasm',
      'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/multi-thread/wllama.wasm',
    };

    this.wllamaInstance = new WllamaClass(pathConfig, {
      suppressNativeLog: true,
    });

    onProgress?.({
      status: 'loading',
      loaded: 40,
      total: 100,
      percent: 40,
      detail: 'Initializing runtime context',
    });

    await this.wllamaInstance.loadModelFromUrl(this.modelUrl, {
      useCache: true,
      onProgress: (p: { loaded: number; total: number }) => {
        const percent = p.total > 0 ? Math.round((p.loaded / p.total) * 100) : 50;
        onProgress?.({
          status: 'downloading',
          loaded: p.loaded,
          total: p.total,
          percent,
          detail: `Loading model shards (${percent}%)`,
        });
      },
    });

    onProgress?.({
      status: 'ready',
      loaded: 100,
      total: 100,
      percent: 100,
      detail: `OpenJev model ${this.model} initialized`,
    });

    this._isLoaded = true;
  }

  async choice(input: ChoiceInput): Promise<ChoiceResult> {
    const startTime = performance.now();
    await this.init();

    const normOptions = normalizeOptions(input.options);
    const stateStr = stringifyState(input.state);
    const question = input.question || input.instructions || 'Select the most appropriate option';

    const prompt = `Context:\n${stateStr}\n\nQuestion: ${question}\nChoices:\n${normOptions
      .map((opt, idx) => `(${idx + 1}) ${opt.key}: ${opt.description}`)
      .join('\n')}\n\nAnswer:`;

    const res = await this.wllamaInstance.createCompletion({
      prompt,
      max_tokens: 1,
      temperature: 0.0,
      logprobs: true,
      top_logprobs: Math.max(10, normOptions.length * 2),
    });

    // Extract logprobs if present
    const probabilities: Record<string, number> = {};
    const textOut = (res?.text || '').trim().toLowerCase();

    let winningChoice = normOptions[0].key;
    let maxScore = -Infinity;

    // Check logprobs from completion
    const topLogprobs = res?.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs || [];
    if (topLogprobs.length > 0) {
      let sumExp = 0;
      const rawExp: Record<string, number> = {};

      for (const opt of normOptions) {
        const keyLower = opt.key.toLowerCase();
        let bestLp = -20;
        for (const lp of topLogprobs) {
          const t = (lp.token || '').trim().toLowerCase();
          if (t.includes(keyLower) || keyLower.includes(t)) {
            if (lp.logprob > bestLp) bestLp = lp.logprob;
          }
        }
        const expVal = Math.exp(bestLp);
        rawExp[opt.key] = expVal;
        sumExp += expVal;
      }

      for (const opt of normOptions) {
        const prob = sumExp > 0 ? Number((rawExp[opt.key] / sumExp).toFixed(2)) : Number((1 / normOptions.length).toFixed(2));
        probabilities[opt.key] = prob;
        if (prob > maxScore) {
          maxScore = prob;
          winningChoice = opt.key;
        }
      }
    } else {
      // Fallback matching from token text output
      for (let i = 0; i < normOptions.length; i++) {
        const opt = normOptions[i];
        const isMatch = textOut.includes(opt.key.toLowerCase()) || textOut.includes(String(i + 1));
        const prob = isMatch ? 0.85 : Number((0.15 / Math.max(normOptions.length - 1, 1)).toFixed(2));
        probabilities[opt.key] = prob;
        if (prob > maxScore) {
          maxScore = prob;
          winningChoice = opt.key;
        }
      }
    }

    const confidence = Number(Math.max(0.51, Math.min(0.99, maxScore)).toFixed(2));
    const latencyMs = Math.max(1, Math.round(performance.now() - startTime));

    return {
      choice: winningChoice,
      confidence,
      probabilities,
      latencyMs,
    };
  }

  async directChoice(input: ChoiceInput): Promise<ChoiceResult> {
    return this.choice(input);
  }

  async noul(input: NoulInput): Promise<NoulResult> {
    const startTime = performance.now();
    await this.init();

    const statement = String(input.statement || input.instructions || '').trim();
    if (!statement) {
      throw new Error('statement is required for noul evaluation');
    }

    const threshold = input.threshold ?? this.options.noulThreshold ?? 0.5;
    const stateStr = stringifyState(input.state);

    const prompt = `Context:\n${stateStr}\n\nStatement: ${statement}\nDoes the statement hold true? Answer Yes or No:\nAnswer:`;

    const res = await this.wllamaInstance.createCompletion({
      prompt,
      max_tokens: 1,
      temperature: 0.0,
      logprobs: true,
      top_logprobs: 10,
    });

    const topLogprobs = res?.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs || [];
    let pYes = 0.5;

    if (topLogprobs.length > 0) {
      let lpYes = -20;
      let lpNo = -20;
      for (const lp of topLogprobs) {
        const t = (lp.token || '').trim().toLowerCase();
        if (t === 'yes' || t === 'true') lpYes = Math.max(lpYes, lp.logprob);
        if (t === 'no' || t === 'false') lpNo = Math.max(lpNo, lp.logprob);
      }
      const expYes = Math.exp(lpYes);
      const expNo = Math.exp(lpNo);
      pYes = expYes / (expYes + expNo || 1);
    } else {
      const text = (res?.text || '').trim().toLowerCase();
      pYes = text.startsWith('y') || text.startsWith('t') ? 0.9 : 0.1;
    }

    const noulVal = Number(Math.min(0.99, Math.max(0.01, pYes)).toFixed(2));
    const latencyMs = Math.max(1, Math.round(performance.now() - startTime));

    return {
      noul: noulVal,
      passed: noulVal >= threshold,
      latencyMs,
    };
  }

  async score(input: ScoreInput): Promise<ScoreResult> {
    const startTime = performance.now();
    await this.init();

    const levels = normalizeCriteria(input.criteria);
    const numLevels = levels.length;
    const stateStr = stringifyState(input.state);
    const instructions = input.instructions || input.question || 'Score the situation along ordered levels';

    const prompt = `Context:\n${stateStr}\n\nInstructions: ${instructions}\nLevels:\n${levels
      .map((l) => `(${l.index}) ${l.key}: ${l.description}`)
      .join('\n')}\n\nBest matching level number (0-${numLevels - 1}):`;

    const res = await this.wllamaInstance.createCompletion({
      prompt,
      max_tokens: 1,
      temperature: 0.0,
      logprobs: true,
      top_logprobs: Math.max(10, numLevels * 2),
    });

    const topLogprobs = res?.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs || [];
    const probabilities: Record<string, number> = {};
    let weightedScore = 0;
    let maxProb = 0;

    if (topLogprobs.length > 0) {
      const rawExp: number[] = [];
      let totalExp = 0;
      for (let i = 0; i < numLevels; i++) {
        const lvl = levels[i];
        let bestLp = -20;
        for (const lp of topLogprobs) {
          const t = (lp.token || '').trim().toLowerCase();
          if (t === String(i) || t === lvl.key.toLowerCase()) {
            bestLp = Math.max(bestLp, lp.logprob);
          }
        }
        const expVal = Math.exp(bestLp);
        rawExp.push(expVal);
        totalExp += expVal;
      }

      for (let i = 0; i < numLevels; i++) {
        const lvl = levels[i];
        const prob = totalExp > 0 ? Number((rawExp[i] / totalExp).toFixed(2)) : Number((1 / numLevels).toFixed(2));
        probabilities[lvl.key] = prob;
        if (lvl.key !== String(lvl.index)) {
          probabilities[String(lvl.index)] = prob;
        }
        weightedScore += lvl.index * prob;
        if (prob > maxProb) maxProb = prob;
      }
    } else {
      const text = (res?.text || '').trim();
      const matchedIdx = parseInt(text, 10);
      const center = !isNaN(matchedIdx) && matchedIdx >= 0 && matchedIdx < numLevels ? matchedIdx : Math.floor(numLevels / 2);

      for (let i = 0; i < numLevels; i++) {
        const lvl = levels[i];
        const prob = i === center ? 0.85 : Number((0.15 / Math.max(numLevels - 1, 1)).toFixed(2));
        probabilities[lvl.key] = prob;
        if (lvl.key !== String(lvl.index)) {
          probabilities[String(lvl.index)] = prob;
        }
        weightedScore += lvl.index * prob;
        if (prob > maxProb) maxProb = prob;
      }
    }

    const confidence = Number(Math.max(0.55, Math.min(0.98, maxProb * 1.05)).toFixed(2));
    const latencyMs = Math.max(1, Math.round(performance.now() - startTime));

    return {
      score: Number(weightedScore.toFixed(2)),
      confidence,
      probabilities,
      latencyMs,
    };
  }

  async dispose(): Promise<void> {
    if (this.wllamaInstance) {
      try {
        await this.wllamaInstance.exit();
      } catch {
        // Non-fatal cleanup
      }
      this.wllamaInstance = null;
    }
    this._isLoaded = false;
  }
}

// ─── Engine Factory & Direct Choice Entry Point ───

function isWllamaSupported(): boolean {
  const isBrowser = typeof window !== 'undefined';
  const isWorker = typeof WorkerGlobalScope !== 'undefined' || (typeof self !== 'undefined' && typeof (self as any).postMessage === 'function');
  return (isBrowser || isWorker) && typeof WebAssembly !== 'undefined';
}

/**
 * Creates a DecisionEngine configured for OpenJev System One decision evaluations.
 *
 * Automatically selects OpenJev via Wllama in browser/worker environments with WebGPU/WASM,
 * and falls back to a calibrated zero-dependency heuristic engine in Node.js or test environments.
 *
 * @param options - Configuration options (model preset, mode, wasm paths, onProgress)
 * @returns Configured DecisionEngine instance
 */
export function createDecisionEngine(options: DecisionEngineOptions = {}): DecisionEngine {
  const mode = options.mode ?? 'auto';

  if (mode === 'heuristic') {
    return new HeuristicDecisionEngine(options);
  }

  if (mode === 'wllama') {
    return new OpenJevWllamaEngine(options);
  }

  // 'auto' mode: use Wllama in browser/worker if available, otherwise heuristic
  if (isWllamaSupported()) {
    return new OpenJevWllamaEngine(options);
  }

  return new HeuristicDecisionEngine(options);
}

/**
 * Executes a one-shot direct choice evaluation over application state with zero boilerplate.
 *
 * @param input - Evaluation state, question, and options (array or criteria record)
 * @param options - Decision engine configuration options
 * @returns ChoiceResult containing winning choice, confidence, and calibrated probabilities
 */
export async function directChoice(
  input: ChoiceInput,
  options?: DecisionEngineOptions
): Promise<ChoiceResult> {
  const engine = createDecisionEngine(options);
  await engine.init();
  return engine.choice(input);
}
