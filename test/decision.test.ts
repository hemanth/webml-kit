import { describe, it, expect } from 'vitest';
import {
  createDecisionEngine,
  directChoice,
  OPENJEV_MODELS,
  HeuristicDecisionEngine,
  normalizeOptions,
  normalizeCriteria,
  validateModelOptions,
} from '../src/decision.js';
import { webml, inferTask } from '../src/loader.js';

describe('OpenJev Model Presets and Options Validation', () => {
  it('registers all standard OpenJev model presets', () => {
    expect(OPENJEV_MODELS['minicpm5-2b']).toBeDefined();
    expect(OPENJEV_MODELS['qwen3-0.6b']).toBeDefined();
    expect(OPENJEV_MODELS['qwen3.5-4b']).toBeDefined();

    expect(OPENJEV_MODELS['minicpm5-2b'].family).toBe('minicpm');
    expect(OPENJEV_MODELS['qwen3-0.6b'].family).toBe('qwen');
    expect(OPENJEV_MODELS['qwen3.5-4b'].family).toBe('qwen');

    expect(OPENJEV_MODELS['minicpm5-2b'].url).toContain('minicpm');
    expect(OPENJEV_MODELS['qwen3-0.6b'].sizeMB).toBeGreaterThan(0);
  });

  it('validates preset models successfully', () => {
    const res1 = validateModelOptions({ model: 'minicpm5-2b' });
    expect(res1.modelId).toBe('minicpm5-2b');
    expect(res1.modelUrl).toBe(OPENJEV_MODELS['minicpm5-2b'].url);

    const res2 = validateModelOptions({ model: 'qwen3-0.6b' });
    expect(res2.modelId).toBe('qwen3-0.6b');

    const res3 = validateModelOptions({ model: 'qwen3.5-4b' });
    expect(res3.modelId).toBe('qwen3.5-4b');
  });

  it('accepts custom model URLs or repository IDs', () => {
    const customUrl = 'https://example.com/models/my-custom-jev.gguf';
    const res1 = validateModelOptions({ modelUrl: customUrl });
    expect(res1.modelUrl).toBe(customUrl);

    const res2 = validateModelOptions({ model: 'custom-org/custom-jev' });
    expect(res2.modelId).toBe('custom-org/custom-jev');
  });

  it('throws descriptive error on invalid model preset without custom URL', () => {
    expect(() => validateModelOptions({ model: 'non-existent-preset' })).toThrowError(
      /Unknown model preset "non-existent-preset"/
    );
  });

  it('validates execution modes and thresholds', () => {
    expect(() => validateModelOptions({ mode: 'invalid-mode' as any })).toThrowError(
      /Invalid mode "invalid-mode"/
    );

    expect(() => validateModelOptions({ noulThreshold: 1.5 })).toThrowError(
      /threshold must be between 0 and 1/
    );

    expect(() => validateModelOptions({ noulThreshold: -0.1 })).toThrowError(
      /threshold must be between 0 and 1/
    );
  });
});

describe('Input Normalization', () => {
  it('normalizes string array options into key-description entries', () => {
    const normalized = normalizeOptions(['approve', 'reject', 'escalate']);
    expect(normalized).toHaveLength(3);
    expect(normalized[0]).toEqual({ key: 'approve', description: 'approve' });
    expect(normalized[1]).toEqual({ key: 'reject', description: 'reject' });
    expect(normalized[2]).toEqual({ key: 'escalate', description: 'escalate' });
  });

  it('normalizes criteria object options into key-description entries', () => {
    const criteriaObj = {
      approve: 'Transaction is safe and verified',
      reject: 'High risk of fraudulent activity',
      review: 'Requires manual verification by analyst',
    };
    const normalized = normalizeOptions(criteriaObj);
    expect(normalized).toHaveLength(3);
    expect(normalized[0]).toEqual({ key: 'approve', description: 'Transaction is safe and verified' });
    expect(normalized[1]).toEqual({ key: 'reject', description: 'High risk of fraudulent activity' });
    expect(normalized[2]).toEqual({ key: 'review', description: 'Requires manual verification by analyst' });
  });

  it('throws when fewer than 2 options are provided', () => {
    expect(() => normalizeOptions(['single-option'])).toThrowError(
      /options must provide at least 2 distinct choices/
    );
    expect(() => normalizeOptions({ onlyOne: 'Alone' })).toThrowError(
      /options must provide at least 2 distinct choices/
    );
    expect(() => normalizeOptions([] as any)).toThrowError(
      /options must provide at least 2 distinct choices/
    );
  });

  it('normalizes string array criteria into ordered level entries', () => {
    const levels = normalizeCriteria(['Low', 'Medium', 'High']);
    expect(levels).toHaveLength(3);
    expect(levels[0]).toEqual({ index: 0, key: 'Low', description: 'Low' });
    expect(levels[1]).toEqual({ index: 1, key: 'Medium', description: 'Medium' });
    expect(levels[2]).toEqual({ index: 2, key: 'High', description: 'High' });
  });

  it('normalizes numeric criteria map in numeric index order', () => {
    const levels = normalizeCriteria({
      '2': 'High urgency',
      '0': 'Low urgency',
      '1': 'Medium urgency',
    });
    expect(levels).toHaveLength(3);
    expect(levels[0]).toEqual({ index: 0, key: '0', description: 'Low urgency' });
    expect(levels[1]).toEqual({ index: 1, key: '1', description: 'Medium urgency' });
    expect(levels[2]).toEqual({ index: 2, key: '2', description: 'High urgency' });
  });

  it('normalizes non-numeric criteria map into level entries', () => {
    const levels = normalizeCriteria({
      beginner: 'Novice level',
      expert: 'Master level',
    });
    expect(levels).toHaveLength(2);
    expect(levels[0]).toEqual({ index: 0, key: 'beginner', description: 'Novice level' });
    expect(levels[1]).toEqual({ index: 1, key: 'expert', description: 'Master level' });
  });

  it('throws when fewer than 2 criteria levels are provided', () => {
    expect(() => normalizeCriteria(['one'])).toThrowError(
      /criteria must contain at least 2 levels/
    );
    expect(() => normalizeCriteria({ single: 'level' })).toThrowError(
      /criteria must contain at least 2 levels/
    );
  });
});

describe('Zero-dep Heuristic Decision Mode (Node / Fallback)', () => {
  it('instantiates heuristic decision engine in Node environment', () => {
    const engine = createDecisionEngine({ model: 'qwen3-0.6b' });
    expect(engine).toBeInstanceOf(HeuristicDecisionEngine);
    expect(engine.isHeuristic).toBe(true);
    expect(engine.model).toBe('qwen3-0.6b');
    expect(engine.isLoaded).toBe(false);
  });

  it('initializes engine and reports progress lifecycle events', async () => {
    const events: any[] = [];
    const engine = createDecisionEngine({
      model: 'minicpm5-2b',
      onProgress: (e) => events.push(e),
    });

    await engine.init();
    expect(engine.isLoaded).toBe(true);
    expect(events.length).toBeGreaterThanOrEqual(1);
    expect(events[events.length - 1].status).toBe('ready');
    expect(events[events.length - 1].percent).toBe(100);

    engine.dispose();
    expect(engine.isLoaded).toBe(false);
  });
});

describe('Direct Logit Readout Choice Evaluation (`choice` / `directChoice`)', () => {
  it('evaluates choice and returns correct format and types', async () => {
    const engine = createDecisionEngine({ model: 'qwen3-0.6b' });

    const result = await engine.choice({
      state: 'User reports: "My subscription charged twice, please refund my credit card immediately."',
      question: 'Which department should handle this request?',
      options: ['billing', 'technical-support', 'sales'],
    });

    expect(result).toHaveProperty('choice');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('probabilities');
    expect(result).toHaveProperty('latencyMs');

    expect(typeof result.choice).toBe('string');
    expect(typeof result.confidence).toBe('number');
    expect(typeof result.latencyMs).toBe('number');
    expect(typeof result.probabilities).toBe('object');

    expect(['billing', 'technical-support', 'sales']).toContain(result.choice);
    expect(result.choice).toBe('billing');
    expect(result.confidence).toBeGreaterThan(0.5);
    expect(result.confidence).toBeLessThanOrEqual(1.0);
    expect(result.latencyMs).toBeGreaterThan(0);

    // Check probability distribution
    expect(result.probabilities['billing']).toBeGreaterThan(result.probabilities['sales']);
    const sum = Object.values(result.probabilities).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 1);
  });

  it('evaluates choice with criteria record object', async () => {
    const engine = createDecisionEngine();

    const result = await engine.choice({
      state: {
        error: 'Database connection failed with 500 internal server error',
        stackTrace: 'Error: ECONNREFUSED at db.connect()',
      },
      question: 'Categorize this issue',
      options: {
        bug: 'Software bug or infrastructure outage',
        feature_request: 'User asking for new feature',
        inquiry: 'General question about product',
      },
    });

    expect(result.choice).toBe('bug');
    expect(result.confidence).toBeGreaterThan(0.6);
    expect(result.probabilities['bug']).toBeGreaterThan(result.probabilities['feature_request']);
  });

  it('evaluates choice via engine.directChoice and standalone directChoice', async () => {
    const engine = createDecisionEngine();

    const resEngine = await engine.directChoice({
      state: 'Great job! This was an amazing and wonderful experience.',
      options: ['positive', 'negative', 'neutral'],
    });
    expect(resEngine.choice).toBe('positive');

    const resStandalone = await directChoice({
      state: 'Terrible service. Awful quality and broken product.',
      options: ['positive', 'negative', 'neutral'],
    });
    expect(resStandalone.choice).toBe('negative');
  });

  it('throws when state is missing or options are invalid', async () => {
    const engine = createDecisionEngine();
    await expect(engine.choice({ state: null as any, options: ['a', 'b'] })).rejects.toThrowError(
      /state is required/
    );
    await expect(engine.choice({ state: 'ok', options: ['single'] })).rejects.toThrowError(
      /options must provide at least 2 distinct choices/
    );
  });
});

describe('Direct Binary Evaluation (`noul`)', () => {
  it('evaluates binary condition and returns correct format and types', async () => {
    const engine = createDecisionEngine();

    const result = await engine.noul({
      state: 'Critical alert: Production cluster database outage, customer payments are failing ASAP!',
      statement: 'Is this an urgent or critical issue?',
    });

    expect(result).toHaveProperty('noul');
    expect(result).toHaveProperty('passed');
    expect(result).toHaveProperty('latencyMs');

    expect(typeof result.noul).toBe('number');
    expect(typeof result.passed).toBe('boolean');
    expect(typeof result.latencyMs).toBe('number');

    expect(result.noul).toBeGreaterThan(0.6);
    expect(result.passed).toBe(true);
  });

  it('correctly reports passed: false for condition not met', async () => {
    const engine = createDecisionEngine();

    const result = await engine.noul({
      state: 'Everything is running smoothly on routine schedule with no issues.',
      statement: 'Is there a severe emergency or critical bug?',
    });

    expect(result.noul).toBeLessThan(0.4);
    expect(result.passed).toBe(false);
  });

  it('honors custom threshold parameter and engine default', async () => {
    const engine = createDecisionEngine();
    const state = 'There is occasional mild latency in background sync.';
    const statement = 'Is there an urgent disruption?';

    const resLow = await engine.noul({ state, statement, threshold: 0.05 });
    const resHigh = await engine.noul({ state, statement, threshold: 0.95 });

    expect(resLow.noul).toBe(resHigh.noul);
    expect(resLow.passed).toBe(true);
    expect(resHigh.passed).toBe(false);

    // Test engine-level default threshold option
    const engineHighThreshold = createDecisionEngine({ noulThreshold: 0.95 });
    const resEngineDefault = await engineHighThreshold.noul({ state, statement });
    expect(resEngineDefault.passed).toBe(false);
  });

  it('throws error when statement is missing or threshold is invalid', async () => {
    const engine = createDecisionEngine();
    await expect(engine.noul({ state: 'test', statement: '' })).rejects.toThrowError(
      /statement is required/
    );
    await expect(engine.noul({ state: 'test', statement: 'valid', threshold: -0.5 })).rejects.toThrowError(
      /threshold must be between 0 and 1/
    );
  });
});

describe('Continuous Scoring Evaluation (`score`)', () => {
  it('evaluates continuous score and returns correct format and types', async () => {
    const engine = createDecisionEngine();

    const result = await engine.score({
      state: 'System is running normally, routine background task completed.',
      instructions: 'Rate story urgency from low to critical',
      criteria: ['Low', 'Medium', 'High', 'Critical'],
    });

    expect(result).toHaveProperty('score');
    expect(result).toHaveProperty('confidence');
    expect(result).toHaveProperty('probabilities');
    expect(result).toHaveProperty('latencyMs');

    expect(typeof result.score).toBe('number');
    expect(typeof result.confidence).toBe('number');
    expect(typeof result.probabilities).toBe('object');
    expect(typeof result.latencyMs).toBe('number');

    // Expected low score near 0
    expect(result.score).toBeLessThan(1.5);
    expect(result.probabilities['Low']).toBeGreaterThan(result.probabilities['Critical']);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it('evaluates high severity situation towards highest score level', async () => {
    const engine = createDecisionEngine();

    const result = await engine.score({
      state: 'CRITICAL EMERGENCY: Massive security breach and 500 outage, customer data compromised!',
      instructions: 'Rate incident severity',
      criteria: ['P3', 'P2', 'P1', 'P0'],
    });

    // Score is 0-indexed (0 to 3), so P0 is index 3
    expect(result.score).toBeGreaterThan(2.0);
    expect(result.probabilities['P0']).toBeGreaterThan(result.probabilities['P3']);
  });

  it('supports criteria object mapping and index access', async () => {
    const engine = createDecisionEngine();

    const result = await engine.score({
      state: 'The service is experiencing a moderate slowdown with occasional latency spikes.',
      instructions: 'Rate issue impact',
      criteria: {
        '0': 'Negligible impact',
        '1': 'Moderate disruption',
        '2': 'Total outage',
      },
    });

    expect(result.probabilities['1']).toBeDefined();
    expect(result.probabilities['0']).toBeDefined();
    expect(result.probabilities['2']).toBeDefined();
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(2);
  });

  it('throws error when state is missing or criteria has fewer than 2 levels', async () => {
    const engine = createDecisionEngine();
    await expect(engine.score({ state: null as any, criteria: ['low', 'high'] })).rejects.toThrowError(
      /state is required/
    );
    await expect(engine.score({ state: 'test', criteria: ['single-level'] })).rejects.toThrowError(
      /criteria must contain at least 2 levels/
    );
  });
});

describe('WebML Convenience API Integration', () => {
  it('exposes webml.decision and webml.directChoice methods', async () => {
    expect(typeof webml.decision).toBe('function');
    expect(typeof webml.directChoice).toBe('function');

    const engine = webml.decision({ model: 'qwen3-0.6b' });
    expect(engine).toBeInstanceOf(HeuristicDecisionEngine);

    const direct = await webml.directChoice({
      state: 'Customer sent positive review: "Excellent product!"',
      options: ['positive', 'negative'],
    });
    expect(direct.choice).toBe('positive');
  });

  it('auto-infers decision task from model names and prefixes', async () => {
    expect(await inferTask('minicpm5-2b')).toBe('decision');
    expect(await inferTask('qwen3-0.6b')).toBe('decision');
    expect(await inferTask('qwen3.5-4b')).toBe('decision');
    expect(await inferTask('openjev/MiniCPM-2B-GGUF')).toBe('decision');
    expect(await inferTask('typesafe-ai/jev-decision')).toBe('decision');
  });

  it('loads decision model via webml callable wrapper', async () => {
    const model = await webml('qwen3-0.6b');
    expect(model.task).toBe('decision');
    expect(model.modelId).toBe('qwen3-0.6b');

    // Callable directly
    const choiceRes = (await model({
      state: 'Customer needs refund for double charge on invoice',
      options: ['billing', 'tech', 'general'],
    })) as any;
    expect(choiceRes.choice).toBe('billing');

    // Method calls
    expect(typeof model.choice).toBe('function');
    expect(typeof model.noul).toBe('function');
    expect(typeof model.score).toBe('function');

    const noulRes = await model.noul!({
      state: 'System is operating at 100% health',
      statement: 'Is there a defect or outage?',
    });
    expect(noulRes.passed).toBe(false);

    model.dispose();
  });
});
