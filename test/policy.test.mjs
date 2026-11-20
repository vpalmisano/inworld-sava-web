import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_POLICY, FRAME_MS, TurnSegmenter, segment } from '../dist/index.js';

const batch = (first, speech, turn = speech.map(() => 0)) => ({ first, speech, turn });

const turn = (start, end, speechMs, reason, kept) => ({
  start,
  end,
  startSeconds: start * 0.04,
  endSeconds: end * 0.04,
  speechMs,
  reason,
  kept,
});

test('reference defaults, inclusive thresholds, and both keep gates', () => {
  assert.equal(FRAME_MS, 40);
  assert.deepEqual(DEFAULT_POLICY, {
    speechThreshold: 0.5,
    turnThreshold: 0.9,
    preRollMs: 800,
    minSilenceBeforeCheckMs: 0,
    maxSilenceMs: 1600,
    minSpeechMs: 400,
    minConfidentSpeechMs: 200,
    confidentSpeechThreshold: 0.9,
  });
  const s = new TurnSegmenter(0);

  assert.deepEqual(s.feed(batch(0, [...Array(10).fill(0.5), 0.49], [...Array(10).fill(1), 0.9])), [
    turn(0, 11, 400, 'turn', true),
  ]);
  assert.deepEqual(s.feed(batch(11, [...Array(5).fill(0.9), 0], [0, 0, 0, 0, 0, 0.9])), [
    turn(0, 17, 200, 'turn', true),
  ]);
  assert.deepEqual(s.feed(batch(17, [0.89, 0], [0, 0.9])), [turn(0, 19, 40, 'turn', false)]);
  assert.deepEqual(s.finish(), []);
});

test('delayed verdicts use the emitted turn score and close at emission time', () => {
  const s = new TurnSegmenter(2, { preRollMs: 0, minSpeechMs: 40 });

  assert.deepEqual(s.feed(batch(0, [1, 1, 0.5], [1, 1, 0])), []);
  assert.equal(s.open, true);
  assert.deepEqual(s.feed(batch(3, [0, 0.6, 0], [0.9, 1, 0.9])), [
    turn(0, 4, 40, 'turn', true),
    turn(2, 6, 40, 'turn', true),
  ]);
  assert.equal(s.open, false);
});

test('silence timer resets on speech and forced silence wins over turn', () => {
  const s = new TurnSegmenter(0, {
    preRollMs: 0,
    minSpeechMs: 80,
    minSilenceBeforeCheckMs: 80,
    maxSilenceMs: 120,
  });

  assert.deepEqual(s.feed(batch(0, [1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1])), [
    turn(0, 6, 80, 'silence', true),
  ]);
  const check = new TurnSegmenter(0, { preRollMs: 0, minSilenceBeforeCheckMs: 80 });

  assert.deepEqual(check.feed(batch(0, [1, 0, 0], [0, 1, 0.9])), [turn(0, 3, 40, 'turn', false)]);
});

test('default maximum silence closes at exactly 1600 ms without a turn signal', () => {
  const s = new TurnSegmenter(0, { preRollMs: 0 });

  assert.deepEqual(s.feed(batch(0, [1, ...Array(39).fill(0)])), []);
  assert.equal(s.open, true);
  assert.deepEqual(s.feed(batch(40, [0])), [turn(0, 41, 40, 'silence', false)]);
});

test('finish closes at the last judged frame and is idempotent', () => {
  const s = new TurnSegmenter(3, { preRollMs: 0 });

  assert.deepEqual(s.feed(batch(0, [0, 0, 0, 1, 0], [0, 0, 0, 0, 0])), []);
  assert.equal(s.open, true);
  assert.deepEqual(s.finish(), [turn(0, 2, 40, 'end', false)]);
  assert.equal(s.open, false);
  assert.deepEqual(s.finish(), []);
});

test('pre-roll rounds ties to even and clamps at stream start', () => {
  for (const [ms, expected] of [
    [20, 5],
    [60, 3],
    [100, 3],
    [140, 1],
    [800, 0],
  ]) {
    const s = new TurnSegmenter(0, { preRollMs: ms });

    s.feed(batch(0, [0, 0, 0, 0, 0, 1]));
    assert.deepEqual(s.finish(), [turn(expected, 6, 40, 'end', false)]);
  }
});

test('chunking and flush preserve the whole-recording result', () => {
  const speech = [1, 0, 1, 0, 1];

  const onTime = [0, 0, 0, 0.9, 0];

  const flush = [0.9, 0];

  const policy = { preRollMs: 0, minSpeechMs: 40 };

  const expected = [
    turn(0, 4, 40, 'turn', true),
    turn(2, 6, 40, 'turn', true),
    turn(4, 5, 40, 'end', true),
  ];

  assert.deepEqual(segment({ speech, turn: onTime, flushTurn: flush }, 2, policy), expected);
  const s = new TurnSegmenter(2, policy);

  const emitted = [0, 0, ...speech];

  const steps = [...onTime, ...flush];

  const result = [
    ...s.feed(batch(0, emitted.slice(0, 1), steps.slice(0, 1))),
    ...s.feed(batch(1, emitted.slice(1, 5), steps.slice(1, 5))),
    ...s.feed(batch(5, emitted.slice(5), steps.slice(5))),
    ...s.finish(),
  ];

  assert.deepEqual(result, expected);
});

test('empty recordings, typed arrays and malformed batch lengths', () => {
  assert.deepEqual(segment({ speech: [], turn: [], flushTurn: [0, 0] }, 2), []);
  const s = new TurnSegmenter(0, { preRollMs: 0 });

  assert.deepEqual(
    s.feed({ first: 0, speech: new Float32Array([1]), turn: new Float32Array([0]) }),
    [],
  );
  assert.deepEqual(s.finish(), [turn(0, 1, 40, 'end', false)]);
  assert.throws(() => new TurnSegmenter(-1), RangeError);
  assert.throws(() => s.feed({ first: 0, speech: [1], turn: [] }), RangeError);
  assert.throws(() => segment({ speech: [1], turn: [0], flushTurn: [] }, 1), RangeError);
});
