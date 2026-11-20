// Adapted from Inworld AI's SAVA segmenter.py (Copyright 2026 Inworld AI),
// licensed under the Apache License, Version 2.0:
// https://www.apache.org/licenses/LICENSE-2.0
// https://github.com/inworld-ai/sava/blob/main/src/inworld/sava/segmenter.py

/** SAVA audio frames are 640 samples at 16 kHz. */
export const FRAME_MS = 40;

export const FRAME_SECONDS = FRAME_MS / 1000;

export interface Policy {
  speechThreshold: number;
  turnThreshold: number;
  preRollMs: number;
  minSilenceBeforeCheckMs: number;
  maxSilenceMs: number;
  minSpeechMs: number;
  minConfidentSpeechMs: number;
  confidentSpeechThreshold: number;
}

export const DEFAULT_POLICY: Readonly<Policy> = Object.freeze({
  speechThreshold: 0.5,
  turnThreshold: 0.9,
  preRollMs: 800,
  minSilenceBeforeCheckMs: 0,
  maxSilenceMs: 1600,
  minSpeechMs: 400,
  minConfidentSpeechMs: 200,
  confidentSpeechThreshold: 0.9,
});

export type TurnReason = 'turn' | 'silence' | 'end';

/** Frame span [start, end); dropped turns are returned with kept=false. */
export interface Turn {
  start: number;
  end: number;
  startSeconds: number;
  endSeconds: number;
  speechMs: number;
  reason: TurnReason;
  kept: boolean;
}

/** Scores in emission order; speech[i] judges first+i-lag, turn[i] first+i. */
export interface Frames {
  first: number;
  speech: ArrayLike<number>;
  turn: ArrayLike<number>;
}

/** Whole-recording curves: aligned speech and on-time horizon-zero turn scores. */
export interface Curves {
  speech: ArrayLike<number>;
  turn: ArrayLike<number>;
  flushTurn: ArrayLike<number>;
}

// Python round() uses ties-to-even, unlike Math.round().
function frames(ms: number): number {
  const value = ms / FRAME_MS;

  const lower = Math.floor(value);

  return value - lower === 0.5 ? lower + (lower % 2 === 0 ? 0 : 1) : Math.round(value);
}

/** Consumes consecutive model output batches, including the model's flush. */
export class TurnSegmenter {
  readonly policy: Readonly<Policy>;
  readonly lag: number;
  private start: number | null = null;
  private speechMs = 0;
  private confidentMs = 0;
  private pauseFrames = 0;
  private lastJudged = -1;

  constructor(lag: number, policy: Partial<Policy> = {}) {
    if (!Number.isSafeInteger(lag) || lag < 0) {
      throw new RangeError('lag must be a non-negative integer');
    }
    this.lag = lag;
    this.policy = Object.freeze({ ...DEFAULT_POLICY, ...policy });
  }

  get open(): boolean {
    return this.start !== null;
  }

  feed(batch: Frames): Turn[] {
    if (!Number.isSafeInteger(batch.first) || batch.first < 0) {
      throw new RangeError('first must be a non-negative integer');
    }
    if (batch.speech.length !== batch.turn.length) {
      throw new RangeError('speech and turn must have equal lengths');
    }
    const closed: Turn[] = [];

    for (let i = 0; i < batch.speech.length; i++) {
      const now = batch.first + i;

      const judged = now - this.lag;

      if (judged < 0) continue;
      const turn = this.frame(judged, now, batch.speech[i]!, batch.turn[i]!);

      if (turn !== null) closed.push(turn);
    }
    return closed;
  }

  /** Call after feeding flush scores; closes at the last judged frame, not emission time. */
  finish(): Turn[] {
    return this.start === null ? [] : [this.close(this.lastJudged + 1, 'end')];
  }

  private frame(judged: number, now: number, speech: number, turn: number): Turn | null {
    const policy = this.policy;

    this.lastJudged = judged;
    const isSpeech = speech >= policy.speechThreshold;

    if (this.start === null) {
      if (!isSpeech) return null;
      this.start = Math.max(0, judged - frames(policy.preRollMs));
      this.speechMs = 0;
      this.confidentMs = 0;
      this.pauseFrames = 0;
    }
    if (isSpeech) {
      this.pauseFrames = 0;
      this.speechMs += FRAME_MS;
      if (speech >= policy.confidentSpeechThreshold) this.confidentMs += FRAME_MS;
      return null;
    }
    this.pauseFrames++;
    const pauseMs = this.pauseFrames * FRAME_MS;

    if (pauseMs >= policy.maxSilenceMs) return this.close(now + 1, 'silence');
    if (pauseMs >= policy.minSilenceBeforeCheckMs && turn >= policy.turnThreshold) {
      return this.close(now + 1, 'turn');
    }
    return null;
  }

  private close(end: number, reason: TurnReason): Turn {
    const start = this.start!;

    const result: Turn = {
      start,
      end,
      startSeconds: start * FRAME_SECONDS,
      endSeconds: end * FRAME_SECONDS,
      speechMs: this.speechMs,
      reason,
      kept:
        this.speechMs >= this.policy.minSpeechMs ||
        this.confidentMs >= this.policy.minConfidentSpeechMs,
    };

    this.start = null;
    return result;
  }
}

/** Segments aligned whole-recording curves using the same emitted turn scores as streaming. */
export function segment(curves: Curves, lag: number, policy: Partial<Policy> = {}): Turn[] {
  const segmenter = new TurnSegmenter(lag, policy);

  if (curves.turn.length !== curves.speech.length || curves.flushTurn.length !== lag) {
    throw new RangeError('turn must match speech length and flushTurn must match lag');
  }
  const turns = Array.from(curves.turn).concat(Array.from(curves.flushTurn));

  return segmenter
    .feed({
      first: lag,
      speech: curves.speech,
      turn: turns.slice(lag, lag + curves.speech.length),
    })
    .concat(segmenter.finish());
}
