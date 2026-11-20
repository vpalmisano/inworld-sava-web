// Implements the causal frontend described by Inworld AI's Apache-2.0 SAVA.
// https://github.com/inworld-ai/sava/blob/main/src/inworld/sava/mel.py
export const SAMPLE_RATE = 16_000;

export const FRAME_SAMPLES = 640;

export const MEL_BINS = 128;

const FFT_SIZE = 400;

const HOP = 160;

const CONTEXT = FFT_SIZE - HOP;

const FFT_BINS = FFT_SIZE / 2 + 1;

const LOG_STEP = Math.log(6.4) / 27;

function hzToMel(hz: number): number {
  return hz < 1000 ? (3 * hz) / 200 : 15 + Math.log(hz / 1000) / LOG_STEP;
}

function melToHz(mel: number): number {
  return mel < 15 ? (200 * mel) / 3 : 1000 * Math.exp(LOG_STEP * (mel - 15));
}

/** Stateful, causal fp32 SAVA log-mel frontend. Input must already be mono 16 kHz. */
export class SavaFeatures {
  private readonly context = new Float32Array(CONTEXT);
  private readonly real = new Float32Array(FFT_BINS * FFT_SIZE);
  private readonly imag = new Float32Array(FFT_BINS * FFT_SIZE);
  private readonly bank = new Float32Array(MEL_BINS * FFT_BINS);

  constructor() {
    for (let bin = 0; bin < FFT_BINS; bin++) {
      for (let i = 0; i < FFT_SIZE; i++) {
        const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT_SIZE);

        const angle = (-2 * Math.PI * bin * i) / FFT_SIZE;

        this.real[bin * FFT_SIZE + i] = Math.cos(angle) * window;
        this.imag[bin * FFT_SIZE + i] = Math.sin(angle) * window;
      }
    }
    // The upstream filterbank computes its upper mel bound in float32.
    const maxMel = Math.fround(hzToMel(8000));

    const points = Array.from({ length: MEL_BINS + 2 }, (_, i) =>
      melToHz((maxMel * i) / (MEL_BINS + 1)),
    );

    for (let m = 0; m < MEL_BINS; m++) {
      const left = points[m]!;

      const center = points[m + 1]!;

      const right = points[m + 2]!;

      const norm = 2 / (right - left);

      for (let k = 0; k < FFT_BINS; k++) {
        const hz = (k * SAMPLE_RATE) / FFT_SIZE;

        this.bank[m * FFT_BINS + k] =
          Math.max(0, Math.min((hz - left) / (center - left), (right - hz) / (right - center))) *
          norm;
      }
    }
  }

  reset(): void {
    this.context.fill(0);
  }

  /** Returns four hop-major 128-bin log-mel vectors. */
  extract(frame: Float32Array): Float32Array {
    if (frame.length !== FRAME_SAMPLES) throw new Error('Expected exactly 640 samples');
    const pcm = new Float32Array(CONTEXT + FRAME_SAMPLES);

    pcm.set(this.context);
    pcm.set(frame, CONTEXT);
    this.context.set(pcm.subarray(FRAME_SAMPLES));
    const output = new Float32Array(4 * MEL_BINS);

    const power = new Float32Array(FFT_BINS);

    for (let hop = 0; hop < 4; hop++) {
      const offset = hop * HOP;

      for (let k = 0; k < FFT_BINS; k++) {
        let re = 0;

        let im = 0;

        const basis = k * FFT_SIZE;

        for (let i = 0; i < FFT_SIZE; i++) {
          const sample = pcm[offset + i]!;

          re += sample * this.real[basis + i]!;
          im += sample * this.imag[basis + i]!;
        }
        re = Math.fround(re);
        im = Math.fround(im);
        power[k] = Math.fround(Math.fround(re * re) + Math.fround(im * im));
      }
      for (let m = 0; m < MEL_BINS; m++) {
        let sum = 0;

        const start = m * FFT_BINS;

        for (let k = 0; k < FFT_BINS; k++) sum += power[k]! * this.bank[start + k]!;
        output[hop * MEL_BINS + m] = (Math.log10(Math.max(sum, 1e-10)) + 4) / 4;
      }
    }
    return output;
  }
}
