import { FRAME_SAMPLES, SAMPLE_RATE, SavaFeatures } from './features.js';

declare const sampleRate: number;
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  abstract process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean;
}
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;

class SavaAudioProcessor extends AudioWorkletProcessor {
  private readonly frontend = new SavaFeatures();
  private readonly pending = new Float32Array(FRAME_SAMPLES);
  private count = 0;
  private frameIndex = 0;
  private finished = false;

  constructor() {
    super();
    if (sampleRate !== SAMPLE_RATE)
      throw new Error(`SAVA requires a 16000 Hz AudioContext, got ${sampleRate} Hz`);
    this.port.onmessage = event => {
      if (event.data?.type === 'flush' && !this.finished) {
        this.finished = true;
        if (this.count) {
          const count = this.count;

          this.pending.fill(0, count);
          this.emitFrame(count, false);
        }
        for (let i = 0; i < 3; i++) {
          this.pending.fill(0);
          this.emitFrame(0, true);
        }
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }

  private emitFrame(validSamples: number, synthetic: boolean): void {
    const samples = this.pending.slice();

    const mel = this.frontend.extract(samples);

    this.port.postMessage(
      { type: 'frame', index: this.frameIndex++, samples, mel, validSamples, synthetic },
      [samples.buffer, mel.buffer],
    );
    this.count = 0;
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    if (this.finished) return false;
    const channels = inputs[0] ?? [];

    const length = channels[0]?.length ?? outputs[0]?.[0]?.length ?? 128;

    for (let i = 0; i < length; i++) {
      let sample = 0;

      for (const channel of channels) sample += channel[i] ?? 0;
      this.pending[this.count++] = channels.length ? sample / channels.length : 0;
      if (this.count === FRAME_SAMPLES) this.emitFrame(FRAME_SAMPLES, false);
    }
    return true;
  }
}

registerProcessor('sava-features', SavaAudioProcessor);
