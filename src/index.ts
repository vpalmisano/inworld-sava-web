import { TurnSegmenter, type Policy, type Turn } from './policy.js';
import type { AudioFrame, Backend, FrameScores, WorkerRequest, WorkerResponse } from './types.js';

export { DEFAULT_POLICY, FRAME_MS, FRAME_SECONDS, TurnSegmenter, segment } from './policy.js';
export type { Curves, Frames, Policy, Turn, TurnReason } from './policy.js';
export type { Backend, FrameScores } from './types.js';

export interface SavaOptions {
  modelUrl: string;
  manifestUrl: string;
  backend?: Backend;
  /** Override when hosting the worklet outside the bundled package. */
  workletUrl?: string | URL;
  /** Override when hosting the inference worker outside the bundled package. */
  workerUrl?: string | URL;
  wasmPaths?: string;
  numThreads?: number;
  maxPendingFrames?: number;
  policy?: Partial<Policy>;
  onFrame?: (frame: FrameScores) => void;
  onSpeechStart?: (judgedFrame: number) => void;
  onTurn?: (turn: Turn) => void;
  onError?: (error: Error) => void;
}

type State = 'idle' | 'starting' | 'running' | 'stopping' | 'closed';

function isLive(track: MediaStreamTrack): boolean {
  return track.readyState === 'live';
}

function packagedUrl(name: string): URL {
  return new URL(name, import.meta.url);
}

export class SavaDetector {
  private state: State = 'idle';
  private worker: Worker | null = null;
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private mute: GainNode | null = null;
  private track: MediaStreamTrack | null = null;
  private segmenter: TurnSegmenter;
  private pending = 0;
  private resolveReady: (() => void) | null = null;
  private rejectReady: ((error: Error) => void) | null = null;
  private resolveStop: (() => void) | null = null;
  private rejectStop: ((error: Error) => void) | null = null;
  private stopping: Promise<void> | null = null;
  private lastError: Error | null = null;
  private readonly maxPending: number;
  private readonly ended = () => {
    void this.stop().catch(() => {});
  };
  backend: 'wasm' | 'webgpu' | null = null;

  constructor(private readonly options: SavaOptions) {
    this.maxPending = options.maxPendingFrames ?? 50;
    if (!Number.isSafeInteger(this.maxPending) || this.maxPending < 1)
      throw new RangeError('maxPendingFrames must be a positive integer');
    this.segmenter = new TurnSegmenter(3, options.policy);
  }

  async start(track: MediaStreamTrack): Promise<void> {
    if (this.state !== 'idle') throw new Error('SavaDetector instances can only be started once');
    if (track.kind !== 'audio' || !isLive(track)) throw new Error('A live audio track is required');
    this.state = 'starting';
    try {
      this.worker = this.options.workerUrl
        ? new Worker(this.options.workerUrl, { type: 'module' })
        : new Worker(packagedUrl('./inference-worker.js'), { type: 'module' });
      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) =>
        this.handleWorker(event.data);
      this.worker.onerror = event => this.fail(new Error(event.message || 'SAVA worker failed'));
      const ready = new Promise<void>((resolve, reject) => {
        this.resolveReady = resolve;
        this.rejectReady = reject;
      });

      this.post({
        type: 'load',
        modelUrl: this.options.modelUrl,
        manifestUrl: this.options.manifestUrl,
        backend: this.options.backend ?? 'auto',
        wasmPaths: this.options.wasmPaths,
        numThreads: this.options.numThreads,
      });
      await ready;
      if (this.state !== 'starting') throw this.lastError ?? new Error('SAVA startup cancelled');
      const context = new AudioContext({ sampleRate: 16000, latencyHint: 'interactive' });

      this.context = context;
      if (context.sampleRate !== 16000)
        throw new Error(`SAVA requires a 16000 Hz AudioContext; got ${context.sampleRate}`);
      await context.audioWorklet.addModule(
        this.options.workletUrl ?? packagedUrl('./audio-worklet.js'),
      );
      if (this.state !== 'starting') throw this.lastError ?? new Error('SAVA startup cancelled');
      if (context.state === 'suspended') await context.resume();
      if (this.state !== 'starting') throw this.lastError ?? new Error('SAVA startup cancelled');
      if (!isLive(track)) throw new Error('Audio track ended during startup');
      this.track = track;
      track.addEventListener('ended', this.ended);
      this.node = new AudioWorkletNode(context, 'sava-features', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.node.onprocessorerror = () => this.fail(new Error('SAVA audio worklet failed'));
      this.node.port.onmessage = event => this.handleAudio(event.data);
      this.source = context.createMediaStreamSource(new MediaStream([track]));
      this.mute = context.createGain();
      this.mute.gain.value = 0;
      this.source.connect(this.node);
      this.node.connect(this.mute);
      this.mute.connect(context.destination);
      this.state = 'running';
    } catch (error) {
      this.cleanup();
      throw error;
    }
  }

  /** Flushes the partial frame and lookahead, waits for inference, then releases resources. */
  stop(): Promise<void> {
    if (this.stopping) return this.stopping;
    if (this.state === 'idle' || this.state === 'closed') return Promise.resolve();
    if (this.state === 'starting') {
      const error = new Error('SAVA startup cancelled');

      this.lastError = error;
      this.rejectReady?.(error);
      this.cleanup();
      return Promise.resolve();
    }
    this.state = 'stopping';
    this.stopping = new Promise<void>((resolve, reject) => {
      this.resolveStop = resolve;
      this.rejectStop = reject;
    });
    this.node?.port.postMessage({ type: 'flush' });
    return this.stopping;
  }

  private post(message: WorkerRequest, transfer: Transferable[] = []): void {
    this.worker?.postMessage(message, transfer);
  }

  private handleAudio(message: AudioFrame | { type: 'flushed' }): void {
    if (this.state !== 'running' && this.state !== 'stopping') return;
    if (message.type === 'flushed') {
      this.post({ type: 'flush' });
      return;
    }
    if (++this.pending > this.maxPending) {
      this.fail(new Error('SAVA inference is slow'));
      return;
    }
    this.post(message, [message.samples.buffer, message.mel.buffer]);
  }

  private handleWorker(message: WorkerResponse): void {
    if (message.type === 'error') {
      this.fail(new Error(message.message));
      return;
    }
    if (message.type === 'ready') {
      this.backend = message.backend;
      this.resolveReady?.();
      this.resolveReady = null;
      this.rejectReady = null;
      return;
    }
    if (message.type === 'frame') {
      this.pending--;
      const score = message.scores;

      this.options.onFrame?.(score);
      const wasOpen = this.segmenter.open;

      const turns = this.segmenter.feed({
        first: score.index,
        speech: [score.speech],
        turn: [score.turn],
      });

      if (!wasOpen && this.segmenter.open) this.options.onSpeechStart?.(score.index - 3);
      for (const turn of turns) this.options.onTurn?.(turn);
      return;
    }
    if (message.type === 'flushed') {
      for (const turn of this.segmenter.finish()) this.options.onTurn?.(turn);
      const resolve = this.resolveStop;

      this.cleanup();
      resolve?.();
    }
  }

  private fail(error: Error): void {
    if (this.state === 'closed') return;
    this.lastError = error;
    const rejectReady = this.rejectReady;

    const rejectStop = this.rejectStop;

    const starting = this.state === 'starting';

    this.cleanup();
    rejectReady?.(error);
    rejectStop?.(error);
    if (!starting) this.options.onError?.(error);
  }

  private cleanup(): void {
    this.state = 'closed';
    this.track?.removeEventListener('ended', this.ended);
    this.track = null;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    this.source?.disconnect();
    this.source = null;
    this.mute?.disconnect();
    this.mute = null;
    if (this.context) {
      void this.context.close().catch(() => {});
      this.context = null;
    }
    if (this.worker) {
      this.worker.onmessage = null;
      this.worker.onerror = null;
      this.worker.terminate();
      this.worker = null;
    }
    this.resolveReady = null;
    this.rejectReady = null;
    this.resolveStop = null;
    this.rejectStop = null;
  }
}
