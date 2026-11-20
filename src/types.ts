export type Backend = 'auto' | 'wasm' | 'webgpu';

export interface AudioFrame {
  type: 'frame';
  index: number;
  samples: Float32Array;
  mel: Float32Array;
  validSamples: number;
  synthetic: boolean;
}

export interface FrameScores {
  index: number;
  /** Emitted at index; judges audio at index - 3. */
  speech: number;
  /** The on-time speech probability used internally by the turn head. */
  speechOnTime: number;
  /** End-of-turn probability for the current frame (horizon zero). */
  turn: number;
  synthetic: boolean;
  validSamples: number;
}

export type WorkerRequest =
  | {
      type: 'load';
      modelUrl: string;
      manifestUrl: string;
      backend: Backend;
      wasmPaths?: string;
      numThreads?: number;
    }
  | AudioFrame
  | { type: 'flush' };

export type WorkerResponse =
  | { type: 'ready'; backend: Exclude<Backend, 'auto'> }
  | { type: 'frame'; scores: FrameScores }
  | { type: 'flushed' }
  | { type: 'error'; message: string };
