import * as ort from 'onnxruntime-web/webgpu';
import type { AudioFrame, WorkerRequest, WorkerResponse } from './types.js';

interface StateSpec {
  input: string;
  output: string;
  shape: number[];
  dtype: 'float32' | 'int64';
}

interface Manifest {
  format: 'sava-streaming-onnx-v1';
  states: StateSpec[];
}

const scope = self as unknown as DedicatedWorkerGlobalScope;

let session: ort.InferenceSession | null = null;

let manifest: Manifest | null = null;

let states: Record<string, ort.Tensor> = {};

let queue = Promise.resolve();

let failed = false;

function send(message: WorkerResponse): void {
  scope.postMessage(message);
}

async function load(message: Extract<WorkerRequest, { type: 'load' }>): Promise<void> {
  if (session) throw new Error('SAVA worker is already loaded');
  if (
    message.numThreads !== undefined &&
    (!Number.isSafeInteger(message.numThreads) || message.numThreads < 1)
  ) {
    throw new Error('numThreads must be a positive integer');
  }
  const response = await fetch(message.manifestUrl);

  if (!response.ok) throw new Error(`Failed to fetch SAVA manifest: ${response.status}`);
  const data: unknown = await response.json();

  if (
    !data ||
    typeof data !== 'object' ||
    (data as Manifest).format !== 'sava-streaming-onnx-v1' ||
    !Array.isArray((data as Manifest).states)
  ) {
    throw new Error('Unsupported SAVA model manifest');
  }
  manifest = data as Manifest;
  for (const state of manifest.states) {
    if (
      typeof state.input !== 'string' ||
      typeof state.output !== 'string' ||
      !Array.isArray(state.shape) ||
      !state.shape.every(size => Number.isSafeInteger(size) && size >= 0) ||
      (state.dtype !== 'float32' && state.dtype !== 'int64')
    )
      throw new Error('Invalid SAVA state specification');
    const size = state.shape.reduce((total, dimension) => total * dimension, 1);

    states[state.input] =
      state.dtype === 'int64'
        ? new ort.Tensor('int64', new BigInt64Array(size), state.shape)
        : new ort.Tensor('float32', new Float32Array(size), state.shape);
  }
  if (message.wasmPaths) ort.env.wasm.wasmPaths = message.wasmPaths;
  ort.env.wasm.numThreads = message.numThreads ?? 1;
  let backend: 'wasm' | 'webgpu' = message.backend === 'wasm' ? 'wasm' : 'webgpu';

  try {
    session = await ort.InferenceSession.create(message.modelUrl, {
      executionProviders: [backend],
    });
  } catch (error) {
    if (message.backend !== 'auto') throw error;
    backend = 'wasm';
    session = await ort.InferenceSession.create(message.modelUrl, {
      executionProviders: [backend],
    });
  }
  send({ type: 'ready', backend });
}

function score(output: ort.InferenceSession.ReturnType, name: string): number {
  const value = output[name]?.data[0];

  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)
    throw new Error(`Invalid SAVA output: ${name}`);
  return value;
}

async function frame(message: AudioFrame): Promise<void> {
  if (!session || !manifest) throw new Error('SAVA worker is not loaded');
  if (message.samples.length !== 640 || message.mel.length !== 512)
    throw new Error('Invalid SAVA audio frame');
  const output = await session.run({
    samples: new ort.Tensor('float32', message.samples, [1, 1, 640]),
    mel: new ort.Tensor('float32', message.mel, [1, 1, 512]),
    ...states,
  });

  const speech = score(output, 'speech');

  const speechOnTime = score(output, 'speech_on_time');

  const turn = score(output, 'turn');

  const next: Record<string, ort.Tensor> = {};

  for (const state of manifest.states) {
    const tensor = output[state.output];

    if (!tensor) throw new Error(`Missing SAVA state output: ${state.output}`);
    next[state.input] = tensor;
  }
  states = next;
  send({
    type: 'frame',
    scores: {
      index: message.index,
      speech,
      speechOnTime,
      turn,
      synthetic: message.synthetic,
      validSamples: message.validSamples,
    },
  });
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  queue = queue
    .then(async () => {
      if (failed) return;
      if (event.data.type === 'load') await load(event.data);
      else if (event.data.type === 'frame') await frame(event.data);
      else send({ type: 'flushed' });
    })
    .catch((error: unknown) => {
      failed = true;
      send({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    });
};
