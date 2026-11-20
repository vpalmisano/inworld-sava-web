import test from 'node:test';
import assert from 'node:assert/strict';
import { SavaDetector } from '../dist/index.js';

class FakeWorker {
  static last;
  constructor(url) {
    FakeWorker.last = this;
    this.url = url;
    this.terminated = false;
  }

  postMessage(message) {
    if (message.type === 'load' && !FakeWorker.blockLoad)
      queueMicrotask(() => this.onmessage?.({ data: { type: 'ready', backend: 'wasm' } }));
    if (message.type === 'frame' && !FakeWorker.block)
      queueMicrotask(() =>
        this.onmessage?.({
          data: {
            type: 'frame',
            scores: {
              index: message.index,
              speech: message.index === 3 ? 1 : 0,
              speechOnTime: 0,
              turn: 0,
              synthetic: message.synthetic,
              validSamples: message.validSamples,
            },
          },
        }),
      );
    if (message.type === 'flush')
      queueMicrotask(() => this.onmessage?.({ data: { type: 'flushed' } }));
  }

  terminate() {
    this.terminated = true;
  }
}
class FakeNode {
  static last;
  constructor() {
    FakeNode.last = this;
    this.port = {
      onmessage: null,
      postMessage: message => {
        if (message.type === 'flush')
          queueMicrotask(() => this.port.onmessage?.({ data: { type: 'flushed' } }));
      },
    };
  }

  connect() {}

  disconnect() {}
}
class FakeContext {
  sampleRate = 16000;
  state = 'running';
  destination = {};
  audioWorklet = { addModule: async () => {} };
  createMediaStreamSource() {
    return { connect() {}, disconnect() {} };
  }

  createGain() {
    return { gain: { value: 1 }, connect() {}, disconnect() {} };
  }

  close() {
    this.state = 'closed';
    return Promise.resolve();
  }
}
class FakeTrack extends EventTarget {
  kind = 'audio';
  readyState = 'live';
}

globalThis.Worker = FakeWorker;
globalThis.AudioContext = FakeContext;
globalThis.AudioWorkletNode = FakeNode;
globalThis.MediaStream = class {
  constructor(tracks) {
    this.tracks = tracks;
  }
};

function frame(index) {
  return {
    type: 'frame',
    index,
    samples: new Float32Array(640),
    mel: new Float32Array(512),
    validSamples: 640,
    synthetic: false,
  };
}

const options = { modelUrl: '/sava.onnx', manifestUrl: '/sava.json' };

test('start, deliver delayed speech start, and flush without owning the track', async () => {
  const starts = [];

  const turns = [];

  const track = new FakeTrack();

  const detector = new SavaDetector({
    ...options,
    onSpeechStart: index => starts.push(index),
    onTurn: turn => turns.push(turn),
  });

  await detector.start(track);
  assert.equal(detector.backend, 'wasm');
  assert.equal(
    FakeWorker.last.url.href,
    new URL('../dist/inference-worker.js', import.meta.url).href,
  );
  FakeNode.last.port.onmessage({ data: frame(3) });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.deepEqual(starts, [0]);
  await detector.stop();
  assert.equal(turns.length, 1);
  assert.equal(turns[0].reason, 'end');
  assert.equal(FakeWorker.last.terminated, true);
  assert.equal(track.readyState, 'live');
});

test('stop cancels a pending model load', async () => {
  FakeWorker.blockLoad = true;
  const detector = new SavaDetector(options);

  const starting = detector.start(new FakeTrack());

  await detector.stop();
  await assert.rejects(starting, /startup cancelled/);
  assert.equal(FakeWorker.last.terminated, true);
  FakeWorker.blockLoad = false;
});

test('queue overload stops instead of silently dropping audio', async () => {
  FakeWorker.block = true;
  const errors = [];

  const detector = new SavaDetector({
    ...options,
    maxPendingFrames: 1,
    onError: error => errors.push(error.message),
  });

  await detector.start(new FakeTrack());
  FakeNode.last.port.onmessage({ data: frame(0) });
  FakeNode.last.port.onmessage({ data: frame(1) });
  assert.match(errors[0], /is slow/);
  assert.equal(FakeWorker.last.terminated, true);
  FakeWorker.block = false;
});
