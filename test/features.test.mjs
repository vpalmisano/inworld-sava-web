import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  configFile: false,
  logLevel: 'silent',
  server: { middlewareMode: true, watch: null },
});

let FRAME_SAMPLES;

let SavaFeatures;

try {
  ({ FRAME_SAMPLES, SavaFeatures } = await server.ssrLoadModule('/src/features.ts'));
} finally {
  await server.close();
}

test('silence has the reference log floor', () => {
  const features = new SavaFeatures().extract(new Float32Array(FRAME_SAMPLES));

  assert.equal(features.length, 512);
  assert.ok(features.every(value => value === -1.5));
});

test('causal context carries across frames and reset clears it', () => {
  const frontend = new SavaFeatures();

  const input = new Float32Array(FRAME_SAMPLES);

  input[639] = 1;
  frontend.extract(input);
  const following = frontend.extract(new Float32Array(FRAME_SAMPLES));

  assert.ok(following.subarray(0, 256).some(value => value > -1.5));
  assert.ok(following.subarray(256).every(value => value === -1.5));
  frontend.reset();
  assert.ok(frontend.extract(new Float32Array(FRAME_SAMPLES)).every(value => value === -1.5));
  assert.throws(() => frontend.extract(new Float32Array(639)), /640/);
});

test('worklet emits continuous audio and flushes padded tail and lookahead', async () => {
  let Processor;

  globalThis.sampleRate = 16000;
  globalThis.AudioWorkletProcessor = class {
    port = {
      messages: [],
      postMessage(message) {
        this.messages.push(message);
      },
      onmessage: null,
    };
  };
  globalThis.registerProcessor = (name, type) => {
    assert.equal(name, 'sava-features');
    Processor = type;
  };
  await import('../dist/audio-worklet.js');
  const worklet = new Processor();

  for (let i = 0; i < 6; i++) worklet.process([[new Float32Array(128)]], [[new Float32Array(128)]]);
  assert.equal(worklet.port.messages.length, 1);
  worklet.port.onmessage({ data: { type: 'flush' } });
  const frames = worklet.port.messages.filter(message => message.type === 'frame');

  assert.deepEqual(
    frames.map(frame => frame.validSamples),
    [640, 128, 0, 0, 0],
  );
  assert.deepEqual(
    frames.map(frame => frame.index),
    [0, 1, 2, 3, 4],
  );
  assert.equal(worklet.port.messages.at(-1).type, 'flushed');
  assert.equal(worklet.process([], []), false);
});
