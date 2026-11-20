import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { access, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { ESLint } from 'eslint';
import { createServer } from 'vite';

test('the packaged inference worker exists', async () => {
  await assert.doesNotReject(
    access(fileURLToPath(new URL('../dist/inference-worker.js', import.meta.url))),
  );
  await assert.doesNotReject(
    access(fileURLToPath(new URL('../dist/ort.webgpu.min.mjs', import.meta.url))),
  );
  await assert.doesNotReject(
    access(fileURLToPath(new URL('../dist/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url))),
  );
  await assert.doesNotReject(
    access(fileURLToPath(new URL('../dist/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url))),
  );
});

test('single build emits the library, a self-contained worklet, worker, and runtime assets', async () => {
  const outputDir = await mkdtemp(join(tmpdir(), 'sava-worker-'));

  try {
    await promisify(execFile)(process.execPath, [
      fileURLToPath(new URL('../node_modules/vite/bin/vite.js', import.meta.url)),
      'build',
      '--outDir',
      outputDir,
    ]);
    const index = await readFile(join(outputDir, 'index.js'), 'utf8');

    assert.doesNotMatch(index, /^\s*import(?:\s|\()/m);
    await assert.rejects(access(join(outputDir, 'features.js')), { code: 'ENOENT' });
    await assert.rejects(access(join(outputDir, 'policy.js')), { code: 'ENOENT' });
    await access(join(outputDir, 'inference-worker.js'));
    const worklet = await readFile(join(outputDir, 'audio-worklet.js'), 'utf8');

    assert.match(worklet, /registerProcessor/);
    assert.doesNotMatch(worklet, /^\s*import(?:\s|\()/m);
    for (const name of [
      'ort.webgpu.min.mjs',
      'ort-wasm-simd-threaded.asyncify.mjs',
      'ort-wasm-simd-threaded.asyncify.wasm',
    ]) {
      assert.deepEqual(
        await readFile(join(outputDir, name)),
        await readFile(fileURLToPath(new URL(name, import.meta.resolve('onnxruntime-web/webgpu')))),
      );
    }
  } finally {
    await rm(outputDir, { recursive: true, force: true });
  }
});

test('demo watcher excludes the Python virtual environment', async () => {
  const root = fileURLToPath(new URL('..', import.meta.url));

  const server = await createServer({ root, logLevel: 'silent', server: { middlewareMode: true } });

  try {
    assert.equal(
      server.watcher._isIgnored(join(root, '.venv/lib/python3.14/site-packages/triton')),
      true,
    );
    assert.equal(server.watcher._isIgnored(join(root, 'demo/index.html')), false);
    assert.equal(server.watcher._isIgnored(join(root, 'models/sava.json')), false);
  } finally {
    await server.close();
  }
});

test('lint requires blank lines after functions, methods, and variables', async () => {
  const eslint = new ESLint();

  const cases = [
    {
      code: 'const first = 1;\nconst second = 2;\n',
      rule: '@stylistic/padding-line-between-statements',
    },
    {
      code: 'function first() {}\nfunction second() {}\n',
      rule: '@stylistic/padding-line-between-statements',
    },
    {
      code: 'export const first = 1;\nexport function second() {}\n',
      rule: '@stylistic/padding-line-between-statements',
    },
    {
      code: 'export function first() {}\nexport const second = 1;\n',
      rule: '@stylistic/padding-line-between-statements',
    },
    {
      code: 'class Example {\n  first() {}\n  second() {}\n}\n',
      rule: '@stylistic/lines-between-class-members',
    },
  ];

  for (const { code, rule } of cases) {
    const [result] = await eslint.lintText(code, { filePath: 'src/spacing-fixture.ts' });

    assert.ok(result.messages.some(message => message.ruleId === rule));
  }

  const [result] = await eslint.lintText(
    'const first = 1;\n\nexport function second() {}\n\nclass Example {\n  first() {}\n\n  second() {}\n}\n',
    { filePath: 'src/spacing-fixture.ts' },
  );

  assert.ok(!result.messages.some(message => message.ruleId?.startsWith('@stylistic/')));
});
