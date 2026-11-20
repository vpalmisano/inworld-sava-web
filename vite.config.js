import { copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

function copyRuntimeAssets() {
  let outputDir;

  return {
    name: 'copy-onnx-runtime-assets',
    apply: 'build',
    configResolved(config) {
      outputDir = resolve(config.root, config.build.outDir);
    },
    async writeBundle() {
      await Promise.all(
        [
          'ort.webgpu.min.mjs',
          'ort-wasm-simd-threaded.asyncify.mjs',
          'ort-wasm-simd-threaded.asyncify.wasm',
        ].map(name =>
          copyFile(
            fileURLToPath(new URL(name, import.meta.resolve('onnxruntime-web/webgpu'))),
            resolve(outputDir, name),
          ),
        ),
      );
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [copyRuntimeAssets()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    watch: { ignored: ['**/.venv/**', 'export/**'] },
  },
  preview: { headers: { 'Cross-Origin-Opener-Policy': 'same-origin' } },
  build: {
    emptyOutDir: false,
    minify: false,
    sourcemap: true,
    target: 'es2022',
    lib: {
      entry: {
        index: 'src/index.ts',
        'audio-worklet': 'src/audio-worklet.ts',
        'inference-worker': 'src/inference-worker.ts',
      },
      formats: ['es'],
      fileName: (_format, entryName) => `${entryName}.js`,
    },
    rolldownOptions: {
      external: ['onnxruntime-web/webgpu'],
      output: { paths: { 'onnxruntime-web/webgpu': './ort.webgpu.min.mjs' } },
    },
  },
});
