import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));

function normalizeBase(value: string | undefined): string {
  if (!value) return './';
  return value.endsWith('/') ? value : `${value}/`;
}

export default defineConfig({
  base: normalizeBase(process.env.VITE_BASE_PATH),
  resolve: {
    alias: {
      'onnxruntime-web/wasm': resolve(
        __dirname,
        'node_modules/onnxruntime-web/dist/ort.wasm.min.mjs'
      ),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022',
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
});
