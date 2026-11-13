import { copyFileSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const wasmSrc = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDst = join(root, 'public', 'mediapipe', 'wasm');
mkdirSync(wasmDst, { recursive: true });
for (const file of readdirSync(wasmSrc)) {
  copyFileSync(join(wasmSrc, file), join(wasmDst, file));
}

const fontSrc = join(root, 'node_modules', '@fontsource', 'vazirmatn', 'files');
const fontDst = join(root, 'public', 'fonts');
mkdirSync(fontDst, { recursive: true });
const weights = ['400', '500', '700', '800'];
const subsets = ['arabic', 'latin'];
for (const subset of subsets) {
  for (const weight of weights) {
    const filename = `vazirmatn-${subset}-${weight}-normal.woff2`;
    copyFileSync(join(fontSrc, filename), join(fontDst, filename));
  }
}

const ortSrc = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortDst = join(root, 'public', 'ort');
mkdirSync(ortDst, { recursive: true });
const ortFiles = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];
for (const file of ortFiles) {
  copyFileSync(join(ortSrc, file), join(ortDst, file));
}

console.log('Synced self-hosted MediaPipe WASM, Vazirmatn fonts, and ORT-Web WASM assets to public/.');
