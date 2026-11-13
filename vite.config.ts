import { defineConfig } from 'vite';

function normalizeBase(value: string | undefined): string {
  if (!value) return './';
  return value.endsWith('/') ? value : `${value}/`;
}

export default defineConfig({
  base: normalizeBase(process.env.VITE_BASE_PATH),
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
