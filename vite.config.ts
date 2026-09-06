import { defineConfig } from 'vite'
import { resolve } from 'node:path'

// BASE_PATH lets the same build be served from a project subdirectory, which is what
// GitHub Pages does. It is read at build time and baked into import.meta.env.BASE_URL,
// which is how the worker finds the model in public/.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        home: resolve(import.meta.dirname, 'index.html'),
        denoise: resolve(import.meta.dirname, 'denoise/index.html'),
        dereverb: resolve(import.meta.dirname, 'dereverb/index.html'),
      },
    },
    // The weights are the download that matters; keeping the JS in one chunk avoids a
    // waterfall in front of them.
    chunkSizeWarningLimit: 2048,
  },
  server: { host: '127.0.0.1' },
})
