import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/facemask-detection/',
  build: { outDir: 'docs', emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
