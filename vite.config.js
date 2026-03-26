import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: '/facemask-detection/',
  build: { outDir: 'docs' },
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
