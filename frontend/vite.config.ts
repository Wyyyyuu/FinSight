import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'process.env': {}
  },
  build: {
    // Keep dependency chunks aligned with the routes that import them.
    chunkSizeWarningLimit: 1500,
  },
})
