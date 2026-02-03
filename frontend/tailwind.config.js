/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        nebula: {
          bg: "#0B0B10",
          surface: "#15151A",
          surfaceHighlight: "#1E1E24",
          border: "#2D2D3A",
          primary: "#00FFFF",     // Cyan
          secondary: "#7B61FF",   // Deep Purple
          accent: "#FF00FF",      // Magenta
          text: {
            main: "#E0E0FF",
            muted: "#94A3B8",
            dim: "#64748B"
          }
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'nebula-gradient': 'linear-gradient(to right bottom, #0B0B10, #15151A)',
        'glow-primary': 'conic-gradient(from 180deg at 50% 50%, #00FFFF33 0deg, transparent 180deg)',
      }
    },
  },
  plugins: [],
}
