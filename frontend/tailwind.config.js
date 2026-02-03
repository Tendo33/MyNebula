/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Notion-like Semantic Palette
        bg: {
          main: "#FFFFFF",        // White
          sidebar: "#F7F7F5",     // Light Gray
          hover: "#EFEFED",       // Hover Gray
        },
        text: {
          main: "#37352F",        // Dark Gray (Primary)
          muted: "#787774",       // Medium Gray (Secondary)
          dim: "#9B9A97",         // Light Gray (Placeholder/Icon)
        },
        border: {
          light: "#E9E9E7",       // Pale Gray
        },
        action: {
          primary: "#2383E2",     // Notion Blue
          hover: "#1C6EB4",       // Darker Blue
        },
        // Dark Mode Support (Optional/Mapped)
        dark: {
          bg: {
            main: "#191919",
            sidebar: "#202020",
          },
          text: {
            main: "#D4D4D4",
          },
          border: "#2F2F2F",
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        serif: ['Lyon-Text', 'Georgia', 'serif'], // Notion uses serif for some content
      },
      boxShadow: {
        'sm': 'rgba(15, 15, 15, 0.1) 0px 0px 0px 1px, rgba(15, 15, 15, 0.1) 0px 2px 4px',
        'md': '0px 1px 4px rgba(0, 0, 0, 0.08)',
      }
    },
  },
  plugins: [],
}
