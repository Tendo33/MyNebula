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
        bg: {
          main: "#FBFAF6",
          sidebar: "#F2EFE6",
          hover: "#ECE6DA",
        },
        text: {
          main: "#272B35",
          muted: "#616A78",
          dim: "#8A92A0",
        },
        border: {
          light: "#DAD5C8",
        },
        action: {
          primary: "#2D59C8",
          hover: "#2347A3",
        },
        dark: {
          bg: {
            main: "#171A22",
            sidebar: "#1D212C",
          },
          text: {
            main: "#ECE8DE",
          },
          border: "#343B4A",
        }
      },
      fontFamily: {
        sans: ['Avenir Next', 'Segoe UI', '-apple-system', 'BlinkMacSystemFont', 'Helvetica Neue', 'sans-serif'],
        heading: ['Avenir Next', 'Segoe UI', '-apple-system', 'BlinkMacSystemFont', 'Helvetica Neue', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        serif: ['Spectral', 'Georgia', 'serif'],
      },
      boxShadow: {
        'sm': '0 1px 2px rgba(34, 38, 47, 0.05), 0 0 0 1px rgba(77, 88, 107, 0.08)',
        'md': '0 14px 34px -24px rgba(24, 30, 45, 0.28)',
      }
    },
  },
  plugins: [],
}
