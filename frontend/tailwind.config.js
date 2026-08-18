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
        // Point at the CSS variables declared in src/index.css so every token
        // is theme-aware. `html.dark` swaps the variables; no `dark:` variant
        // is needed on individual elements.
        bg: {
          main: "var(--color-bg-main)",
          sidebar: "var(--color-bg-sidebar)",
          hover: "var(--color-bg-hover)",
        },
        text: {
          main: "var(--color-text-main)",
          muted: "var(--color-text-muted)",
          dim: "var(--color-text-dim)",
        },
        border: {
          light: "var(--color-border-light)",
        },
        action: {
          primary: "var(--color-action-primary)",
          hover: "var(--color-action-hover)",
          on: "var(--color-on-action)",
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
