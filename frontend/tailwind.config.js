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
          elevated: "var(--color-bg-elevated)",
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
        danger: {
          DEFAULT: "var(--color-danger)",
          bg: "var(--color-danger-bg)",
          foreground: "var(--color-danger-foreground)",
        },
        success: {
          DEFAULT: "var(--color-success)",
          bg: "var(--color-success-bg)",
        },
        warning: {
          DEFAULT: "var(--color-warning)",
          bg: "var(--color-warning-bg)",
          foreground: "var(--color-warning-foreground)",
        },
        overlay: "var(--color-overlay)",
        link: "var(--color-link)",
        dark: {
          bg: {
            main: "var(--color-bg-main)",
            sidebar: "var(--color-bg-sidebar)",
            hover: "var(--color-bg-hover)",
          },
          text: {
            main: "var(--color-text-main)",
          },
          border: "var(--color-border-light)",
        }
      },
      fontFamily: {
        sans: ['Geist Variable', 'Geist', 'Arial', 'sans-serif'],
        heading: ['Geist Variable', 'Geist', 'Arial', 'sans-serif'],
        mono: ['Geist Mono Variable', 'Geist Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        'sm': '0 1px 1px rgba(0, 0, 0, 0.04)',
        'md': '0 2px 2px rgba(0, 0, 0, 0.04), 0 8px 16px -4px rgba(0, 0, 0, 0.08)',
      }
    },
  },
  plugins: [],
}
