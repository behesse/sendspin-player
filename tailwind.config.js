/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/**/*.js",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark theme colors
        theme: {
          bg: {
            DEFAULT: '#111827',      // gray-900 - main background
            card: '#1f2937',        // gray-800 - card background
            input: '#374151',       // gray-700 - input background
            disabled: '#374151',    // gray-700 - disabled elements
          },
          text: {
            DEFAULT: '#f9fafb',     // gray-50 - primary text
            muted: '#d1d5db',        // gray-300 - muted text
            secondary: '#9ca3af',    // gray-400 - secondary text
          },
          border: {
            DEFAULT: '#374151',      // gray-700 - default borders
            light: '#4b5563',        // gray-600 - lighter borders
          },
          primary: {
            DEFAULT: '#2563eb',      // blue-600 - primary actions
            hover: '#1d4ed8',        // blue-700 - primary hover
            light: '#3b82f6',        // blue-500 - primary light
          },
          success: {
            DEFAULT: '#166534',      // green-900 - success background
            text: '#bbf7d0',          // green-200 - success text
            border: '#16a34a',        // green-600 - success border
            indicator: '#10b981',     // green-500 - success indicator
          },
          error: {
            DEFAULT: '#7f1d1d',      // red-900 - error background
            text: '#fecaca',          // red-200 - error text
            border: '#dc2626',        // red-600 - error border
            indicator: '#ef4444',     // red-500 - error indicator
          },
          warning: {
            DEFAULT: '#854d0e',       // yellow-900 - warning background
            text: '#fef08a',          // yellow-200 - warning text
            border: '#ca8a04',        // yellow-600 - warning border
            button: '#ca8a04',        // yellow-600 - warning button
            'button-hover': '#a16207', // yellow-700 - warning button hover
          },
          action: {
            start: '#2563eb',         // blue-600 - start button
            'start-hover': '#1d4ed8', // blue-700 - start hover
            stop: '#dc2626',          // red-600 - stop button
            'stop-hover': '#b91c1c',  // red-700 - stop hover
            restart: '#ca8a04',      // yellow-600 - restart button
            'restart-hover': '#a16207', // yellow-700 - restart hover
          },
        },
      },
    },
  },
  plugins: [],
}

