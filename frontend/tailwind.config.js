/** @type {import('tailwindcss').Config} */
export default {
	content: [
		'./index.html',
		'./src/**/*.{js,ts,jsx,tsx}',
	],
	theme: {
		extend: {
			fontFamily: {
				sans: ['"DM Sans"', '"Noto Sans KR"', 'sans-serif'],
				mono: ['"DM Mono"', 'monospace'],
				display: ['"Syne"', '"Noto Sans KR"', 'sans-serif'],
			},
			colors: {
				surface: {
					DEFAULT: '#111113',
					card: '#18181b',
					elevated: '#1f1f23',
				},
			},
			animation: {
				'dot-bounce': 'dotBounce 1.4s ease-in-out infinite',
				'slide-up': 'slideUp 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
				'slide-in-left': 'slideInLeft 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
				'slide-in-right': 'slideInRight 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
				'fade-in': 'fadeIn 0.25s ease-out',
				'shimmer': 'shimmer 2s linear infinite',
			},
			keyframes: {
				dotBounce: {
					'0%, 80%, 100%': { transform: 'scale(0.4)', opacity: '0.3' },
					'40%': { transform: 'scale(1)', opacity: '1' },
				},
				slideUp: {
					'0%': { opacity: '0', transform: 'translateY(12px)' },
					'100%': { opacity: '1', transform: 'translateY(0)' },
				},
				slideInLeft: {
					'0%': { opacity: '0', transform: 'translateX(-12px)' },
					'100%': { opacity: '1', transform: 'translateX(0)' },
				},
				slideInRight: {
					'0%': { opacity: '0', transform: 'translateX(12px)' },
					'100%': { opacity: '1', transform: 'translateX(0)' },
				},
				fadeIn: {
					'0%': { opacity: '0' },
					'100%': { opacity: '1' },
				},
				shimmer: {
					'0%': { backgroundPosition: '-200% 0' },
					'100%': { backgroundPosition: '200% 0' },
				},
			},
		},
	},
	plugins: [],
}