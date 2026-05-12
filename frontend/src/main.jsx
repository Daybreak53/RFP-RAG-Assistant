import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import App from './App.jsx'
import './index.css'

const queryClient = new QueryClient({
	defaultOptions: {
		queries: {
			retry: 1,
			staleTime: 5 * 60 * 1000,
			refetchOnWindowFocus: false,
		},
	},
})

ReactDOM.createRoot(document.getElementById('root')).render(
	<React.StrictMode>
		<QueryClientProvider client={queryClient}>
			<App />
			<Toaster
				position="top-right"
				gutter={8}
				toastOptions={{
					duration: 3500,
					style: {
						background: '#ffffff',
						color: '#0f172a',
						border: '1px solid #e2e8f0',
						borderRadius: '12px',
						fontSize: '13px',
						fontFamily: '"DM Sans", "Noto Sans KR", sans-serif',
						boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
					},
					success: {
						iconTheme: { primary: '#8b5cf6', secondary: '#ffffff' },
					},
					error: {
						iconTheme: { primary: '#ef4444', secondary: '#ffffff' },
					},
				}}
			/>
		</QueryClientProvider>
	</React.StrictMode>
)