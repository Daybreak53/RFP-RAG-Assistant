import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

// ── Axios Instance ──────────────────────────────────────────────────────────
export const apiClient = axios.create({
	baseURL: BASE_URL,
	headers: { 'Content-Type': 'application/json' },
	timeout: 60_000,
})

apiClient.interceptors.response.use(
	(res) => res,
	(err) => {
		const msg =
			err.response?.data?.detail ||
			err.response?.data?.message ||
			err.message ||
			'알 수 없는 오류가 발생했습니다'
		return Promise.reject(new Error(msg))
	},
)

// ── Chat API ────────────────────────────────────────────────────────────────
export const chatApi = {
	send: (payload) =>
		apiClient.post('/chat', payload).then((r) => r.data),

	stream: async (payload, onChunk, onMetadata, signal) => {
		const res = await fetch(`${BASE_URL}/chat/stream`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(payload),
			signal,
		})

		if (!res.ok) {
			const errBody = await res.json().catch(() => ({}))
			throw new Error(errBody.detail || `HTTP ${res.status}`)
		}

		const reader = res.body.getReader()
		const decoder = new TextDecoder()
		let buffer = ''

		while (true) {
			const { done, value } = await reader.read()
			if (done) break

			buffer += decoder.decode(value, { stream: true })
			const blocks = buffer.split(/\r?\n\r?\n/) 
			buffer = blocks.pop() ?? ''

			for (const block of blocks) {
				const lines = block.split(/\r?\n/);
				for (const line of lines) {
					const trimmed = line.trim()
					if (!trimmed.startsWith('data:')) continue
					const raw = trimmed.slice(5).trim()
					if (raw === '[DONE]') return

					try {
						const parsed = JSON.parse(raw)
						if (parsed.type === 'content') onChunk(parsed.text ?? '')
						if (parsed.type === 'metadata') {
							const { ...meta } = parsed
							onMetadata(meta)
						}
					} catch (e) {
						console.warn('SSE 파싱 에러 (무시됨):', raw, e)
					}
				}
			}
		}
	},
}

// ── Documents API ───────────────────────────────────────────────────────────
export const documentsApi = {
	list: () => apiClient.get('/documents').then((r) => r.data),

	upload: (file, onProgress) => {
		const form = new FormData()
		form.append('file', file)
		return apiClient
			.post('/documents/upload', form, {
				headers: { 'Content-Type': 'multipart/form-data' },
				onUploadProgress: (e) => {
					if (e.total) {
						onProgress?.(Math.round((e.loaded * 100) / e.total))
					}
				},
			})
			.then((r) => r.data)
	},

	delete: (id) =>
		apiClient.delete(`/documents/${id}`).then((r) => r.data),
}
