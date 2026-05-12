import { create } from 'zustand'

// ── 모델 리스트 ───────────────────────────────────────────────────────────────
export const AVAILABLE_MODELS = [
	{ id: 'gemini-3-1-flash-lite', label: 'Gemini 3.1 Flash-Lite', shortLabel: 'Gemini 3.1 Flash' },
	{ id: 'qwen3-5', label: 'Qwen3.5-9B', shortLabel: 'Qwen3.5' },
	{ id: 'eeve', label: 'EEVE-Korean-Instruct-10.8B-v1.0-GGUF', shortLabel: 'EEVE' },
	{ id: 'ministral-3', label: 'Ministral-3-14B-Reasoning-2512 ', shortLabel: 'Ministral-3' },
]

// ── 문서 분할 방법 리스트 ─────────────────────────────────────────────────────
export const AVAILABLE_SPLIT_METHODS = [
	{ id: 'recursive', label: '재귀 분할', shortLabel: '재귀' },
	{ id: 'fixed',     label: '고정 길이', shortLabel: '고정' },
	{ id: 'sentence',  label: '문장 단위', shortLabel: '문장' },
	{ id: 'token',     label: '토큰 단위', shortLabel: '토큰' },
]

// ── Default options ──────────────────────────────────────────────────────────
export const DEFAULT_OPTIONS = {
	// Retrieval
	searchMethod: 'hybrid',    // 'bm25' | 'vector' | 'hybrid'
	topK: 5,                   // 검색 문서 수
	reranker: false,
	rerankTopK: 3,             // 리랭크 문서 수
	semanticCache: false,

	// Chunking
	splitMethod: 'recursive',  // 'recursive' | 'fixed' | 'sentence' | 'token'
	chunkSize: 512,
	chunkOverlap: 50,

	// Generation
	model: AVAILABLE_MODELS[0].id,
	temperature: 0.5,
	maxTokens: 1024,

	// Internal
	streamResponse: true,
}

// ── Store ────────────────────────────────────────────────────────────────────
export const useChatStore = create((set, get) => ({
	// ── Messages ─────────────────────────────────────────────────
	messages: [],
	isLoading: false,
	abortController: null,

	// ── UI ────────────────────────────────────────────────────────
	sidebarOpen: true,
	optionsPanelOpen: false,

	// ── Options ───────────────────────────────────────────────────
	options: { ...DEFAULT_OPTIONS },

	// ── Documents ─────────────────────────────────────────────────
	documents: [],

	// ════════════════════════════════════════════════════════════
	// Message Actions
	addMessage: (message) =>
		set((s) => ({
			messages: [
				...s.messages,
				{ id: crypto.randomUUID(), timestamp: new Date().toISOString(), ...message },
			],
		})),

	updateLastMessage: (updater) =>
		set((s) => {
			if (s.messages.length === 0) return {}
			const msgs = [...s.messages]
			const last = msgs[msgs.length - 1]
			msgs[msgs.length - 1] =
				typeof updater === 'function' ? { ...last, ...updater(last) } : { ...last, ...updater }
			return { messages: msgs }
		}),

	clearMessages: () => set({ messages: [] }),

	setLoading: (isLoading) => set({ isLoading }),

	setAbortController: (controller) => set({ abortController: controller }),

	abortRequest: () => {
		const { abortController } = get()
		abortController?.abort()
		set({ abortController: null, isLoading: false })
	},

	// ════════════════════════════════════════════════════════════
	// UI Actions
	toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
	setSidebarOpen: (open) => set({ sidebarOpen: open }),
	toggleOptionsPanel: () => set((s) => ({ optionsPanelOpen: !s.optionsPanelOpen })),
	setOptionsPanelOpen: (open) => set({ optionsPanelOpen: open }),

	// ════════════════════════════════════════════════════════════
	// Options Actions
	setOption: (key, value) =>
		set((s) => ({ options: { ...s.options, [key]: value } })),

	resetOptions: () => set({ options: { ...DEFAULT_OPTIONS } }),

	// ════════════════════════════════════════════════════════════
	// Document Actions
	setDocuments: (documents) => set({ documents }),

	addDocument: (doc) =>
		set((s) => ({ documents: [...s.documents, doc] })),

	removeDocument: (id) =>
		set((s) => ({ documents: s.documents.filter((d) => d.id !== id) })),

	updateDocument: (id, updates) =>
		set((s) => ({
			documents: s.documents.map((d) => (d.id === id ? { ...d, ...updates } : d)),
		})),
}))