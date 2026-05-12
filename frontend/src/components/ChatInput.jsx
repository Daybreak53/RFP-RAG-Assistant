import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, Square, Bot, Search, Database, Layers, Zap, Thermometer, ListFilter, Scissors, Hash } from 'lucide-react'
import { useChatStore, AVAILABLE_MODELS, AVAILABLE_SPLIT_METHODS } from '../store/chatStore'

export default function ChatInput({ onSend, onAbort }) {
	const [value, setValue] = useState('')
	const textareaRef = useRef(null)
	const isLoading = useChatStore((s) => s.isLoading)
	const options = useChatStore((s) => s.options)

	const resize = () => {
		const el = textareaRef.current
		if (!el) return
		el.style.height = 'auto'
		el.style.height = `${Math.min(el.scrollHeight, 200)}px`
	}

	useEffect(() => { resize() }, [value])

	const handleChange = (e) => setValue(e.target.value)

	const handleSubmit = useCallback(() => {
		const trimmed = value.trim()
		if (!trimmed || isLoading) return
		onSend(trimmed)
		setValue('')
	}, [value, isLoading, onSend])

	const handleKeyDown = (e) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault()
			handleSubmit()
		}
	}

	const selectedModel = AVAILABLE_MODELS.find((m) => m.id === options.model)
	const displayModelName = selectedModel?.shortLabel ?? options.model

	return (
		<div className="space-y-3">
			<div className="relative flex flex-col w-full bg-white border border-slate-200 rounded-2xl shadow-sm focus-within:border-violet-400 focus-within:ring-1 focus-within:ring-violet-400 transition-all">
				<textarea
					ref={textareaRef}
					value={value}
					onChange={handleChange}
					onKeyDown={handleKeyDown}
					placeholder="메시지를 입력하세요... (Shift+Enter로 줄바꿈)"
					className="w-full max-h-[200px] bg-transparent text-slate-800 placeholder:text-slate-400 text-sm px-4 py-3.5 resize-none focus:outline-none"
					rows={1}
				/>

				<div className="flex items-center justify-between px-3 pb-3">
					<div className="flex items-center gap-2" />

					{isLoading ? (
						<button
							onClick={onAbort}
							className="w-8 h-8 rounded-xl flex items-center justify-center transition-all bg-red-50 border border-red-200 text-red-500 hover:bg-red-100 hover:border-red-300"
							title="응답 중단"
						>
							<Square size={13} fill="currentColor" />
						</button>
					) : (
						<button
							onClick={handleSubmit}
							disabled={!value.trim()}
							className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150 disabled:opacity-40 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:border-slate-200 disabled:text-slate-400 bg-violet-600 hover:bg-violet-500 text-white border border-violet-500 enabled:shadow-[0_2px_8px_rgba(139,92,246,0.3)]"
							title="전송 (Enter)"
						>
							<Send size={13} />
						</button>
					)}
				</div>
			</div>

			{/* Footer 배지 */}
			<div className="flex flex-wrap items-center justify-end gap-1.5 select-none opacity-80">
				{/* 모델 */}
				<Badge icon={<Bot size={10} />} label={displayModelName} />

				{/* 검색 방식 */}
				<Badge icon={<Search size={10} />} label={options.searchMethod} className="capitalize" />

				{/* 검색 문서 수 */}
				<Badge icon={<ListFilter size={10} />} label={`Top-${options.topK}`} />

				{/* 청크 */}
				<Badge icon={<Scissors size={10} />} label={AVAILABLE_SPLIT_METHODS.find((m) => m.id === options.splitMethod).shortLabel} />
				<Badge icon={<Database size={10} />} label={`Chunk: ${options.chunkSize} / ${options.chunkOverlap}`} />

				{/* 온도 */}
				<Badge icon={<Thermometer size={10} />} label={`T: ${options.temperature.toFixed(1)}`} />
				<Badge icon={<Hash size={10} />} label={`${Number(options.maxTokens).toLocaleString()} token`} />

				{/* Reranker (활성 시) */}
				{options.reranker && (
					<Badge
						icon={<Layers size={10} />}
						label={`Rerank-${options.rerankTopK}`}
						className="bg-violet-50 text-violet-600 border-violet-200/60"
					/>
				)}

				{/* 시맨틱 캐시 (활성 시) */}
				{options.semanticCache && (
					<Badge
						icon={<Zap size={10} className="fill-yellow-500 text-yellow-500" />}
						label="Cache"
						className="bg-yellow-50 text-yellow-600 border-yellow-200/60"
					/>
				)}
			</div>
		</div>
	)
}

// ── 배지 공통 컴포넌트 ────────────────────────────────────────────────────────
function Badge({ icon, label, className = '' }) {
	return (
		<div className={`flex items-center gap-1 px-2 py-1 rounded-md bg-slate-100 text-slate-600 border border-slate-200/60 text-[10px] ${className}`}>
			{icon}
			<span className="font-medium">{label}</span>
		</div>
	)
}