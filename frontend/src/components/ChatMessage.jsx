import { useState, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
	Copy, Check, ChevronDown, ChevronUp, Clock, Calendar, Activity,
	Bot, User, Zap, Search, Database, Layers, Info, AlertTriangle,
	Scissors, Thermometer, Hash, ListFilter, Filter,
} from 'lucide-react'
import ThinkingIndicator from './ThinkingIndicator'
import { AVAILABLE_MODELS, AVAILABLE_SPLIT_METHODS } from '../store/chatStore'

const METHOD_LABEL = { bm25: 'BM25', vector: 'Vector', hybrid: 'Hybrid' }

// ── 복사 버튼 ────────────────────────────────────────────────────────────────
function CopyButton({ text, className = '' }) {
	const [copied, setCopied] = useState(false)

	const handle = async () => {
		await navigator.clipboard.writeText(text).catch(() => {})
		setCopied(true)
		setTimeout(() => setCopied(false), 2000)
	}

	return (
		<button
			onClick={handle}
			className={`flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium text-slate-400 hover:text-slate-700 bg-transparent hover:bg-slate-100 rounded-md transition-colors ${className}`}
			title="복사하기"
		>
			{copied ? <Check size={13} className="text-green-500" /> : <Copy size={13} />}
			{copied ? '복사됨' : '복사'}
		</button>
	)
}

// ── 메타데이터 배지 ──────────────────────────────────────────────────────────
function MetaBadge({ icon, label, variant = 'default' }) {
	const styles = {
		default: 'bg-slate-100 text-slate-600 border-slate-200/60',
		violet:  'bg-violet-50 text-violet-600 border-violet-200/60',
		yellow:  'bg-yellow-50 text-yellow-600 border-yellow-200/60',
	}
	return (
		<div className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] border ${styles[variant]}`}>
			{icon}
			<span>{label}</span>
		</div>
	)
}

const formatDateTime = (isoString) => {
	if (!isoString) return '시간 정보 없음'
	const date = new Date(isoString)
	if (isNaN(date.getTime())) return '시간 정보 없음'
	return date.toLocaleString('ko-KR', {
		year: 'numeric', month: '2-digit', day: '2-digit',
		hour: '2-digit', minute: '2-digit', second: '2-digit',
	})
}

// ── ChatMessage 본체 ─────────────────────────────────────────────────────────
export default function ChatMessage({ message }) {
	const { role, content, metadata, streaming, error, timestamp } = message
	const isUser = role === 'user'
	const [showSources, setShowSources] = useState(false)
	const [showDetails, setShowDetails] = useState(false)

	const showThinking = streaming && !content

	const meta = useMemo(() => {
		const opts = metadata?.options ?? {}
		const modelId = opts.model ?? metadata?.model
		return {
			model:        modelId,
			displayModel: AVAILABLE_MODELS.find((m) => m.id === modelId)?.shortLabel ?? modelId,
			searchMethod: opts.searchMethod ?? metadata?.searchMethod ?? metadata?.search_method,
			chunkSize:    opts.chunkSize    ?? metadata?.chunkSize,
			chunkOverlap: opts.chunkOverlap ?? metadata?.chunkOverlap,
			splitMethod:  opts.splitMethod  ?? metadata?.splitMethod,
			splitLabel:   AVAILABLE_SPLIT_METHODS.find((m) => m.id === (opts.splitMethod ?? metadata?.splitMethod))?.shortLabel,
			topK:         opts.topK         ?? metadata?.topK,
			reranker:     opts.reranker     ?? metadata?.reranker,
			rerankTopK:   opts.rerankTopK   ?? metadata?.rerankTopK,
			temperature:  opts.temperature  ?? metadata?.temperature,
			maxTokens:    opts.maxTokens    ?? metadata?.maxTokens,
			cached:       opts.semanticCache ?? metadata?.cached,
			tokensUsed:   metadata?.tokensUsed   ?? 0,
			responseTime: metadata?.responseTime ?? 0,
		}
	}, [metadata])

	const bubbleClasses = isUser
		? 'bg-violet-50 border border-violet-100 rounded-tr-sm text-slate-800'
		: error
			? 'bg-red-50 border border-red-100 rounded-tl-sm text-red-600'
			: 'bg-white border border-slate-200 rounded-tl-sm text-slate-700'

	return (
		<div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} animate-slide-up`}>
			<div className={`flex max-w-full gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>

				{/* Avatar */}
				<div className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center mt-1 shadow-sm ${isUser ? 'bg-violet-600 text-white' : 'bg-white border border-slate-200 text-violet-600'}`}>
					{isUser ? <User size={14} /> : <Bot size={14} />}
				</div>

				{/* Bubble */}
				<div className={`group relative max-w-[82%] md:max-w-[75%] rounded-2xl px-5 py-3.5 shadow-sm ${bubbleClasses}`}>

					{/* Content */}
					{showThinking ? (
						<ThinkingIndicator />
					) : isUser ? (
						<p className="text-sm leading-relaxed text-slate-800 whitespace-pre-wrap break-words">
							{content}
						</p>
					) : error ? (
						<div className="flex items-center gap-2 text-sm text-red-500">
							<AlertTriangle size={14} />
							{content}
						</div>
					) : (
						<div className="text-sm markdown-body">
							<ReactMarkdown
								remarkPlugins={[remarkGfm]}
								components={{
									code({ inline, className, children, ...props }) {
										const match = /language-(\w+)/.exec(className || '')
										return !inline && match ? (
											<SyntaxHighlighter
												{...props}
												style={oneDark}
												language={match[1]}
												PreTag="div"
												className="rounded-lg text-xs"
											>
												{String(children).replace(/\n$/, '')}
											</SyntaxHighlighter>
										) : (
											<code {...props} className={className}>{children}</code>
										)
									},
								}}
							>
								{content}
							</ReactMarkdown>
						</div>
					)}

					{/* 하단 액션바 */}
					{!isUser && !showThinking && !error && (
						<div className="mt-3 pt-2 border-t border-slate-100 flex flex-col">

							<div className="flex items-center justify-between">
								<button
									onClick={() => setShowDetails(!showDetails)}
									className="flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-violet-500 transition-colors"
								>
									<Info size={13} />
									{showDetails ? '설정 정보 숨기기' : '설정 정보 보기'}
									{showDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
								</button>
								{content && <CopyButton text={content} />}
							</div>

							{/* 펼쳐지는 설정 정보 */}
							{showDetails && (
								<div className="mt-3 space-y-3 animate-fade-in">

									{/* ── Retrieval 배지 ─────────────────── */}
									<div className="space-y-1">
										<p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">Retrieval</p>
										<div className="flex flex-wrap gap-1.5">
											{meta.displayModel && (
												<MetaBadge icon={<Bot size={10} />} label={meta.displayModel} />
											)}
											{meta.searchMethod && (
												<MetaBadge icon={<Search size={10} />} label={METHOD_LABEL[meta.searchMethod] ?? meta.searchMethod} />
											)}
											{meta.topK != null && (
												<MetaBadge icon={<ListFilter size={10} />} label={`Top-${meta.topK}`} />
											)}
											{meta.reranker === true && (
												<MetaBadge icon={<Layers size={10} />} label="Reranker" variant="violet" />
											)}
											{meta.reranker === true && meta.rerankTopK != null && (
												<MetaBadge icon={<Filter size={10} />} label={`Rerank-${meta.rerankTopK}`} variant="violet" />
											)}
											{meta.cached === true && (
												<MetaBadge
													icon={<Zap size={10} className="fill-yellow-500 text-yellow-500" />}
													label="Cached"
													variant="yellow"
												/>
											)}
										</div>
									</div>

									{/* ── Chunking 배지 ──────────────────── */}
									<div className="space-y-1">
										<p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">Chunking</p>
										<div className="flex flex-wrap gap-1.5">
											{meta.splitLabel && (
												<MetaBadge icon={<Scissors size={10} />} label={meta.splitLabel} />
											)}
											{meta.chunkSize != null && (
												<MetaBadge icon={<Database size={10} />} label={`Size: ${meta.chunkSize}`} />
											)}
											{meta.chunkOverlap != null && (
												<MetaBadge icon={<Database size={10} />} label={`Overlap: ${meta.chunkOverlap}`} />
											)}
										</div>
									</div>

									{/* ── Generation 배지 ────────────────── */}
									<div className="space-y-1">
										<p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">Generation</p>
										<div className="flex flex-wrap gap-1.5">
											{meta.temperature != null && (
												<MetaBadge icon={<Thermometer size={10} />} label={`T: ${Number(meta.temperature).toFixed(1)}`} />
											)}
											{meta.maxTokens != null && (
												<MetaBadge icon={<Hash size={10} />} label={`${Number(meta.maxTokens).toLocaleString()} token`} />
											)}
										</div>
									</div>

									{/* ── 성능 지표 ──────────────────────── */}
									<div className="flex flex-col gap-1 text-[10px] text-slate-400 font-mono">
										<div className="flex items-center gap-1.5">
											<Activity size={11} />
											<span>Tokens: {meta.tokensUsed}</span>
											<span className="mx-1 opacity-30">|</span>
											<Clock size={11} />
											<span>Latency: {(meta.responseTime / 1000).toFixed(2)}s</span>
										</div>
										<div className="flex items-center gap-1.5">
											<Calendar size={11} />
											<span>답변일시: {formatDateTime(timestamp)}</span>
										</div>
									</div>

								</div>
							)}
						</div>
					)}

					{/* 참고 문서 */}
					{metadata?.sources?.length > 0 && (
						<div className="mt-3 overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
							<button
								onClick={() => setShowSources(!showSources)}
								className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100 transition-colors"
							>
								<div className="flex items-center gap-1.5">
									<Search size={12} className="text-violet-500" />
									<span>참고 문서 ({metadata.sources.length}건)</span>
								</div>
								{showSources ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
							</button>

							{showSources && (
								<div className="p-3 border-t border-slate-200 space-y-2 max-h-60 overflow-y-auto">
									{metadata.sources.map((src, i) => (
										<div key={i} className="flex gap-2 text-xs bg-white p-2 rounded border border-slate-200 shadow-sm">
											<div className="w-5 h-5 rounded-full bg-violet-100 text-violet-700 flex items-center justify-center flex-shrink-0 font-mono text-[10px] font-bold">
												{i + 1}
											</div>
											<div className="flex-1 overflow-hidden">
												<p className="font-medium text-slate-700 truncate">{src.document}</p>
												{src.content && (
													<p className="text-slate-500 mt-1 line-clamp-2 leading-relaxed">{src.content}</p>
												)}
												{src.score && (
													<p className="text-[10px] text-slate-400 mt-1 font-mono">
														유사도: {src.score.toFixed(3)}
													</p>
												)}
											</div>
										</div>
									))}
								</div>
							)}
						</div>
					)}
				</div>
			</div>
		</div>
	)
}