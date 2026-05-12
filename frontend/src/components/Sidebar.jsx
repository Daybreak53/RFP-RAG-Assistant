import { Settings2, Database, Zap, Bot, Layers, Search, Scissors, SlidersHorizontal } from 'lucide-react'
import { useChatStore, AVAILABLE_MODELS, AVAILABLE_SPLIT_METHODS } from '../store/chatStore'
import DocumentUpload from './DocumentUpload'

// ── 슬라이더 공통 컴포넌트 ────────────────────────────────────────────────────
function RangeSlider({ label, value, min, max, step, onChange, display, hint }) {
	const pct = max > min ? ((value - min) / (max - min)) * 100 : 100

	return (
		<div className="space-y-3">
			<div className="flex items-center justify-between">
				<label className="text-xs text-slate-500 font-medium">{label}</label>
				<span className="text-xs font-mono text-slate-700 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
					{display ?? value}
				</span>
			</div>
			<input
				type="range"
				min={min} max={max} step={step}
				value={value}
				onChange={(e) => onChange(e.target.value)}
				className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-violet-600"
				style={{
					background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${pct}%, #e2e8f0 ${pct}%, #e2e8f0 100%)`,
				}}
			/>
			{hint && <p className="text-[10px] text-slate-400">{hint}</p>}
		</div>
	)
}

// ── 토글 스위치 공통 컴포넌트 ─────────────────────────────────────────────────
function Toggle({ label, description, checked, onChange, icon }) {
	return (
		<div className={`flex items-center justify-between p-3 rounded-lg border transition-all duration-200 ${checked ? 'border-violet-200 bg-violet-50' : 'border-slate-200 bg-slate-50'}`}>
			<div className="space-y-0.5">
				<span className={`text-sm flex items-center gap-1 font-medium transition-colors ${checked ? 'text-violet-800' : 'text-slate-700'}`}>
					{icon}
					{label}
				</span>
				<span className={`text-[10px] transition-colors ${checked ? 'text-violet-600/80' : 'text-slate-500'}`}>
					{description}
				</span>
			</div>
			<label className="relative inline-flex items-center cursor-pointer flex-shrink-0">
				<input type="checkbox" className="sr-only peer" checked={checked} onChange={onChange} />
				<div className="w-9 h-5 bg-slate-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-violet-500" />
			</label>
		</div>
	)
}

// ── 세그먼트 버튼 공통 컴포넌트 ───────────────────────────────────────────────
function SegmentedControl({ options, value, onChange, cols = 3 }) {
	return (
		<div className={`grid gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200`} style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
			{options.map(({ id, shortLabel }) => (
				<button
					key={id}
					onClick={() => onChange(id)}
					className={`
						text-xs py-1.5 rounded-md font-medium transition-all focus:outline-none
						${value === id
							? 'bg-white text-violet-600 shadow-sm border border-slate-200/60'
							: 'text-slate-500 hover:text-slate-700 border border-transparent'}
					`}
				>
					{shortLabel}
				</button>
			))}
		</div>
	)
}

// ── 섹션 헤더 공통 컴포넌트 ───────────────────────────────────────────────────
function SectionHeader({ icon, title }) {
	return (
		<div className="flex items-center gap-2 text-slate-500">
			{icon}
			<h3 className="text-xs font-semibold uppercase tracking-wider">{title}</h3>
		</div>
	)
}

// ── Sidebar 본체 ─────────────────────────────────────────────────────────────
export default function Sidebar() {
	const { options, setOption } = useChatStore()

	// topK 변경 시 rerankTopK도 함께 클램핑
	const handleTopKChange = (val) => {
		const newTopK = parseInt(val)
		setOption('topK', newTopK)
		if (options.rerankTopK > newTopK) {
			setOption('rerankTopK', newTopK)
		}
	}

	// chunkSize 변경 시 chunkOverlap도 함께 클램핑
	const handleChunkSizeChange = (val) => {
		const newSize = parseInt(val)
		setOption('chunkSize', newSize)
		const newMaxOverlap = Math.floor(newSize / 2)
		if (options.chunkOverlap > newMaxOverlap) {
			setOption('chunkOverlap', newMaxOverlap)
		}
	}

	const chunkOverlapMax = Math.floor(options.chunkSize / 2)
	const rerankTopKPct = options.topK > 1
		? ((options.rerankTopK - 1) / (options.topK - 1)) * 100
		: 100

	return (
		<div className="flex flex-col h-full">
			<div className="p-5 flex items-center gap-2 border-b border-slate-200">
				<Settings2 size={18} className="text-violet-500" />
				<h2 className="font-semibold text-sm text-slate-800">Control Panel</h2>
			</div>

			<div className="flex-1 overflow-y-auto p-5 space-y-8">

				{/* ── 1. Knowledge Base ───────────────────────────────────────── */}
				<section className="space-y-3">
					<SectionHeader icon={<Database size={15} />} title="Knowledge Base" />
					<DocumentUpload />
				</section>

				<div className="h-px bg-slate-200" />

				{/* ── 2. Retrieval 검색 설정 ──────────────────────────────────── */}
				<section className="space-y-5">
					<SectionHeader icon={<Search size={15} />} title="Retrieval" />

					{/* LLM 모델 */}
					<div className="space-y-2">
						<label className="text-xs text-slate-500 font-medium">LLM 모델</label>
						<div className="relative">
							<Bot size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
							<select
								value={options.model}
								onChange={(e) => setOption('model', e.target.value)}
								className="w-full bg-white border border-slate-200 rounded-lg text-sm text-slate-700 py-2 pl-9 pr-3 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 appearance-none shadow-sm cursor-pointer transition-colors"
							>
								{AVAILABLE_MODELS.map((m) => (
									<option key={m.id} value={m.id}>{m.label}</option>
								))}
							</select>
						</div>
					</div>

					{/* 검색 방식 */}
					<div className="space-y-2">
						<label className="text-xs text-slate-500 font-medium">검색 방식</label>
						<SegmentedControl
							options={[
								{ id: 'bm25',   shortLabel: 'BM25'   },
								{ id: 'vector', shortLabel: 'Vector' },
								{ id: 'hybrid', shortLabel: 'Hybrid' },
							]}
							value={options.searchMethod}
							onChange={(v) => setOption('searchMethod', v)}
							cols={3}
						/>
					</div>

					{/* 검색 문서 수 */}
					<RangeSlider
						label="검색 문서 수"
						value={options.topK}
						min={1} max={20} step={1}
						onChange={handleTopKChange}
					/>

					{/* Reranker 토글 */}
					<Toggle
						label="Reranker 사용"
						description="CrossEncoder로 재정렬"
						checked={options.reranker}
						onChange={(e) => setOption('reranker', e.target.checked)}
						icon={<Layers size={13} className={options.reranker ? 'text-violet-600' : 'text-slate-400'} />}
					/>

					{/* 리랭크 문서 수 — reranker 활성화 시만 표시 */}
					{options.reranker && (
						<div className="pl-3 border-l-2 border-violet-200 space-y-3">
							<div className="flex items-center justify-between">
								<label className="text-xs text-slate-500 font-medium">리랭크 문서 수</label>
								<span className="text-xs font-mono text-slate-700 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
									{options.rerankTopK}
								</span>
							</div>
							<input
								type="range"
								min={1} max={options.topK} step={1}
								value={options.rerankTopK}
								onChange={(e) => setOption('rerankTopK', parseInt(e.target.value))}
								className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-violet-600"
								style={{
									background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${rerankTopKPct}%, #e2e8f0 ${rerankTopKPct}%, #e2e8f0 100%)`,
								}}
							/>
							<p className="text-[10px] text-slate-400">최대 {options.topK}개 (검색 문서 수 이하)</p>
						</div>
					)}

					{/* 시맨틱 캐시 */}
					<Toggle
						label="시맨틱 캐시"
						description="유사 질문 시 캐시 반환"
						checked={options.semanticCache}
						onChange={(e) => setOption('semanticCache', e.target.checked)}
						icon={<Zap size={13} className={options.semanticCache ? 'text-yellow-500 fill-yellow-500' : 'text-slate-400'} />}
					/>
				</section>

				<div className="h-px bg-slate-200" />

				{/* ── 3. Chunking 청킹 설정 ───────────────────────────────────── */}
				<section className="space-y-5">
					<SectionHeader icon={<Scissors size={15} />} title="Chunking" />

					{/* 문서 분할 방법 */}
					<div className="space-y-2">
						<label className="text-xs text-slate-500 font-medium">문서 분할 방법</label>
						<SegmentedControl
							options={AVAILABLE_SPLIT_METHODS}
							value={options.splitMethod}
							onChange={(v) => setOption('splitMethod', v)}
							cols={2}
						/>
					</div>

					{/* 청크 사이즈 */}
					<RangeSlider
						label="청크 사이즈"
						value={options.chunkSize}
						min={128} max={2048} step={128}
						onChange={handleChunkSizeChange}
					/>

					{/* 청크 오버랩 */}
					<RangeSlider
						label="청크 오버랩"
						value={Math.min(options.chunkOverlap, chunkOverlapMax)}
						min={0} max={chunkOverlapMax} step={16}
						onChange={(val) => setOption('chunkOverlap', parseInt(val))}
						hint={`최대 ${chunkOverlapMax} (청크 사이즈의 1/2)`}
					/>
				</section>

				<div className="h-px bg-slate-200" />

				{/* ── 4. Generation 생성 설정 ─────────────────────────────────── */}
				<section className="space-y-5">
					<SectionHeader icon={<SlidersHorizontal size={15} />} title="Generation" />

					{/* 온도 */}
					<div className="space-y-3">
						<RangeSlider
							label="온도 (Temperature)"
							value={options.temperature}
							min={0} max={2.0} step={0.1}
							onChange={(val) => setOption('temperature', parseFloat(val))}
							display={options.temperature.toFixed(1)}
						/>
						<div className="flex justify-between text-[10px] text-slate-400 -mt-1">
							<span>정확 (0.0)</span>
							<span>창의적 (2.0)</span>
						</div>
					</div>

					{/* 최대 토큰 수 */}
					<RangeSlider
						label="최대 토큰 수"
						value={options.maxTokens}
						min={256} max={4096} step={256}
						onChange={(val) => setOption('maxTokens', parseInt(val))}
						display={options.maxTokens.toLocaleString()}
					/>
				</section>

			</div>
		</div>
	)
}