export default function ThinkingIndicator() {
	return (
		<div className="flex items-center gap-3">
			<div className="flex items-center gap-1.5">
				{[0, 1, 2].map((i) => (
					<span
						key={i}
						className="block w-1.5 h-1.5 rounded-full bg-violet-500 animate-dot-bounce"
						style={{ animationDelay: `${i * 0.18}s` }}
					/>
				))}
			</div>
			<span className="text-xs font-medium text-slate-400 tracking-wide whitespace-nowrap">
				생각 중...
			</span>
		</div>
	)
}