import { useEffect, useRef } from 'react'
import { PanelLeftClose, PanelLeft, MessageSquare } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { useChat } from '../hooks/useChat'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import Sidebar from '../components/Sidebar'

export default function ChatPage() {
	const { messages, sidebarOpen, toggleSidebar } = useChatStore()
	const { sendMessage, abortRequest } = useChat()
	const scrollRef = useRef(null)

	useEffect(() => {
		if (scrollRef.current) {
			scrollRef.current.scrollTop = scrollRef.current.scrollHeight
		}
	}, [messages])

	return (
		<div className="flex h-screen w-full bg-slate-50 text-slate-900 overflow-hidden relative">
			
			<div
				className={`
					absolute z-20 md:relative flex-shrink-0 h-full transition-all duration-300 ease-in-out
					${sidebarOpen ? 'w-80 translate-x-0' : 'w-80 -translate-x-full md:w-0 md:translate-x-0'}
				`}
			>
				<div className={`w-80 h-full border-r border-slate-200 bg-white/90 glass ${sidebarOpen ? 'opacity-100' : 'opacity-0 md:hidden'}`}>
					<Sidebar />
				</div>
			</div>

			{sidebarOpen && (
				<div 
					className="absolute inset-0 z-10 bg-slate-900/20 md:hidden backdrop-blur-sm"
					onClick={toggleSidebar}
				/>
			)}

			<div className="flex flex-col flex-1 h-full min-w-0 relative">
				<header className="flex flex-shrink-0 items-center justify-between px-4 h-14 border-b border-slate-200 bg-white/50 glass z-10">
					<div className="flex items-center gap-3">
						<button
							onClick={toggleSidebar}
							className="p-2 -ml-2 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
						>
							{sidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeft size={20} />}
						</button>
						<div className="flex items-center gap-2">
							<div className="w-6 h-6 rounded-md bg-gradient-to-tr from-violet-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
								<MessageSquare size={14} className="text-white" />
							</div>
							<h1 className="font-semibold text-sm text-slate-800 tracking-wide">AI Assistant</h1>
						</div>
					</div>
				</header>

				<div ref={scrollRef} className="flex-1 overflow-y-auto px-4 md:px-8 py-6 scroll-smooth">
					<div className="max-w-3xl mx-auto space-y-6">
						{messages.length === 0 ? (
							<div className="flex flex-col items-center justify-center h-full text-center space-y-4 mt-20 opacity-80">
								<div className="w-16 h-16 rounded-2xl bg-white flex items-center justify-center border border-slate-200 shadow-sm">
									<MessageSquare size={28} className="text-violet-500" />
								</div>
								<div className="space-y-1">
									<h2 className="text-lg font-medium text-slate-700">무엇을 도와드릴까요?</h2>
									<p className="text-sm text-slate-500">문서를 업로드하고 관련된 내용을 질문해보세요.</p>
								</div>
							</div>
						) : (
							messages.map((msg) => (
								<ChatMessage key={msg.id} message={msg} />
							))
						)}
					</div>
				</div>

				<div className="flex-shrink-0 p-4 md:px-8 pb-6 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent">
					<div className="max-w-3xl mx-auto">
						<ChatInput onSend={sendMessage} onAbort={abortRequest} />
					</div>
				</div>
			</div>
		</div>
	)
}