import { useCallback } from 'react'
import toast from 'react-hot-toast'
import { chatApi } from '../api/api'
import { useChatStore } from '../store/chatStore'

export function useChat() {
	const {
		options,
		addMessage,
		updateLastMessage,
		setLoading,
		setAbortController,
		abortRequest,
	} = useChatStore()

	const sendMessage = useCallback(
		async (content) => {
			if (!content.trim()) return

			const startTime = Date.now()
			const currentOptions = { ...options }

			addMessage({ 
				role: 'user', 
				content,
				timestamp: new Date().toISOString() 
			})

			setLoading(true)

			const payload = { message: content, options }

			if (options.streamResponse) {
				const controller = new AbortController()
				setAbortController(controller)

				addMessage({
					role: 'assistant',
					content: '',
					streaming: true,
					metadata: { options: currentOptions }, 
					timestamp: new Date().toISOString()
				})

				try {
					await chatApi.stream(
						payload,
						(chunk) => {
							updateLastMessage((msg) => ({ content: msg.content + chunk }))
						},
						(meta) => {
							updateLastMessage({
								streaming: false,
								metadata: { ...meta, responseTime: Date.now() - startTime },
							})
						},
						controller.signal,
					)
					updateLastMessage((msg) =>
						msg.streaming ? { streaming: false } : {},
					)
				} catch (err) {
					if (err.name === 'AbortError') {
						updateLastMessage({ streaming: false })
						toast('응답이 중단되었습니다', { icon: '⏹' })
					} else {
						toast.error(err.message || '응답 생성 중 오류가 발생했습니다')
						updateLastMessage({
							streaming: false,
							error: true,
							content: '응답을 받지 못했습니다. 다시 시도해 주세요.',
						})
					}
				} finally {
					setLoading(false)
					setAbortController(null)
				}

			// ── Non-streaming mode ────────────────────────────────────
			} else {
				try {
					const data = await chatApi.send(payload)
					addMessage({
						role: 'assistant',
						content: data.answer ?? '',
						metadata: {
							options: currentOptions,
							model: data.model,
							searchMethod: data.search_method,
							tokensUsed: data.tokens_used,
							cached: data.cached,
							sources: data.sources,
							responseTime: Date.now() - startTime,
						},
					})
				} catch (err) {
					toast.error(err.message || '오류가 발생했습니다')
					addMessage({
						role: 'assistant',
						content: '응답을 받지 못했습니다. 다시 시도해 주세요.',
						error: true,
					})
				} finally {
					setLoading(false)
				}
			}
		},
		[options, addMessage, updateLastMessage, setLoading, setAbortController],
	)

	return { sendMessage, abortRequest }
}