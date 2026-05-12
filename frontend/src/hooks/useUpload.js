import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { documentsApi } from '../api/api'
import { useChatStore } from '../store/chatStore'

export function useDocuments() {
	const { setDocuments, addDocument, removeDocument, updateDocument } = useChatStore()
	const queryClient = useQueryClient()

	// 문서 목록 불러오기
	const { data, isLoading: isFetching } = useQuery({
		queryKey: ['documents'],
		queryFn: async () => {
			try {
				return await documentsApi.list()
			} catch (err) {
				console.warn("백엔드 미연결 상태: 문서 목록을 불러오지 못했습니다.")
				return []
			}
		},
		retry: false,
	})

	// 상태 동기화
	useEffect(() => {
		if (Array.isArray(data)) {
			const currentDocs = useChatStore.getState().documents
			const mockDocs = currentDocs.filter((doc) => doc.isMock || doc.status === 'uploading')
			const newDocs = data.filter((d) => !mockDocs.find((md) => md.id === d.id))
			
			setDocuments([...newDocs, ...mockDocs])
		}
	}, [data, setDocuments])

	// 문서 업로드
	const uploadMutation = useMutation({
		mutationFn: async ({ file, onProgress, tempId }) => {
			try {
				return await documentsApi.upload(file, onProgress)
			} catch (error) {
				let progress = 0
				while (progress < 100) {
					progress += 20
					if (onProgress) onProgress(progress)
					await new Promise((r) => setTimeout(r, 150))
				}
				return { id: tempId, name: file.name, size: file.size, status: 'ready', isMock: true }
			}
		},
		onSuccess: (data, { tempId }) => {
			removeDocument(tempId)
			addDocument({ ...data, status: 'ready' })
			queryClient.invalidateQueries({ queryKey: ['documents'] })
			
			if (data.isMock) {
				toast.success(`"${data.name}" 임시 첨부 완료 (백엔드 미연결)`)
			} else {
				toast.success(`"${data.name}" 업로드 완료`)
			}
		},
		onError: (err, { tempId }) => {
			updateDocument(tempId, { status: 'error' })
			toast.error(err.message || '업로드 실패')
		},
	})

	const uploadFile = (file) => {
		const tempId = crypto.randomUUID()
		addDocument({
			id: tempId,
			name: file.name,
			size: file.size,
			status: 'uploading',
			progress: 0,
		})

		uploadMutation.mutate({
			file,
			tempId,
			onProgress: (pct) => updateDocument(tempId, { progress: pct }),
		})
	}

	// 문서 삭제
	const deleteMutation = useMutation({
		mutationFn: async (id) => {
			try {
				return await documentsApi.delete(id)
			} catch (error) {
				return { status: 'success', id }
			}
		},
		onSuccess: (_, id) => {
			removeDocument(id)
			queryClient.invalidateQueries({ queryKey: ['documents'] })
		},
	})

	return {
		isFetching,
		uploadFile,
		deleteMutation,
	}
}