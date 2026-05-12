import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { UploadCloud, File, X, Loader2, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import { useDocuments } from '../hooks/useUpload'
import { useChatStore } from '../store/chatStore'

export default function DocumentUpload() {
	const { uploadFile, deleteMutation } = useDocuments()
	const { documents } = useChatStore()

	const onDrop = useCallback((acceptedFiles, fileRejections) => {
		acceptedFiles.forEach((file) => {
			uploadFile(file)
		})

		// 용량 초과 등 거절된 파일 안내 토스트
		fileRejections.forEach(({ file, errors }) => {
			const isTooLarge = errors.find(e => e.code === 'file-too-large')
			if (isTooLarge) {
				toast.error(`"${file.name}" 파일이 너무 큽니다. (최대 100MB)`)
			} else {
				toast.error(`"${file.name}" 파일을 업로드할 수 없습니다.`)
			}
		})
	}, [uploadFile])

	const { getRootProps, getInputProps, isDragActive } = useDropzone({
		onDrop,
		accept: {
			'application/pdf': ['.pdf'],
			'text/plain': ['.txt'],
			'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
		},
		maxSize: 100 * 1024 * 1024, 
	})

	return (
		<div className="space-y-3">
			{/* Dropzone */}
			<div
				{...getRootProps()}
				className={`
					p-4 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all duration-200 ease-in-out
					${isDragActive 
						? 'border-violet-500 bg-violet-50 text-violet-600' 
						: 'border-slate-300 bg-slate-50 text-slate-500 hover:border-slate-400 hover:bg-slate-100'
					}
				`}
			>
				<input {...getInputProps()} />
				<UploadCloud size={20} className="mx-auto mb-2 opacity-70" />
				<p className="text-xs font-medium">클릭하거나 파일을 드롭하세요</p>
				<p className="text-[10px] opacity-70 mt-1">PDF, TXT, DOCX (Max 100MB)</p>
			</div>

			{/* 업로드 문서 리스트 */}
			{documents.length > 0 && (
				<div className="space-y-2 mt-4">
					{documents.map((doc) => (
						<div 
							key={doc.id} 
							className={`flex items-center justify-between p-2 rounded-lg border shadow-sm group ${doc.status === 'error' ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'}`}
						>
							<div className="flex items-center gap-2 overflow-hidden w-full pr-2">
								{doc.status === 'error' ? (
									<AlertCircle size={14} className="text-red-400 flex-shrink-0" />
								) : (
									<File size={14} className="text-slate-400 flex-shrink-0" />
								)}
								
								<div className="flex flex-col overflow-hidden w-full">
									<span className={`text-xs truncate ${doc.status === 'error' ? 'text-red-600 font-medium' : 'text-slate-700'}`}>
										{doc.name}
									</span>
									
									{/* 업로드 프로그레스 바 */}
									{doc.status === 'uploading' && (
										<div className="w-full bg-slate-200 h-1 mt-1.5 rounded-full overflow-hidden">
											<div 
												className="bg-violet-500 h-full transition-all duration-200 ease-linear" 
												style={{ width: `${doc.progress || 0}%` }}
											/>
										</div>
									)}
								</div>
							</div>
							
							{doc.status === 'uploading' ? (
								<Loader2 size={13} className="animate-spin text-slate-400 flex-shrink-0 mr-1" />
							) : (
								<button
									onClick={() => deleteMutation.mutate(doc.id)}
									className="p-1 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors opacity-0 group-hover:opacity-100 flex-shrink-0"
									title="삭제"
								>
									<X size={13} />
								</button>
							)}
						</div>
					))}
				</div>
			)}
		</div>
	)
}