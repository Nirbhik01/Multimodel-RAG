import { useState, useRef, useEffect } from "react"

export default function HomePage() {
    const API = import.meta.env.VITE_API_BASE_URL
    const [messages, setMessages] = useState([])
    const [inputValue, setInputValue] = useState("")
    const [selectedFile, setSelectedFile] = useState(null)
    const [previewUrl, setPreviewUrl] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)
    
    const messagesEndRef = useRef(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages, isLoading])

    const handleFileChange = (e) => {
        const file = e.target.files[0]
        if (file) {
            setSelectedFile(file)
            setPreviewUrl(URL.createObjectURL(file))
        }
    }

    const removeFile = () => {
        setSelectedFile(null)
        setPreviewUrl(null)
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (isLoading || (!inputValue.trim() && !selectedFile)) return

        const userMessage = {
            id: Date.now(),
            role: 'user',
            content: inputValue,
            image: previewUrl
        }

        setMessages(prev => [...prev, userMessage])
        const currentInput = inputValue
        const currentFile = selectedFile
        
        setInputValue("")
        setSelectedFile(null)
        setPreviewUrl(null)
        setIsLoading(true)
        setError(null)

        const formData = new FormData()
        if (currentFile) formData.append('image', currentFile)
        formData.append('query', currentInput)

        try {
            const response = await fetch(`${API}/getResponse`, {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
            }

            const data = await response.json()
            
            const assistantMessage = {
                id: Date.now() + 1,
                role: 'assistant',
                content: data.text.replace("Impression:", "\n\nImpression:"),
                similarityImage: data.image
            }

            setMessages(prev => [...prev, assistantMessage])
        } catch (err) {
            console.error("Error during submission:", err)
            setError(err.message || "Failed to connect to the backend server.")
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="flex flex-col h-screen bg-[#0f172a] text-slate-100 font-sans overflow-hidden">
            {/* Header */}
            <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md z-10">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white shadow-[0_0_15px_rgba(37,99,235,0.4)]">M</div>
                    <h1 className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-300">
                        Multimodal RAG Explorer
                    </h1>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors cursor-pointer">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                        System Active
                    </div>
                </div>
            </header>

            {/* Chat Area */}
            <main className="flex-1 overflow-y-auto px-4 py-8 space-y-10 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                <div className="max-w-4xl mx-auto space-y-8">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center pt-20 text-center space-y-6">
                            <div className="p-4 rounded-3xl bg-blue-600/10 border border-blue-500/20 shadow-inner">
                                <svg className="w-12 h-12 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                            </div>
                            <h2 className="text-2xl font-bold text-white">How can I assist your diagnosis today?</h2>
                            <p className="text-slate-400 max-w-sm">
                                Upload a chest X-ray image and provide your clinical notes to find similar cases and medical insights.
                            </p>
                        </div>
                    )}

                    {messages.map((msg) => (
                        <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                            <div className={`max-w-[85%] md:max-w-[75%] flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                                {/* Avatar */}
                                <div className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center border ${
                                    msg.role === 'user' 
                                    ? 'bg-indigo-600 border-indigo-500 shadow-lg shadow-indigo-900/20' 
                                    : 'bg-slate-800 border-slate-700 shadow-md'
                                }`}>
                                    {msg.role === 'user' ? (
                                        <svg className="w-5 h-5 text-indigo-100" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                        </svg>
                                    ) : (
                                        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                                        </svg>
                                    )}
                                </div>

                                {/* Content */}
                                <div className={`flex flex-col space-y-3 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                    <div className={`p-4 rounded-2xl shadow-sm ${
                                        msg.role === 'user' 
                                        ? 'bg-indigo-600 text-white rounded-tr-none' 
                                        : 'bg-slate-800/80 border border-slate-700 text-slate-200 rounded-tl-none'
                                    }`}>
                                        {msg.image && (
                                            <div className="mb-3 overflow-hidden rounded-lg border border-white/10 shadow-lg">
                                                <img src={msg.image} alt="uploaded xray" className="max-w-full max-h-60 object-cover" />
                                            </div>
                                        )}
                                        {msg.similarityImage && (
                                            <div className="mb-4">
                                                <p className="text-[10px] uppercase font-bold tracking-wider text-blue-400 mb-2 drop-shadow-sm">Matched Reference Case</p>
                                                <div className="overflow-hidden rounded-lg border-2 border-slate-700 shadow-xl bg-black">
                                                    <img src={msg.similarityImage} alt="most similar xray" className="max-w-full max-h-80 object-contain mx-auto" />
                                                </div>
                                            </div>
                                        )}
                                        <p className="whitespace-pre-wrap leading-relaxed text-[15px]">{msg.content || (msg.similarityImage ? "" : "Processing clinical data...")}</p>
                                    </div>
                                    <span className="text-[10px] font-medium text-slate-500 uppercase tracking-widest px-1">
                                        {msg.role === 'user' ? 'Patient Case' : 'Clinical Analysis'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex justify-start animate-pulse">
                            <div className="flex gap-4 max-w-[75%] items-start">
                                <div className="flex-shrink-0 w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                                    <svg className="w-5 h-5 text-blue-400 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                </div>
                                <div className="p-4 rounded-2xl rounded-tl-none bg-slate-800/80 border border-slate-700 text-slate-400 italic text-sm">
                                    Performing multimodal search and analysis...
                                </div>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="flex justify-center">
                            <div className="px-4 py-2 bg-red-500/10 border border-red-500/30 text-red-500 rounded-lg text-sm flex items-center gap-2">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span>Error: {error}</span>
                            </div>
                        </div>
                    )}
                    
                    <div ref={messagesEndRef} />
                </div>
            </main>

            {/* Input Bar Section */}
            <section className="px-4 pb-8 pt-4 bg-gradient-to-t from-[#0f172a] via-[#0f172a]/90 to-transparent">
                <div className="max-w-4xl mx-auto">
                    {/* Image Preview Thumb */}
                    {previewUrl && (
                        <div className="mb-3 px-1">
                            <div className="relative inline-block group">
                                <img src={previewUrl} className="w-20 h-20 object-cover rounded-xl border-2 border-indigo-500 ring-4 ring-indigo-500/10 shadow-lg" alt="Preview" />
                                <button 
                                    onClick={removeFile}
                                    className="absolute -top-2 -right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-1 shadow-md transition-all active:scale-90"
                                >
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="relative group">
                        <div className="flex items-end gap-3 p-3 rounded-[24px] bg-slate-800/80 border border-slate-700 focus-within:border-blue-500/50 focus-within:ring-4 focus-within:ring-blue-500/10 transition-all duration-300 shadow-[0_10px_30px_rgba(0,0,0,0.3)] backdrop-blur-md">
                            {/* File Upload Button */}
                            <label className="flex-shrink-0 cursor-pointer p-2.5 rounded-full hover:bg-slate-700 transition-colors group/btn relative overflow-hidden">
                                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} disabled={isLoading} />
                                <svg className="w-5 h-5 text-slate-400 group-hover/btn:text-blue-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                <div className="absolute inset-0 bg-blue-500/0 hover:bg-blue-500/5 transition-all"></div>
                            </label>

                            {/* Text Input */}
                            <textarea
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault()
                                        handleSubmit(e)
                                    }
                                }}
                                placeholder="Describe current findings or type a query..."
                                className="flex-1 max-h-[150px] min-h-[44px] py-2 px-1 bg-transparent border-none focus:ring-0 text-slate-200 placeholder-slate-500 resize-none text-[15px] leading-relaxed scrollbar-none"
                                disabled={isLoading}
                                rows={1}
                            />

                            {/* Submit Button */}
                            <button 
                                type="submit"
                                disabled={isLoading || (!inputValue.trim() && !selectedFile)}
                                className={`flex-shrink-0 p-2.5 rounded-full transition-all duration-300 ${
                                    isLoading || (!inputValue.trim() && !selectedFile)
                                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                    : 'bg-blue-600 text-white hover:bg-blue-500 shadow-[0_0_15px_rgba(37,99,235,0.4)] active:scale-95'
                                }`}
                            >
                                <svg className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    {isLoading ? (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    ) : (
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 12h14M12 5l7 7-7 7" />
                                    )}
                                </svg>
                            </button>
                        </div>
                        <p className="text-[11px] text-center mt-3 text-slate-600 font-medium tracking-wide first-letter:uppercase">
                            RAG Analysis powered by Gemini & Vector Search
                        </p>
                    </form>
                </div>
            </section>
        </div>
    )
}