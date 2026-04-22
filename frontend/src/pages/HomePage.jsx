import { useState } from "react"

export default function HomePage(){
    const API = import.meta.env.VITE_API_BASE_URL
    const [responseText,setResponseText] = useState("")
    const [responseImage, setResponseImage] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (isLoading) return

        setIsLoading(true)
        setError(null)
        
        const formData = new FormData()
        const imageFile = e.target.image.files[0]
        const queryText = e.target.query.value
        
        console.log("Submitting form with:", { query: queryText, image: imageFile })

        formData.append('image', imageFile)
        formData.append('query', queryText)

        try {
            const response = await fetch(`${API}/getResponse`, {
                method: 'POST',
                body: formData,
            })
            
            if (!response.ok) {
                throw new Error(`Server respondent with ${response.status}: ${response.statusText}`)
            }

            const data = await response.json()
            console.log("Received response:", data)
            setResponseText(data.text.replace("Impression:", "\n\nImpression:"))
            setResponseImage(data.image)
            
            // Empty the form once submitted successfully
            e.target.image.value = ""
            e.target.query.value = ""
        } catch (err) {
            console.error("Error during submission:", err)
            setError(err.message || "Failed to connect to the backend server.")
        } finally {
            setIsLoading(false)
        }
    }
    return (
        <div className="h-[100vh] flex flex-col items-center justify-center border border-red-500 space-y-10 bg-slate-900 text-white">
            <h1 className="text-4xl font-bold mb-5">Multimodal RAG Explorer</h1>
            <form className="flex flex-col items-center justify-center space-y-4 w-full max-w-md" onSubmit={handleSubmit}>
                <div className="flex flex-col w-full">
                    <label className="text-sm font-medium mb-1">X-Ray Image</label>
                    <input className="border border-slate-700 rounded-xl p-2 bg-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500" type="file" name="image" />
                </div>
                <div className="flex flex-col w-full">
                    <label className="text-sm font-medium mb-1">Query Text</label>
                    <input className="border border-slate-700 rounded-xl p-2 bg-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 text-white" type="text" name="query" placeholder="Enter your query..." />
                </div>
                <button 
                    type="submit" 
                    disabled={isLoading}
                    className={`border border-blue-500 bg-blue-600 hover:bg-blue-700 text-white rounded-xl p-3 w-full transition-all duration-200 font-semibold shadow-lg ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    {isLoading ? 'Analyzing...' : 'Analyze Image & Query'}
                </button>
            </form>

            {error && (
                <div className="bg-red-500/10 border border-red-500 text-red-500 p-4 rounded-xl text-center max-w-md animate-pulse">
                    <strong>Error:</strong> {error}
                </div>
            )}

            <div className="border border-slate-700 p-6 rounded-2xl bg-slate-800 shadow-2xl w-full max-w-4xl h-fit">
                <h2 className="text-xl font-semibold mb-4 text-slate-300">Analysis Result</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex flex-col items-center justify-center border border-dashed border-slate-600 rounded-xl p-2 min-h-[200px]">
                        {/* Image insertion */}
                        <p className="text-slate-500 text-sm mb-2">Most Similar X-Ray</p>
                        {responseImage ? (
                            <img src={responseImage} alt="most similar xray" className="max-h-[180px] rounded-lg" />
                        ) : (
                            <div className="w-full h-[180px] flex items-center justify-center bg-slate-800/50 rounded-lg text-slate-600 text-xs text-center px-4">
                                Most similar image will appear here
                            </div>
                        )}
                    </div>
                    <textarea 
                        name='result' 
                        className="border border-slate-700 p-4 rounded-xl resize-none bg-slate-900 text-white focus:outline-none h-[200px]" 
                        placeholder="Responses will appear here..." 
                        rows={10} 
                        value={responseText}
                        readOnly
                    />
                </div>
            </div>
        </div>
    )
}