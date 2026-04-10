import { useState } from "react"

export default function HomePage(){
    const API = import.meta.env.VITE_API_BASE_URL
    const [responseText,setResponseText] = useState("")
    const handleSubmit = (e) => {
        e.preventDefault()
        const formData = new FormData()
        formData.append('image', e.target.image.files[0])
        formData.append('query', e.target.query.value)

        fetch(`${API}/getResponse`, {
            method: 'POST',
            body: formData,
        })
        .then(
            (res) => res.json()
        )
        .then(
            (data) => {
                setResponseText(data.text)
            }
        )
        .catch(
            (err) => console.log(err)
        )
    }
    return (
        <div className="h-[100vh] flex flex-col items-center justify-center border border-red-500 space-y-10">
            <form className="flex flex-col items-center justify-center space-y-4" onSubmit={handleSubmit}>
                <input className="border rounded-xl p-2" type="file" name="image" id="" />
                <input className="border rounded-xl p-2" type="text" name="query" id="" />
                <button type="submit" className="border rounded-xl p-2">Submit</button>
            </form>
            <div className="border p-2 rounded-xl">
                {/* Image insertion */}
                <img src={null} alt="most similar xray" />
                <textarea className="border p-2 rounded-xl resize-none" placeholder="Responses Appear here" rows={5} cols={50} defaultValue={responseText}>
                </textarea>
            </div>
        </div>
    )
}