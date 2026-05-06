import { useState, useEffect } from "react"

export default function Sidebar({ conversations, activeId, onSelect, onNewChat, onDelete }) {
    return (
        <aside className="w-64 md:w-72 h-full bg-[#0b1222] border-r border-slate-800 flex flex-col z-20 transition-all duration-300">
            {/* New Chat Button */}
            <div className="p-4">
                <button 
                    onClick={onNewChat}
                    className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 text-slate-200 transition-all duration-200 group active:scale-95"
                >
                    <div className="w-6 h-6 rounded-lg bg-blue-600/20 flex items-center justify-center text-blue-400 group-hover:bg-blue-600 group-hover:text-white transition-all">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 4v16m8-8H4" />
                        </svg>
                    </div>
                    <span className="font-medium text-sm">New Analysis</span>
                </button>
            </div>

            {/* Conversation List */}
            <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
                <div className="px-3 mb-2">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Recent Sessions</span>
                </div>
                
                {conversations.length === 0 ? (
                    <div className="px-4 py-8 text-center">
                        <p className="text-xs text-slate-500 italic">No recent analyses</p>
                    </div>
                ) : (
                    conversations.map((conv) => (
                        <div 
                            key={conv._id}
                            onClick={() => onSelect(conv._id)}
                            className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-200 ${
                                activeId === conv._id 
                                ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20 shadow-inner' 
                                : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                            }`}
                        >
                            <svg className={`w-4 h-4 flex-shrink-0 ${activeId === conv._id ? 'text-blue-400' : 'text-slate-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                            </svg>
                            <span className="flex-1 text-sm font-medium truncate pr-6">{conv.title}</span>
                            
                            {/* Delete Action */}
                            <button 
                                onClick={(e) => {
                                    e.stopPropagation()
                                    onDelete(conv._id)
                                }}
                                className="absolute right-2 opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
                            >
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* Bottom Section */}
            <div className="p-4 border-t border-slate-800 bg-[#0b1222]/80 backdrop-blur-sm">
                <div className="flex items-center gap-3 px-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold border border-white/10">
                        DR
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-slate-200 truncate">Clinical User</p>
                        <p className="text-[10px] text-slate-500 truncate">Free Plan</p>
                    </div>
                    <button className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-500 hover:text-slate-300">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37a1.724 1.724 0 002.572-1.065z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                    </button>
                </div>
            </div>
        </aside>
    )
}
