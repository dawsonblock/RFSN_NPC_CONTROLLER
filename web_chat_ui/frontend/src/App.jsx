import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Send, RefreshCcw, Download, Copy, MessageSquare, AlertCircle } from 'lucide-react';
import './App.css';

const BACKEND_URL = 'http://localhost:3001';

export default function App() {
    const [sessionId, setSessionId] = useState('');
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [npcId, setNpcId] = useState('npc_001');
    const [npcName, setNpcName] = useState('Uma');
    const [npcState, setNpcState] = useState(JSON.stringify({
        mood: "neutral",
        relationship: "stranger",
        affinity: 0.1,
        combat_active: false
    }, null, 2));
    const [enableVoice, setEnableVoice] = useState(true);

    const [metaData, setMetaData] = useState(null);
    const [status, setStatus] = useState('connected'); // connected, streaming, error
    const [errorMessage, setErrorMessage] = useState('');

    const messagesEndRef = useRef(null);
    const textareaRef = useRef(null);

    // Initialize Session
    useEffect(() => {
        startNewSession();
    }, []);

    // Auto-scroll
    useEffect(() => {
        if (status === 'streaming') {
            messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, status]);

    const startNewSession = () => {
        const newId = uuidv4();
        setSessionId(newId);
        setMessages([]);
        setMetaData(null);
        setStatus('connected');
        setErrorMessage('');
        console.log("New Session:", newId);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleSend = async () => {
        if (!inputText.trim() || status === 'streaming') return;

        const userText = inputText.trim();
        setInputText('');

        // Add User Message
        const userMsg = { id: uuidv4(), role: 'user', text: userText };
        setMessages(prev => [...prev, userMsg]);

        // Add Placeholder Assistant Message
        const assistantId = uuidv4();
        const assistantMsg = { id: assistantId, role: 'assistant', text: '' };
        setMessages(prev => [...prev, assistantMsg]);

        setStatus('streaming');
        setErrorMessage('');

        try {
            let parsedState = {};
            try {
                parsedState = JSON.parse(npcState);
            } catch (e) {
                setErrorMessage("Invalid JSON in NPC State");
                setStatus('error');
                return;
            }

            const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    npc_id: npcId,
                    npc_name: npcName,
                    player_text: userText,
                    npc_state: parsedState,
                    session_id: sessionId,
                    enable_voice: enableVoice
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process buffer for SSE lines
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || ''; // Keep partial line in buffer

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        const parts = line.split('\n');
                        const eventType = parts[0].replace('event: ', '').trim();
                        const dataLine = parts.find(p => p.startsWith('data: '));

                        if (dataLine) {
                            const dataStr = dataLine.replace('data: ', '');
                            try {
                                const data = JSON.parse(dataStr);

                                if (eventType === 'meta') {
                                    setMetaData(data);
                                } else if (eventType === 'text') {
                                    setMessages(prev => prev.map(msg =>
                                        msg.id === assistantId
                                            ? { ...msg, text: msg.text + data.text }
                                            : msg
                                    ));
                                } else if (eventType === 'error') {
                                    setErrorMessage(data.message);
                                    setStatus('error');
                                }
                            } catch (e) {
                                console.error("Parse error", e);
                            }
                        }

                        if (eventType === 'done' || eventType === 'error') {
                            setStatus(eventType === 'error' ? 'error' : 'connected');
                        }
                    }
                }
            }

            setStatus('connected');

        } catch (err) {
            console.error(err);
            setErrorMessage(err.message);
            setStatus('error');
        }
    };

    const copyEpisode = async () => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/episodes/${sessionId}.jsonl`);
            if (response.ok) {
                const text = await response.text();
                await navigator.clipboard.writeText(text);
                alert("Episode JSONL copied to clipboard!");
            } else {
                alert("Episode log not found (maybe empty?).");
            }
        } catch (e) {
            alert("Failed to copy episode.");
        }
    };

    const downloadEpisode = () => {
        window.open(`${BACKEND_URL}/api/episodes/${sessionId}.jsonl`, '_blank');
    };


    // Memory Editor State
    const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'memory'
    const [memoryList, setMemoryList] = useState([]);
    const [selectedMemory, setSelectedMemory] = useState(null);
    const [editorContent, setEditorContent] = useState('');
    const [editorStatus, setEditorStatus] = useState('');

    // Fetch memory list when tab is active
    useEffect(() => {
        if (activeTab === 'memory') {
            fetchMemories();
        }
    }, [activeTab]);

    const fetchMemories = async () => {
        try {
            const res = await fetch(`${BACKEND_URL}/api/memories`);
            if (res.ok) {
                const list = await res.json();
                setMemoryList(list);
            }
        } catch (e) {
            console.error("Failed to fetch memories", e);
        }
    };

    const loadMemory = async (filename) => {
        try {
            const res = await fetch(`${BACKEND_URL}/api/memories/${filename}`);
            if (res.ok) {
                const data = await res.json();
                setSelectedMemory(filename);
                setEditorContent(JSON.stringify(data, null, 2));
                setEditorStatus(`Loaded ${filename}`);
            }
        } catch (e) {
            setEditorStatus("Failed to load file");
        }
    };

    const saveMemory = async () => {
        if (!selectedMemory) return;
        try {
            // Validate JSON
            let jsonContent;
            try {
                jsonContent = JSON.parse(editorContent);
            } catch (e) {
                setEditorStatus("Error: Invalid JSON");
                return;
            }

            const res = await fetch(`${BACKEND_URL}/api/memories/${selectedMemory}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonContent)
            });

            if (res.ok) {
                setEditorStatus(`Saved ${selectedMemory} at ${new Date().toLocaleTimeString()}`);
            } else {
                setEditorStatus("Failed to save");
            }
        } catch (e) {
            setEditorStatus("Error saving content");
        }
    };

    const createNewMemory = async () => {
        const name = prompt("Enter NPC Name (e.g., 'Guard_Whiterun'):");
        if (!name) return;

        const filename = name.endsWith('.json') ? name : `${name}.json`;

        try {
            const res = await fetch(`${BACKEND_URL}/api/memories`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, content: [] })
            });

            if (res.ok) {
                await fetchMemories();
                await loadMemory(filename);
                setEditorStatus(`Created ${filename}`);
            } else {
                alert("Failed to create file (maybe exists?)");
            }
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="layout-container">
            {/* Top Navigation Bar */}
            <div className="top-nav">
                <div className="logo">
                    <MessageSquare size={20} className="text-accent" />
                    <span>RFSN Console</span>
                </div>
                <div className="tabs">
                    <button
                        className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
                        onClick={() => setActiveTab('chat')}
                    >
                        Chat
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'memory' ? 'active' : ''}`}
                        onClick={() => setActiveTab('memory')}
                    >
                        Memory Editor
                    </button>
                </div>
            </div>

            {/* Content Area */}
            <div className="main-content">

                {/* ---------------- CHAT TAB ---------------- */}
                {activeTab === 'chat' && (
                    <div className="layout">
                        {/* Sidebar */}
                        <div className="sidebar">
                            <div className="control-group">
                                <label>NPC Identity</label>
                                <input
                                    value={npcName}
                                    onChange={e => setNpcName(e.target.value)}
                                    placeholder="NPC Name"
                                />
                                <input
                                    value={npcId}
                                    onChange={e => setNpcId(e.target.value)}
                                    placeholder="NPC ID"
                                    style={{ fontSize: '0.8rem' }}
                                />
                            </div>

                            <div className="control-group" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                <label>NPC State (JSON)</label>
                                <textarea
                                    className="npc-state-input"
                                    value={npcState}
                                    onChange={e => setNpcState(e.target.value)}
                                    style={{ flex: 1, fontFamily: 'monospace', fontSize: '0.75rem' }}
                                />
                            </div>

                            <div className="control-group">
                                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                                    <input
                                        type="checkbox"
                                        checked={enableVoice}
                                        onChange={e => setEnableVoice(e.target.checked)}
                                    />
                                    Enable TTS (Speech)
                                </label>
                            </div>

                            <div className="control-group">
                                <label>Latest Meta</label>
                                <div className="meta-panel">
                                    {metaData ? JSON.stringify(metaData, null, 2) : "// Waiting for turn..."}
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto' }}>
                                <button className="btn" onClick={copyEpisode} title="Copy Episode JSONL">
                                    <Copy size={16} /> Copy
                                </button>
                                <button className="btn" onClick={downloadEpisode} title="Download JSONL">
                                    <Download size={16} /> DL
                                </button>
                                <button className="btn btn-danger" onClick={startNewSession} title="Reset">
                                    <RefreshCcw size={16} />
                                </button>
                            </div>
                        </div>

                        {/* Main Chat */}
                        <div className="chat-area">
                            <div className="header">
                                <div className="status-indicator">
                                    <div className={`dot ${status}`} />
                                    {status}
                                </div>
                                {errorMessage && (
                                    <div style={{ color: 'var(--danger)', fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <AlertCircle size={16} />
                                        {errorMessage}
                                    </div>
                                )}
                            </div>

                            <div className="messages">
                                {messages.map((msg, idx) => (
                                    <div key={msg.id} className={`message ${msg.role}`}>
                                        <div className="message-info">{msg.role === 'user' ? 'You' : npcName}</div>
                                        <div className="message-bubble">
                                            {msg.text}
                                            {msg.role === 'assistant' && status === 'streaming' && idx === messages.length - 1 && (
                                                <span className="cursor-pulse">▍</span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                                <div ref={messagesEndRef} />
                            </div>

                            <div className="input-area">
                                <div className="input-wrapper">
                                    <textarea
                                        ref={textareaRef}
                                        className="chat-input"
                                        value={inputText}
                                        onChange={e => setInputText(e.target.value)}
                                        onKeyDown={handleKeyDown}
                                        placeholder={`Message ${npcName}...`}
                                        disabled={status === 'streaming'}
                                    />
                                    <button
                                        className="send-btn"
                                        onClick={handleSend}
                                        disabled={!inputText.trim() || status === 'streaming'}
                                    >
                                        <Send size={20} />
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* ---------------- MEMORY EDITOR TAB ---------------- */}
                {activeTab === 'memory' && (
                    <div className="layout">
                        {/* File Sidebar */}
                        <div className="sidebar">
                            <div className="control-group">
                                <button className="btn" onClick={createNewMemory} style={{ width: '100%', marginBottom: '1rem' }}>
                                    + New NPC Memory
                                </button>
                                <div className="file-list">
                                    <label>Memory Files</label>
                                    <div className="list-container">
                                        {memoryList.map(file => (
                                            <div
                                                key={file}
                                                className={`list-item ${selectedMemory === file ? 'selected' : ''}`}
                                                onClick={() => loadMemory(file)}
                                            >
                                                {file}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Editor Area */}
                        <div className="chat-area">
                            <div className="header">
                                <div className="status-indicator">
                                    <span>{selectedMemory ? `Editing: ${selectedMemory}` : 'Select a file'}</span>
                                </div>
                                <div className="actions">
                                    <span style={{ marginRight: '1rem', fontSize: '0.8rem', opacity: 0.7 }}>{editorStatus}</span>
                                    <button className="btn" onClick={saveMemory} disabled={!selectedMemory}>
                                        Save Changes
                                    </button>
                                </div>
                            </div>
                            <div className="editor-container" style={{ flex: 1, padding: '1rem', display: 'flex' }}>
                                <textarea
                                    value={editorContent}
                                    onChange={e => setEditorContent(e.target.value)}
                                    style={{
                                        flex: 1,
                                        fontFamily: 'monospace',
                                        background: 'var(--bg-secondary)',
                                        color: 'var(--text-primary)',
                                        border: '1px solid var(--border)',
                                        borderRadius: '4px',
                                        padding: '1rem',
                                        fontSize: '0.9rem',
                                        lineHeight: '1.4'
                                    }}
                                    spellCheck={false}
                                />
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

