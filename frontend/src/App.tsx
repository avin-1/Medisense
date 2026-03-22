import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

interface Hypothesis {
  disease: string;
  confidence: number;
  reasoning: string;
}

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  type?: 'diagnosis' | 'alert' | 'clarification' | 'error' | 'update';
  disease?: string;
  diseases?: string[];
  hypotheses?: Hypothesis[];
  reasoning?: string;
  precautions?: string[];
  riskScore?: number;
  urgency?: string;
}

type OnboardingStep = 'AGE' | 'GENDER' | 'LOCATION' | 'HISTORY' | 'SYMPTOMS';

interface UserData {
  age: string;
  gender: string;
  location: string;
  medical_history: string;
}

interface ChatSession {
  id: string;
  messages: Message[];
  userData: UserData;
  step: OnboardingStep;
  timestamp: number;
}

function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(Date.now().toString());
  
  const [step, setStep] = useState<OnboardingStep>('AGE');
  const [userData, setUserData] = useState<UserData>({
    age: '',
    gender: '',
    location: '',
    medical_history: ''
  });

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "init",
      sender: "bot",
      text: "Welcome to Aarogya Doot. To provide a high-precision analysis, I need a few details first. What is your age?",
      type: "clarification"
    }
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Select-to-Explain state
  const [explainTooltip, setExplainTooltip] = useState<{x: number; y: number; text: string} | null>(null);
  const [explainModal, setExplainModal] = useState<{text: string; explanation: string | null} | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  // Sync current state to sessions list
  useEffect(() => {
    setSessions(prev => {
        const otherSessions = prev.filter(s => s.id !== currentSessionId);
        const currentSession: ChatSession = {
            id: currentSessionId,
            messages,
            userData,
            step,
            timestamp: Date.now()
        };
        return [currentSession, ...otherSessions].sort((a, b) => b.timestamp - a.timestamp);
    });
  }, [messages, userData, step, currentSessionId]);

  const startNewChat = () => {
    const newId = Date.now().toString();
    setCurrentSessionId(newId);
    setStep('AGE');
    setUserData({ age: '', gender: '', location: '', medical_history: '' });
    setMessages([{
      id: "init-" + newId,
      sender: "bot",
      text: "Welcome to Aarogya Doot. To provide a high-precision analysis, I need a few details first. What is your age?",
      type: "clarification"
    }]);
  };

  const loadSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setStep(session.step);
    setUserData(session.userData);
    setMessages(session.messages);
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      // Don't focus if already in an input/textarea or if using special keys
      const active = document.activeElement?.tagName;
      if (active === 'INPUT' || active === 'TEXTAREA') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      
      // Printable characters move focus to our chat input
      if (e.key.length === 1 && inputRef.current) {
        inputRef.current.focus();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  // Select-to-Explain: show tooltip on text selection inside chat
  useEffect(() => {
    const handleMouseUp = (e: MouseEvent) => {
      const selection = window.getSelection();
      const selectedText = selection?.toString().trim();
      if (!selectedText || selectedText.length < 3) {
        setExplainTooltip(null);
        return;
      }
      const target = e.target as HTMLElement;
      if (!target.closest('.chat-messages')) {
        setExplainTooltip(null);
        return;
      }
      // Use the selection range rect to position tooltip ABOVE the selected text
      const range = selection?.getRangeAt(0);
      const rect = range?.getBoundingClientRect();
      if (rect) {
        setExplainTooltip({ 
          x: rect.left + rect.width / 2, 
          y: rect.top - 10,  // 10px above the selection top
          text: selectedText 
        });
      }
    };
    document.addEventListener('mouseup', handleMouseUp);
    return () => document.removeEventListener('mouseup', handleMouseUp);
  }, []);

  const handleExplain = async (text: string) => {
    setExplainTooltip(null);
    setExplainModal({ text, explanation: null });
    setExplainLoading(true);
    window.getSelection()?.removeAllRanges();
    try {
      const chatHistoryForExplain = messages.map(m => ({
        role: m.sender === 'user' ? 'user' : 'assistant',
        content: m.text
      }));
      const res = await fetch('http://localhost:8000/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_text: text,
          chat_history: chatHistoryForExplain,
          language: selectedLanguage
        })
      });
      const data = await res.json();
      setExplainModal({ text, explanation: data.explanation });
    } catch {
      setExplainModal({ text, explanation: 'Could not fetch explanation. Please check the server.' });
    } finally {
      setExplainLoading(false);
    }
  };

  const [selectedLanguage, setSelectedLanguage] = useState<string>("en");
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await handleAudioUpload(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      alert("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAudioUpload = async (blob: Blob) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", blob, "recording.wav");
    if (selectedLanguage) formData.append("language", selectedLanguage);

    try {
      const res = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.text) await processDiagnosticFlow(data.text);
    } catch (error) {
      console.error("Transcription Error:", error);
    }
    setLoading(false);
  };

  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() && !selectedImage) return;
    const userText = input.trim();
    const currentImage = selectedImage;
    
    setInput("");
    setSelectedImage(null);
    setImagePreview(null);
    
    await processDiagnosticFlow(userText, currentImage);
  };

  const processDiagnosticFlow = async (text: string, imageFile?: File | null) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: text || "[Symptom Image]"
    };

    setMessages(prev => [...prev, userMsg]);

    setLoading(true);
    try {
        // Calculate the UPDATED userData immediately to avoid async state lag
        // Only update profile fields during active onboarding, never during diagnosis
        const updatedUserData = { ...userData };
        const isOnboarding = step !== 'SYMPTOMS';
        if (isOnboarding) {
          if (step === 'AGE') updatedUserData.age = text;
          else if (step === 'GENDER') updatedUserData.gender = text;
          else if (step === 'LOCATION') updatedUserData.location = text;
          else if (step === 'HISTORY') updatedUserData.medical_history = text;
        }

        // Build a CLEAN, LABELED history for the backend
        // CRITICAL: include followup questions in bot messages so agents can see what was asked
        const chatHistoryForBackend = messages.map(m => {
          let content = m.text;
          // Append followup questions to bot messages so agents see what was asked
          if (m.sender === 'bot' && m.diseases && Array.isArray(m.diseases) && m.diseases.length > 0) {
            content = content + '\n\nDoctor asked: ' + m.diseases.join(' | ');
          }
          return {
            role: m.sender === 'user' ? 'user' : 'assistant',
            content
          };
        });

        const formData = new FormData();
        const requestPayload = {
            user_input: text,
            user_data: updatedUserData,
            chat_history: chatHistoryForBackend
        };
        formData.append("user_data_json", JSON.stringify(requestPayload));
        if (imageFile) formData.append("file", imageFile);

        const res = await fetch("http://localhost:8000/chat", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        
        // Finalize state sync after successful backend response
        setUserData(updatedUserData);

        // Determine next step — ONLY during onboarding, never once profile is complete
        // Use strict prefix/question matching to avoid false triggers from clinical text
        if (step !== 'SYMPTOMS') {
          const msg = data.message.toLowerCase();
          if (msg.includes("what is your age") || msg.includes("your age?")) setStep('AGE');
          else if (msg.includes("what is your gender") || msg.includes("your gender?")) setStep('GENDER');
          else if (msg.includes("currently located") || msg.includes("where are you")) setStep('LOCATION');
          else if (msg.includes("medical conditions") || msg.includes("medical history")) setStep('HISTORY');
          else if (msg.includes("describe the symptoms") || msg.includes("describe your symptoms")) setStep('SYMPTOMS');
        }
        // If already in SYMPTOMS, NEVER change step back — profile is locked

        // Store bot message - include followup questions in the TEXT for history tracking
        const followupText = (data.followup_questions && data.followup_questions.length > 0)
          ? '\n\nDoctor asked: ' + data.followup_questions.join(' | ')
          : '';

        setMessages(prev => [...prev, {
          id: Date.now().toString() + "bot",
          sender: "bot",
          text: data.message + followupText,
          type: data.type,
          hypotheses: data.hypotheses,
          diseases: data.followup_questions,
          reasoning: data.reasoning,
          precautions: data.precautions,
          urgency: data.urgency
        }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now().toString() + "err",
        sender: "bot",
        text: "Connection Error. Check API server.",
        type: "error"
      }]);
    }
    setLoading(false);
  };

  const suggestions = [
    "High fever, severe headache and body aches for 2 days",
    "Persistent cough, chest tightness and breathlessness",
    "Excessive thirst, frequent urination and constant fatigue",
    "Skin rash with joint pain, fever and swollen lymph nodes"
  ];

  const handleSuggestion = (text: string) => {
    if (step === 'SYMPTOMS') {
        processDiagnosticFlow(text);
    }
  };

  const deleteSession = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setSessions(prev => prev.filter(s => s.id !== id));
    if (id === currentSessionId) {
        startNewChat();
    }
  };

  return (
    <div className="app-container">
      {/* Left Sidebar: History */}
      <aside className="sidebar">
        <div className="sidebar-header">Chat History</div>
        <button 
            className="history-item" 
            onClick={startNewChat}
            style={{ background: '#22c55e', color: 'white', border: 'none', width: '100%', marginBottom: '24px', padding: '10px', justifyContent: 'center' }}
        >
          + New Conversation
        </button>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {sessions.length > 0 ? sessions.map(s => (
                <div key={s.id} className={`history-item ${s.id === currentSessionId ? 'active' : ''}`} onClick={() => loadSession(s)} style={{ border: s.id === currentSessionId ? '1px solid #22c55e' : '1px solid transparent' }}>
                    <div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {s.messages[s.messages.length - 1]?.text.substring(0, 30)}...
                        <div style={{ fontSize: '0.65rem', opacity: 0.7 }}>{new Date(s.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <button className="delete-btn" onClick={(e) => deleteSession(e, s.id)}>✕</button>
                </div>
            )) : <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>No past conversations yet.</p>}
        </div>
        
        <div className="sidebar-header" style={{ marginTop: '40px' }}>Medical Profile</div>
        {userData.age ? (
            <div style={{ fontSize: '0.8rem', color: '#475569' }}>
                <p><b>Age:</b> {userData.age}</p>
                <p><b>Gender:</b> {userData.gender}</p>
                <p><b>History:</b> {userData.medical_history || 'None'}</p>
            </div>
        ) : <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Complete onboarding to view profile.</p>}
      </aside>

      {/* Main Content Area */}
      <main className="main-chat-container">
        <header className="main-header">
          <div className="brand">
            <h1>Aarogya Doot</h1>
            <span>आरोग्य दूत</span>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
             <button className="send-btn" onClick={startNewChat} style={{ background: '#f1f5f9', color: '#475569' }}>+ New Chat</button>
             <div className="status" style={{ background: 'white', padding: '4px 12px', borderRadius: '20px', fontSize: '0.75rem', fontWeight: 700 }}>
                • LIVE
             </div>
          </div>
        </header>

        <div className="chat-scroll">
          {messages.length === 1 && step === 'SYMPTOMS' ? (
            <div className="welcome-screen">
              <span>HOW CAN AAROGYA DOOT HELP YOU TODAY?</span>
              <h2>Describe your symptoms.</h2>
              <p>Type in plain Hindi or English. Our 6-agent pipeline will extract, triage, and deliver a ranked clinical assessment in seconds.</p>
              
              <div className="suggestions-grid">
                {suggestions.map((s, i) => (
                  <div key={i} className="suggestion-card" onClick={() => handleSuggestion(s)}>
                    {s}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="chat-messages" style={{ width: '100%', maxWidth: '700px' }}>
              {messages.map((msg, idx) => (
                <div key={msg.id} className={`msg-row ${msg.sender}`}>
                  <div className={`bubble ${msg.sender}`}>
                    {msg.sender === 'bot' ? (
                        <ReactMarkdown>{msg.text}</ReactMarkdown>
                    ) : (
                        <p>{msg.text}</p>
                    )}
                    {msg.hypotheses && (
                       <div className="diag-card">
                          <div className="sidebar-header" style={{ marginBottom: '8px' }}>Current Differential</div>
                          {msg.hypotheses.map((h, j) => (
                            <div key={j} className="diag-header" style={{ borderBottom: '1px solid #e2e8f0', padding: '8px 0' }}>
                                <div>
                                    <span style={{ fontWeight: 600 }}>{h.disease}</span>
                                    {msg.type === 'diagnosis' && <p style={{ fontSize: '0.75rem', color: '#64748b', margin: '4px 0' }}>{h.reasoning}</p>}
                                </div>
                                <span className="diag-confidence">{h.confidence}%</span>
                            </div>
                          ))}
                       </div>
                    )}
                    
                    {/* Follow-up Question Cards */}
                    {msg.diseases && msg.diseases.length > 0 && (
                        <div className="question-card">
                            <div className="question-header">
                                <span>📋</span> NEXT STEPS / NEEDED DETAILS
                            </div>
                            {msg.diseases.map((q, idx) => (
                                <div key={idx} className="question-item">
                                    <div className="question-num">{idx + 1}</div>
                                    <div className="question-text">{q}</div>
                                </div>
                            ))}
                        </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && <div className="msg-row bot"><div className="bubble bot">Analyzing...</div></div>}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="input-container">
          {imagePreview && (
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', background: 'white', padding: '8px', borderRadius: '12px' }}>
                <img src={imagePreview} className="img-preview" />
                <button onClick={() => {setSelectedImage(null); setImagePreview(null);}} className="icon-btn">✕</button>
            </div>
          )}
          <form className="input-bar" onSubmit={handleSend}>
             <label className="icon-btn">
                <input type="file" hidden onChange={handleImageSelect} />
                📎
             </label>
             <button type="button" className={`icon-btn ${isRecording ? 'recording' : ''}`} onClick={isRecording ? stopRecording : startRecording}>
                {isRecording ? '⏹' : '🎤'}
             </button>
             <input 
                ref={inputRef}
                placeholder="Describe your symptoms in plain language..." 
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={loading}
             />
             <select 
                value={selectedLanguage} 
                onChange={e => setSelectedLanguage(e.target.value)}
                style={{ border: 'none', background: 'none', fontWeight: 700, outline: 'none' }}
             >
                <option value="en">EN</option>
                <option value="hi">HI</option>
                <option value="mr">MR</option>
             </select>
             <button type="submit" className="send-btn" disabled={loading || (!input.trim() && !selectedImage)}>
                ↑
             </button>
          </form>
          <p style={{ textAlign: 'center', fontSize: '0.7rem', color: '#94a3b8', marginTop: '12px' }}>
            Educational purposes only · Not a substitute for professional medical advice · Shift+Enter for new line
          </p>
        </div>
      </main>

      {/* Floating Tooltip on text selection */}
      {explainTooltip && (
        <div
          className="explain-tooltip"
          style={{ 
            position: 'fixed',
            top: explainTooltip.y - 40, // appear above the selection
            left: explainTooltip.x, 
            transform: 'translateX(-50%)' 
          }}
          onMouseDown={e => { e.preventDefault(); handleExplain(explainTooltip.text); }}
        >
          💡 Explain this
        </div>
      )}

      {/* Explain Modal */}
      {explainModal && (
        <div className="explain-modal-backdrop" onClick={() => setExplainModal(null)}>
          <div className="explain-modal" onClick={e => e.stopPropagation()}>
            <div className="explain-modal-header">
              <h3>💡 Simple Explanation</h3>
              <button className="explain-modal-close" onClick={() => setExplainModal(null)}>✕</button>
            </div>
            <div className="explain-selected-text">"{explainModal.text}"</div>
            <div className="explain-modal-body">
              {explainLoading ? (
                <div className="explain-loading">
                  <div className="explain-spinner" />
                  Generating friendly explanation...
                </div>
              ) : (
                <ReactMarkdown>{explainModal.explanation ?? ''}</ReactMarkdown>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
