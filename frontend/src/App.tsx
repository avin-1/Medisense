import { useState, useRef, useEffect } from 'react'
import './index.css'

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  type?: 'diagnosis' | 'alert' | 'clarification' | 'error' | 'update';
  disease?: string;
  diseases?: string[];
  reasoning?: string;
  precautions?: string[];
  riskScore?: number;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "init",
      sender: "bot",
      text: "Hello! I am your AI Medical Assistant. Please describe your symptoms in your own words, or type '/update DiseaseName' to teach me a new disease.",
      type: "clarification"
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: input
    };

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      if (input.trim().toLowerCase().startsWith('/update ')) {
        const disease = input.slice(8).trim();
        const res = await fetch("http://localhost:8000/update_disease", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ disease_name: disease })
        });
        const data = await res.json();
        setMessages(prev => [...prev, {
          id: Date.now().toString() + "bot",
          sender: "bot",
          text: data.message || "Updated successfully.",
          type: "update"
        }]);
      } else {
        const res = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_input: userMsg.text })
        });
        const data = await res.json();
        
        setMessages(prev => [...prev, {
          id: Date.now().toString() + "bot",
          sender: "bot",
          text: data.message || "Something went wrong.",
          type: data.type,
          disease: data.disease,
          diseases: data.diseases,
          reasoning: data.reasoning,
          precautions: data.precautions,
          riskScore: data.risk_score
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now().toString() + "err",
        sender: "bot",
        text: "Error connecting to the backend server. Make sure the FastAPI app is running on port 8000.",
        type: "error"
      }]);
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <div className="chat-window">
        <header className="chat-header">
          <h1>Medical Agent AI</h1>
          <span className="status">Online</span>
        </header>

        <div className="chat-messages">
          {messages.map(msg => (
            <div key={msg.id} className={`message-bubble ${msg.sender} ${msg.type === 'alert' ? 'alert-bubble' : ''}`}>
              <div className="message-content">
                <p>{msg.text}</p>
                {msg.type === 'diagnosis' && msg.diseases && msg.diseases.length > 0 && (
                  <div className="precautions-card" style={{marginTop: '12px'}}>
                    <h4>Ranked Possibilities:</h4>
                    <ol style={{ margin: 0, paddingLeft: '20px', color: '#cbd5e1' }}>
                      {msg.diseases.map((d, i) => <li key={i} style={{marginBottom: '4px'}}>{d}</li>)}
                    </ol>
                  </div>
                )}
                {msg.reasoning && (
                  <div className="precautions-card" style={{marginTop: '12px', background: 'rgba(59, 130, 246, 0.1)', borderColor: 'rgba(59, 130, 246, 0.3)'}}>
                    <h4 style={{color: '#93c5fd', borderBottom: '1px solid rgba(59, 130, 246, 0.2)', paddingBottom: '6px', marginBottom: '8px'}}>Chief Medical Synthesizer Logic:</h4>
                    <p style={{margin: '0', fontSize: '0.9rem', color: '#bfdbfe', lineHeight: 1.6}}>{msg.reasoning}</p>
                  </div>
                )}
                {msg.type === 'diagnosis' && msg.precautions && msg.precautions.length > 0 && (
                  <div className="precautions-card">
                    <h4>Preventative Measures:</h4>
                    <ul>
                      {msg.precautions.map((p, i) => <li key={i}>{p}</li>)}
                    </ul>
                  </div>
                )}
                {msg.type === 'alert' && (
                  <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-start' }}>
                    {msg.riskScore && <div className="risk-indicator" style={{margin: 0}}>Critical Triage Score: {msg.riskScore}</div>}
                    {msg.diseases && msg.diseases.length > 0 && (
                      <div className="risk-indicator" style={{margin: 0, background: 'rgba(255,255,255,0.2)', textAlign: 'left'}}>
                        <div style={{marginBottom: '4px'}}><strong>Possible Conditions:</strong></div>
                        <ol style={{ margin: '0 0 0 20px', padding: 0 }}>
                          {msg.diseases.map((d, i) => <li key={i}>{d}</li>)}
                        </ol>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message-bubble bot typing">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form className="chat-input-area" onSubmit={handleSend}>
          <input 
            type="text" 
            placeholder="E.g., I have a bad headache, stiff neck, and high fever..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send 
          </button>
        </form>
      </div>
    </div>
  )
}

export default App
