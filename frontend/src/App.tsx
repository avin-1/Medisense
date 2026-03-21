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

type OnboardingStep = 'AGE' | 'GENDER' | 'LOCATION' | 'HISTORY' | 'SYMPTOMS';

interface UserData {
  age: string;
  gender: string;
  location: string;
  medical_history: string;
}

function App() {
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
      text: "Welcome to Medisense. To provide a high-precision analysis, I need a few details first. What is your age?",
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

    const userText = input.trim();
    const userMsg: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: userText
    };

    setMessages(prev => [...prev, userMsg]);
    setInput("");

    // Handle Onboarding Steps
    if (step === 'AGE') {
      setUserData(prev => ({ ...prev, age: userText }));
      setStep('GENDER');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Got it. What is your gender?", type: "clarification" }]);
      return;
    }
    if (step === 'GENDER') {
      setUserData(prev => ({ ...prev, gender: userText }));
      setStep('LOCATION');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Thank you. Where are you currently located?", type: "clarification" }]);
      return;
    }
    if (step === 'LOCATION') {
      setUserData(prev => ({ ...prev, location: userText }));
      setStep('HISTORY');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Nearly there. Do you have any existing medical conditions or history I should know about? (Type 'None' if not)", type: "clarification" }]);
      return;
    }
    if (step === 'HISTORY') {
      setUserData(prev => ({ ...prev, medical_history: userText }));
      setStep('SYMPTOMS');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Profile updated. Now, please describe the symptoms you are currently experiencing.", type: "clarification" }]);
      return;
    }

    // Normal Symptom Processing
    setLoading(true);

    try {
      if (userText.toLowerCase().startsWith('/update ')) {
        const disease = userText.slice(8).trim();
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
        // Filter out onboarding messages from the chat_history sent to the diagnosis backend
        // We only want the symptom-related turns
        const chatHistoryForBackend = messages
          .filter(m => m.id !== 'init' && !['AGE', 'GENDER', 'LOCATION', 'HISTORY'].includes(m.id))
          .map(m => ({
            role: m.sender === 'user' ? 'user' : 'assistant',
            content: m.text
          }));

        const res = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            user_input: userText,
            user_data: userData,
            chat_history: chatHistoryForBackend
          })
        });
        const data = await res.json();
        
        let botText = data.message;
        if (data.type === 'clarification' && data.followup_questions && data.followup_questions.length > 0) {
          botText += "\n\nFollow-up Questions:\n" + data.followup_questions.map((q: string, i: number) => `${i+1}. ${q}`).join("\n");
        }

        setMessages(prev => [...prev, {
          id: Date.now().toString() + "bot",
          sender: "bot",
          text: botText,
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
