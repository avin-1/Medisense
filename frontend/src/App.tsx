import { useState, useRef, useEffect } from 'react'
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
        // Filter out specific onboarding bot messages and start from when symptoms begin
        const symptomMessages = messages.filter(m => {
          // If it's a bot message, we only want it if it's NOT an onboarding prompt
          if (m.sender === 'bot') {
            return !['Welcome to Medisense.', 'Got it. What is your gender?', 'Thank you. Where are you currently located?', 'Nearly there. Do you have any existing medical conditions', 'Profile updated. Now, please describe'].some(t => m.text.includes(t));
          }
          // If it's a user message, we only want it if it was sent AFTER the onboarding HISTORY step
          // Since messages are in order, we can filter by type or content
          return true; 
        });

        const chatHistoryForBackend = symptomMessages.map(m => ({
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
          hypotheses: data.hypotheses,
          reasoning: data.reasoning,
          precautions: data.precautions,
          riskScore: data.risk_score,
          urgency: data.urgency
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
                {msg.type === 'diagnosis' && msg.hypotheses && msg.hypotheses.length > 0 && (
                  <div className="precautions-card" style={{marginTop: '12px'}}>
                    <h4>Clinical Hypotheses:</h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {msg.hypotheses.map((h, i) => (
                        <div key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '6px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#f8fafc' }}>
                            <strong>{h.disease}</strong>
                            <span style={{ color: h.confidence > 70 ? '#4ade80' : '#fbbf24' }}>{h.confidence}%</span>
                          </div>
                          <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: '#94a3b8' }}>{h.reasoning}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {msg.type === 'diagnosis' && msg.urgency && (
                   <div style={{ marginTop: '12px' }}>
                     <div className="risk-indicator" style={{ 
                       background: msg.urgency.includes('Immediate') ? 'rgba(239, 68, 68, 0.3)' : 'rgba(59, 130, 246, 0.3)',
                       color: 'white',
                       fontWeight: 'bold',
                       textAlign: 'center'
                     }}>
                       Recommended Urgency: {msg.urgency}
                     </div>
                   </div>
                )}
                {msg.type === 'clarification' && msg.hypotheses && msg.hypotheses.length > 0 && (
                  <div className="precautions-card" style={{marginTop: '12px', background: 'rgba(255, 255, 255, 0.05)'}}>
                    <h5 style={{margin: '0 0 8px', color: '#94a3b8', fontSize: '0.8rem', textTransform: 'uppercase'}}>Current Differential:</h5>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {msg.hypotheses.map((h, i) => (
                        <span key={i} style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: '12px', fontSize: '0.8rem', color: '#cbd5e1' }}>
                          {h.disease} ({h.confidence}%)
                        </span>
                      ))}
                    </div>
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
