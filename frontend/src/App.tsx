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
      console.error("Recording Error:", err);
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
    if (selectedLanguage) {
      formData.append("language", selectedLanguage);
    }

    try {
      const res = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.text) {
        // Auto-send the transcribed text
        await processDiagnosticFlow(data.text);
      }
    } catch (error) {
      console.error("Transcription Error:", error);
    }
    setLoading(false);
  };

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim()) return;
    const userText = input.trim();
    setInput("");
    await processDiagnosticFlow(userText);
  };

  const processDiagnosticFlow = async (text: string) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: text
    };

    setMessages(prev => [...prev, userMsg]);

    // Handle Onboarding Steps
    if (step === 'AGE') {
      setUserData(prev => ({ ...prev, age: text }));
      setStep('GENDER');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Got it. What is your gender?", type: "clarification" }]);
      return;
    }
    if (step === 'GENDER') {
      setUserData(prev => ({ ...prev, gender: text }));
      setStep('LOCATION');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Thank you. Where are you currently located?", type: "clarification" }]);
      return;
    }
    if (step === 'LOCATION') {
      setUserData(prev => ({ ...prev, location: text }));
      setStep('HISTORY');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Nearly there. Do you have any existing medical conditions or history I should know about? (Type 'None' if not)", type: "clarification" }]);
      return;
    }
    if (step === 'HISTORY') {
      setUserData(prev => ({ ...prev, medical_history: text }));
      setStep('SYMPTOMS');
      setMessages(prev => [...prev, { id: Date.now().toString() + "bot", sender: "bot", text: "Profile updated. Now, please describe the symptoms you are currently experiencing.", type: "clarification" }]);
      return;
    }

    setLoading(true);
    try {
      if (text.toLowerCase().startsWith('/update ')) {
        const disease = text.slice(8).trim();
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
        const symptomMessages = messages.filter(m => {
          if (m.sender === 'bot') {
            return !['Welcome to Medisense.', 'Got it. What is your gender?', 'Thank you. Where are you currently located?', 'Nearly there. Do you have any existing medical conditions', 'Profile updated. Now, please describe'].some(t => m.text.includes(t));
          }
          return true; 
        });

        const chatHistoryForBackend = [...symptomMessages, userMsg].map(m => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text
        }));

        const res = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            user_input: text,
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
          hypotheses: data.hypotheses,
          reasoning: data.reasoning,
          precautions: data.precautions,
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
          <h1>Medisense AI</h1>
          <span className="status">Diagnostic Center</span>
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
                       background: msg.urgency.includes('Immediate') ? 'rgba(239, 68, 68, 0.4)' : 'rgba(59, 130, 246, 0.4)',
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
          <div className="mic-wrapper" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button 
              type="button" 
              className={`mic-button ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              style={{ 
                background: isRecording ? '#ef4444' : 'rgba(255,255,255,0.1)',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}
            >
              {isRecording ? '⏹' : '🎤'}
            </button>
            <select 
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              className="lang-selector"
              style={{
                background: 'rgba(255,255,255,0.1)',
                border: 'none',
                color: 'white',
                borderRadius: '8px',
                padding: '4px 8px',
                fontSize: '0.8rem'
              }}
            >
              <option value="en">EN</option>
              <option value="hi">HI</option>
              <option value="mr">MR</option>
            </select>
          </div>
          <input 
            type="text" 
            placeholder="Type or use mic..." 
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
