import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class WhisperAgent:
    def __init__(self, model="whisper-large-v3"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

    def transcribe(self, audio_file_path: str, language: str = None) -> str:
        """
        Transcribes an audio file using Groq's Whisper API.
        """
        try:
            with open(audio_file_path, "rb") as file:
                # language should be ISO-639-1 code (e.g. 'en', 'hi', 'mr')
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), file.read()),
                    model=self.model,
                    prompt="The user is describing medical symptoms in a clinical context.",
                    response_format="json",
                    language=language, 
                    temperature=0.0
                )
                return transcription.text
        except Exception as e:
            print(f"Whisper Transcription Error: {e}")
            return ""
