import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
class MainEngine:
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    def speech_to_text(filename):
        with open(filename, "rb") as file:
            transcription=MainEngine.client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                )
            return transcription.text
            

    def process_answer(text):
        completion = MainEngine.client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
            {
                "role": "user",
                "content": ""
            }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        return completion


