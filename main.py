import openai
import pyttsx3
import sounddevice as sd
import queue
import vosk
import json
from dotenv import load_dotenv
import os

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer la clé API OpenAI à partir des variables d'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialisez le client OpenAI
client = openai.OpenAI()

def ask_gpt(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo",  # Remplacez par le modèle GPT-4 approprié
        prompt=prompt,
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()


# Configuration de la synthèse vocale
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Configuration de la reconnaissance vocale avec Vosk
model = vosk.Model("./vosk-model-small-fr-0.22")
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

def listen():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, 16000)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                return result['text']

if __name__ == "__main__":
    print("Parlez maintenant...")
    user_input = listen()
    print("Vous avez dit:", user_input)
    response = ask_gpt(user_input)
    print("GPT-4 répond:", response)
    speak(response)
