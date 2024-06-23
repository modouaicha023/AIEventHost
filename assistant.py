import pyttsx3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import logging
from voice_recognition import listen

# Configuration du modèle GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuration de la synthèse vocale
engine = pyttsx3.init()
voices = engine.getProperty('voices')

fr_voice_id = None

# Chercher une voix en français
for voice in voices:
    if 'FR' in voice.id:
        fr_voice_id = voice.id
        break

def ask_gpt(prompt):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse : {str(e)}")
        response = "Je suis désolé, je n'ai pas pu comprendre votre demande pour le moment."
    return response

if __name__ == "__main__":
    logging.info("Parlez maintenant...")
    try:
        while True:
            user_input = listen()            
            if user_input:
                logging.info(f"Vous avez dit: {user_input}")
                response = ask_gpt(user_input)
                logging.info(f"Assistant répond: {response}")
                engine.say(response)
                engine.runAndWait()
            else:
                logging.info("Aucune entrée détectée. Réessayez.")
    except KeyboardInterrupt:
        logging.info("Arrêt de l'assistant vocal.")
    except Exception as e:
        logging.error(f"Erreur inattendue : {str(e)}")

