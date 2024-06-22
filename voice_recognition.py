import sounddevice as sd
import queue
import vosk
import json
import logging


model_vosk = vosk.Model("vosk-model-small-fr-0.22")
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        logging.warning(f"Status: {status}")
    q.put(bytes(indata))

def listen():
    try:
        logging.info("Initialisation du stream audio...")
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback) as stream:
            logging.info("Stream audio initialisé avec succès")
            try:
                logging.info("Initialisation du KaldiRecognizer...")
                rec = vosk.KaldiRecognizer(model_vosk, 16000, "fr")
                logging.info("KaldiRecognizer initialisé avec succès")
            except Exception as e:
                logging.error(f"Erreur lors de l'initialisation du KaldiRecognizer : {str(e)}")
                return None
            logging.info("Reconnaissance vocale initialisée")
            logging.info("Début de l'écoute...")
            while True:
                try:
                    logging.debug("Attente de données audio...")
                    data = q.get(timeout=5)
                    logging.debug("Données audio reçues")
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        logging.info(f"Résultat de la reconnaissance vocale: {result}")
                        if 'text' in result:
                            return result['text']
                        else:
                            logging.warning("Aucun texte reconnu dans le résultat.")
                except queue.Empty:
                    logging.warning("Timeout: aucune donnée reçue du microphone.")
                except Exception as e:
                    logging.error(f"Erreur lors de la reconnaissance vocale : {str(e)}")
                    return None
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de l'écoute : {str(e)}")
        return None