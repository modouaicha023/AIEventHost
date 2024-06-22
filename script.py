import sounddevice as sd
import numpy as np

duration = 5  # seconds
fs = 44100  # Sample rate

print("Enregistrement...")
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("Enregistrement terminé")