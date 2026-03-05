import sounddevice as sd
import soundfile as sf

# Set your Jabra mic by device index
device_index = 1  # or 11 or 26, if 1 doesn't work

# Recording parameters
duration = 5  # seconds
samplerate = 16000  # Whisper prefers 16kHz
filename = "output.wav"

# Record audio
print("Recording...")
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_index)
sd.wait()
print("Done.")

# Save to WAV
sf.write(filename, audio, samplerate)
print(f"Saved to {filename}")
