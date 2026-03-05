import whisper
from transformers import pipeline
from TTS.api import TTS


# 1. Record Audio
def record_audio(filename="input.wav", duration=10, fs=16000, device=1):
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wav

    print("Recording...")
    sd.default.device = (device, None)  # (input_device, output_device)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, np.squeeze(audio))
    return filename

# 2. Speech to Text (ASR)
print("Loading Whisper ASR model...")
asr_model = whisper.load_model("base")
audio_file = record_audio()

print("Transcribing...")
asr_result = asr_model.transcribe(audio_file)
text = asr_result["text"]
print("You said:", text)

# 3. Chatbot
print("Generating response...")
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
response = chatbot(f"[INST] {text} [/INST]", max_new_tokens=100)[0]["generated_text"]

# Remove prompt if still present in generated text
# if "[/INST]" in response:
#     response = response.split("[/INST]")[-1].strip()
response = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response.strip()

print("Bot says:", response)

# 4. Text to Speech
print("Converting to speech...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
tts.tts_to_file(text=response, file_path="output.wav")

# 5. Play Response
print("Playing response...")
try:
    from playsound import playsound
    playsound("output.wav")
except Exception as e:
    print("Error playing sound:", e)
