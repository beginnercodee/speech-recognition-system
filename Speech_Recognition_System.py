# %% [markdown]
# 📌 Install Dependencies

# %%
!pip install -q transformers torchaudio librosa

# %% [markdown]
# 📌 Import Libraries

# %%
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gradio as gr

# %% [markdown]
# 📌 Load Pretrained Speech Recognition Model

# %%
# Load Wav2Vec2 model and tokenizer from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# %% [markdown]
# **📌** Define Speech-to-Text Function

# %%
def speech_to_text(audio_path):
    # Load and resample the audio
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Preprocess audio into model input format
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

    # Run model inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

# %% [markdown]
# 🎤 Gradio Interface (Frontend)

# %%
interface = gr.Interface(
    fn=speech_to_text,
    inputs=gr.Audio(type="filepath", label="🎤 Speak or Upload Audio (.wav)"),
    outputs=gr.Textbox(label="📝 Transcribed Text"),
    title="🎙️ Speech Recognition System",
    description="Speak or upload an audio file to transcribe it using a deep learning model (Wav2Vec2 + CTC)."
)

interface.launch(share=True)


