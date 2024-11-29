import whisper
from transformers import pipeline

from .utils.feature_extraction import audio_to_mono, audio_resample

DEBUG = False

class TextProcessor:
    def __init__(self, device='cpu'):
        self.whisper_model = whisper.load_model("turbo", device=device)
        self.emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    def transcribe_audio(self, audio_data, sampling_rate):
        # Транскрипция аудио
        result = self.whisper_model.transcribe(audio_data, fp16=False, task='transcribe', language='english') # translate transcribe russian english
        return result['text']

    def analyze_emotion(self, text):
        # Анализ эмоций по тексту
        emotions = self.emotion_analyzer(text)
        return emotions

    def text_emotion(self, audio_data, sampling_rate):
        audio_data = audio_to_mono(audio_data)
        audio_data, sampling_rate = audio_resample(audio_data, sampling_rate, 16000)
        transcript = self.transcribe_audio(audio_data, sampling_rate)
        emotions = self.analyze_emotion(transcript)[0]
        if DEBUG:
            print(emotions)
        
        max_emotion = max(emotions, key=lambda x: x['score'])
        return max_emotion['label'], max_emotion['score'], transcript