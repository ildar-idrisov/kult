import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mode
import joblib

from .utils.feature_extraction import get_features, get_features_file, audio_to_mono
from .models.audio_model import load_audio_model

class AudioProcessor:
    def __init__(self, model_path, scaler_path, device='cpu'):
        self.device = torch.device(device)
        self.model = load_audio_model(model_path, device=self.device)
        self.scaler = joblib.load(scaler_path)
        self.emotions_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

        
    # Функция для изменения размерности массива до требуемого размера (добавляем нули, удаляем окончание)
    def adjust_ndarray_size(self, features, target_length):
        current_length = features.shape[1]

        if current_length < target_length:
            # Добавляем нули, чтобы увеличить размер до target_length
            pad_width = target_length - current_length
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif current_length > target_length:
            # Обрезаем массив до target_length
            features = features[:, :target_length]

        return features


    def speech_emotion(self, audio_data, sampling_rate):
        audio_data = audio_to_mono(audio_data)
        # Преобразование в float32 и нормализация для librosa
        audio_data = audio_data.astype(np.float32) / 32768.0

        features = get_features(audio_data, sampling_rate)
        #features = get_features_file('./1001_DFA_ANG_XX.wav')

        # Проверяем размер фичей и меняем размерность при необходимости
        features = self.adjust_ndarray_size(features, target_length=2376)

        # Масштабирование признаков
        features = self.scaler.transform(features)  # Преобразуем в массив с одной строкой для масштабирования

        # Преобразование в тензор и перенос на устройство
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Предсказание
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)  # Получаем вероятности
            y_pred = torch.argmax(probabilities, dim=1)  # Находим индекс максимальной вероятности

        # Перенос предсказаний на CPU и преобразование в NumPy
        y_pred = y_pred.cpu().numpy()
        
        # Вывод предсказанной эмоции
        try:
            predicted_class = mode(y_pred).mode
            predicted_emotion = self.emotions_classes[predicted_class]
            predicted_probs = mode(probabilities.cpu().numpy()).mode
            predicted_prob = predicted_probs[predicted_class]
        except:
            predicted_class = y_pred[0]
            predicted_emotion = self.emotions_classes[predicted_class]
            predicted_probs = probabilities.cpu().numpy()[0]
            predicted_prob = predicted_probs[predicted_class]

        return predicted_emotion, predicted_prob