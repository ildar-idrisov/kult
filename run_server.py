from kult import StreamProcessor

# Запуск процессора потока
if __name__ == "__main__":
    audio_model_path = './kult/models/checkpoints/speech_model.pth'
    audio_scaler_path = './kult/models/checkpoints/standard_scaler.save'
    processor = StreamProcessor(audio_model_path = audio_model_path, audio_scaler_path = audio_scaler_path)
    processor.start()