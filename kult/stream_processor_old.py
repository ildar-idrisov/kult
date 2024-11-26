import threading
import time
import numpy as np
import av
import logging
import torch

from .audio_processing import AudioProcessor
from .video_processing import VideoProcessor

# Настройка логирования
logging.basicConfig(level=logging.ERROR, format='%(asctime)s [%(levelname)s] %(message)s')

# Определение состояний
STATE_IDLE = 'ОЖИДАНИЕ'
STATE_CAPTURING = 'ЗАХВАТ'
STATE_ANALYZING = 'АНАЛИЗ'

SERVER_IP = '0.0.0.0'  # Принимает соединения с любого адреса
SERVER_PORT = 5000
PASSPHRASE = 'xmRUkUn4gBS4iac'  # Пароль для шифрования SRT-потока

class StreamProcessor:
    def __init__(self, audio_model_path, audio_scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")
        self.audio_processor = AudioProcessor(audio_model_path, audio_scaler_path, self.device)
        self.video_processor = VideoProcessor(self.device)

        self.state = STATE_IDLE
        self.video_frames = []  # Список для хранения видео кадров
        self.audio_frames = []  # Список для хранения аудио кадров
        self.state_lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.is_running = True

        # Настройки времени
        self.capture_duration = 2.4    # Время захвата в секундах
        
        self.capture_thread = None
        self.analysis_thread = None

    def start(self):
        # Запуск потоков захвата и анализа
        self.capture_thread = threading.Thread(target=self.capture_thread_func)
        self.analysis_thread = threading.Thread(target=self.analyze_thread_func)

        self.capture_thread.start()
        self.analysis_thread.start()

        # Основной цикл управления состояниями
        try:
            while self.is_running:
                with self.state_lock:
                    if self.state == STATE_IDLE:
                        logging.info("Состояние: ОЖИДАНИЕ")
                        self.state = STATE_CAPTURING
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

        # Ожидание завершения потоков
        self.capture_thread.join()
        self.analysis_thread.join()
        logging.info("Процесс остановлен.")

    def capture_thread_func(self):
        server_url = f"srt://{SERVER_IP}:{SERVER_PORT}?mode=listener&encryption=encrypt&pbkeylen=16&passphrase={PASSPHRASE}"

        logging.info(f"Ожидание SRT-потока на {server_url}")

        container = None

        try:
            # Открываем SRT-поток
            container = av.open(server_url, format='mpegts', options={'protocol_whitelist': 'file,crypto,srt,udp'})
            packets = container.demux()
            logging.info("Поток принят, начинается захват...")

            start_time = None
            current_time = None

            for packet in packets:
                if not self.is_running:
                    break
                with self.state_lock:
                    current_state = self.state

                if packet.stream.type in ['video', 'audio']:
                    frames = packet.decode()
                    for frame in frames:
                        if current_state == STATE_CAPTURING:
                            if start_time is None:
                                start_time = frame.pts
                                logging.info("Состояние: ЗАХВАТ")

                            current_time = frame.pts
                            if (current_time - start_time) * frame.time_base < self.capture_duration:
                                if packet.stream.type == 'video':
                                    self.video_frames.append(frame)
                                elif packet.stream.type == 'audio':
                                    self.audio_frames.append(frame)
                            else:
                                with self.state_lock:
                                    self.state = STATE_ANALYZING
                                start_time = None
                                break
                        else:
                            # Во время анализа просто продолжаем потреблять пакеты без их обработки
                            pass
            #while self.is_running:
            #    with self.state_lock:
            #        current_state = self.state
#
            #    if current_state == STATE_CAPTURING:
            #        if start_time is None:
            #            #start_time = time.time()
            #            logging.info("Состояние: ЗАХВАТ")
            #        # Захватываем пакеты и записываем их в соответствующие списки
            #        packet = next(packets)
            #        if packet.stream.type in ['video', 'audio']:
            #            with self.buffer_lock:
            #                frames = packet.decode()
            #                for frame in frames:
            #                    if start_time is None:
            #                        start_time = frame.pts
            #                    current_time = frame.pts
            #                    if (current_time - start_time) * frame.time_base < self.capture_duration:
            #                        if packet.stream.type == 'video':
            #                            #logging.info("Сохранение видео кадра в список")
            #                            self.video_frames.append(frame)  # Сохраняем видео кадр в список
            #                        elif packet.stream.type == 'audio':
            #                            #logging.info("Сохранение аудио фрейма в список")
            #                            self.audio_frames.append(frame)  # Сохраняем аудио фрейм в список
            #                    else:
            #                        with self.state_lock:
            #                            self.state = STATE_ANALYZING
            #                        start_time = None
            #                        break
            #    else:
            #        # Отбрасываем данные во время анализа
            #        next(packets)
            #        #logging.info("Отбрасываем данные")
            #        #time.sleep(0.1)
        except av.AVError as e:
            logging.error(f"Ошибка при приеме или декодировании потока: {e}")
        except Exception as e:
            logging.error(f"Непредвиденная ошибка: {e}")
        finally:
            if container:
                container.close()

    def analyze_thread_func(self):
        while self.is_running:
            with self.state_lock:
                current_state = self.state

            if current_state == STATE_ANALYZING:
                logging.info("Состояние: АНАЛИЗ")
                with self.buffer_lock:
                    video_frames_to_analyze = list(self.video_frames)
                    self.video_frames.clear()

                    audio_frames_to_analyze = list(self.audio_frames)
                    self.audio_frames.clear()

                self.analyze_data(video_frames_to_analyze, audio_frames_to_analyze)
                with self.state_lock:
                    self.state = STATE_IDLE
            time.sleep(0.1)

    def analyze_data(self, video_frames, audio_frames):
        try:
            # Анализ видео данных
            for frame in video_frames:
                logging.info("Анализ видео кадра")

                frame_array = frame.to_ndarray(format='bgr24')
                import cv2
                cv2.imwrite('./tmp_image.jpg', frame_array)
                dominant_emotion, emotions, region = self.video_processor.face_emotion(frame_array)
                if (dominant_emotion and emotions and region):
                    logging.info(f"Видео: {dominant_emotion}, {emotions[dominant_emotion]}")

            # Анализ аудио данных
            all_audio_samples = []
            for frame in audio_frames:
                #logging.info("Анализ аудио фрейма")
                audio_data = frame.to_ndarray()  # Преобразуем аудио данные в numpy массив
                sampling_rate = frame.sample_rate  # Получаем частоту дискретизации

                # Преобразуем аудио данные к виду, совместимому с librosa
                audio_data = audio_data.astype(np.float32) / 32768.0  # Преобразуем в float32
                audio_data = np.mean(audio_data, axis=0) # Преобразуем в моно
                all_audio_samples.append(audio_data)
                
                #predicted_emotion, predicted_prob = self.audio_processor.speech_emotion(audio_data, sampling_rate)
                #logging.info(f"Аудио: {predicted_emotion}, {predicted_prob}")
            
            if all_audio_samples or False:
                combined_audio_data = np.concatenate(all_audio_samples)
                #import soundfile as sf
                #sf.write('./tmp_audio.wav', combined_audio_data, samplerate=sampling_rate)
                predicted_emotion, predicted_prob = self.audio_processor.speech_emotion(combined_audio_data, sampling_rate)
                if (predicted_emotion and predicted_prob):
                    logging.info(f"Аудио: {predicted_emotion}, {predicted_prob}")
            print(dominant_emotion, predicted_emotion)

            logging.info("Анализ завершен.")
        except Exception as e:
            logging.error(f"Непредвиденная ошибка при анализе данных: {e}")

    def stop(self):
        self.is_running = False
