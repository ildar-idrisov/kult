import threading
import time
import numpy as np
import av
import logging
import torch
import subprocess
import os
import glob

from .video_processing import VideoProcessor
from .audio_processing import AudioProcessor
from .text_processing import TextProcessor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

SERVER_IP = '0.0.0.0'
SERVER_PORT = 5000
PASSPHRASE = 'SnSu72im4o12'

class StreamProcessor:

    def __init__(self, audio_model_path, audio_scaler_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Device: {self.device}')
        self.audio_processor = AudioProcessor(audio_model_path, audio_scaler_path, self.device)
        self.video_processor = VideoProcessor(self.device)
        self.text_processor = TextProcessor(self.device)

        self.is_running = True

        self.capture_thread = None
        self.analysis_thread = None

        self.segment_directory = 'cache/'
        self.segment_duration = 2.4
        self.last_processed_time = 0

        # Параметры SRT-потока
        self.server_ip = SERVER_IP
        self.server_port = SERVER_PORT
        self.passphrase = PASSPHRASE

    def start(self):
        # Создаем директорию для сегментов, если она не существует
        if not os.path.exists(self.segment_directory):
            os.makedirs(self.segment_directory)

        # Запуск потоков захвата и анализа
        self.capture_thread = threading.Thread(target=self.capture_thread_func)
        self.analysis_thread = threading.Thread(target=self.analysis_thread_func)

        self.capture_thread.start()
        self.analysis_thread.start()

        # Основной цикл
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

        self.capture_thread.join()
        self.analysis_thread.join()
        logging.info('Процесс остановлен.')

    def capture_thread_func(self):
        server_url = f'srt://{self.server_ip}:{self.server_port}?mode=listener&encryption=encrypt&pbkeylen=16&passphrase={self.passphrase}'
        output_pattern = os.path.join(self.segment_directory, 'output_%03d.mp4')
        
        # Формирование команды ffmpeg
        cmd = [
            'ffmpeg',
            '-i', server_url,
            '-c', 'copy',
            '-f', 'segment',
            '-segment_time', str(self.segment_duration),
            '-reset_timestamps', '1',
            '-segment_wrap', '10',
            output_pattern
            ]

        logging.info(f"Запуск ffmpeg с командой: {' '.join(cmd)}")

        try:
            # Запуск процесса ffmpeg
            ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Ожидание, пока is_running не станет False
            while self.is_running:
                time.sleep(0.1)

            # При остановке завершаем процесс ffmpeg
            ffmpeg_process.terminate()
            ffmpeg_process.wait()
        except Exception as e:
            logging.error(f'Ошибка в потоке захвата: {e}')

    def analysis_thread_func(self):
        while self.is_running:
            try:
                # Список файлов сегментов
                segment_files = glob.glob(os.path.join(self.segment_directory, 'output_*.mp4'))

                # Если меньше тре файлов, то ждем
                if len(segment_files) >= 3:
                    segment_files.sort(key=os.path.getmtime, reverse=True)
                    file_to_process = segment_files[2] # ffmpeg пишет в последние двай файла одновременно, поэтому берем третий

                    # Проверить, был ли файл уже обработан
                    mtime = os.path.getmtime(file_to_process)
                    if mtime > self.last_processed_time:
                        self.process_segment_file(file_to_process)
                        self.last_processed_time = mtime
                    else:
                        pass

                time.sleep(0.1)
            except Exception as e:
                logging.error(f'Ошибка в потоке анализа: {e}')

    def process_segment_file(self, segment_file):
        logging.info(f'Обработка файла сегмента: {segment_file}')

        try:
            container = av.open(segment_file)
            video_stream = None
            audio_stream = None

            # Поиск видео и аудио потоков
            for stream in container.streams:
                if stream.type == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.type == 'audio':
                    if audio_stream is None:
                        audio_stream = stream

            # Сбор кадров
            video_frames = []
            audio_frames = []

            for packet in container.demux():
                if packet.stream == video_stream:
                    for frame in packet.decode():
                        video_frames.append(frame)
                elif packet.stream == audio_stream:
                    for frame in packet.decode():
                        audio_frames.append(frame)

            # Анализ собранных кадров
            self.analyze_data(video_frames, audio_frames)

            container.close()
        except Exception as e:
            logging.error(f'Ошибка при обработке файла {segment_file}: {e}')

    def analyze_data(self, video_frames, audio_frames):
        try:
            # Анализ видео данных
            for frame in video_frames:
                frame_array = frame.to_ndarray(format='bgr24')
                emotion, prob, region = self.video_processor.face_emotion(frame_array)
                if emotion and prob and (region['x'] != 0) and (region['y'] != 0):
                    logging.info(f'Видео: {emotion}, {prob:.2f}, {region}')

            # Анализ аудио данных
            all_audio_samples = []
            sampling_rate = None
            for frame in audio_frames:
                audio_data = frame.to_ndarray()
                sampling_rate = frame.sample_rate
                all_audio_samples.append(audio_data)

            if all_audio_samples:
                combined_audio_data = np.concatenate(all_audio_samples, axis=1)
                emotion, prob = self.audio_processor.speech_emotion(combined_audio_data, sampling_rate)
                if emotion and prob:
                    logging.info(f'Аудио: {emotion}, {prob:.2f}')

                emotion, prob, text = self.text_processor.text_emotion(combined_audio_data, sampling_rate)
                if emotion and prob:
                    logging.info(f'Текст: {emotion}, {prob:.2f}, {text}')

        except Exception as e:
            logging.error(f'Непредвиденная ошибка при анализе данных: {e}')

    def stop(self):
        self.is_running = False