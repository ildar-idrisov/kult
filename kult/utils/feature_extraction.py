import numpy as np
import librosa
from .augmentation import inject_noise, pitching, stretching

def zero_crossing_rate(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y = data, frame_length = frame_length, hop_length = hop_length)
    return np.squeeze(zcr)

def root_mean_square(data, frame_length = 2048, hop_length = 512):
    rms = librosa.feature.rms(y = data, frame_length = frame_length, hop_length = hop_length)
    return np.squeeze(rms)


def mel_frequency_cepstral_coefficients(data, sampling_rate, frame_length = 2048, hop_length = 512, flatten:bool = True):
    mfcc = librosa.feature.mfcc(y = data,sr = sampling_rate)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def chroma_stft(data, sampling_rate, frame_length = 2048, hop_length = 512, flatten: bool = True):
    short_time_fourier_transform = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(sr = sampling_rate, S = short_time_fourier_transform)
    return np.squeeze(chroma.T) if not flatten else np.ravel(chroma.T)


def melspectrogram(data, sampling_rate, frame_length = 2048, hop_length = 512, flatten: bool = True):
    melspect = librosa.feature.melspectrogram(y = data, sr = sampling_rate)
    return np.squeeze(melspect.T) if not flatten else np.ravel(melspect.T)


def spectral_centroid(data, sampling_rate, frame_length = 2048, hop_length = 512):
    scentroid = librosa.feature.spectral_centroid(y = data, sr = sampling_rate, n_fft = frame_length, hop_length = hop_length)
    return np.squeeze(scentroid)


def spectral_bandwidth(data, sampling_rate, frame_length = 2048, hop_length = 512):
    sbandwidth = librosa.feature.spectral_bandwidth(y = data, sr = sampling_rate, n_fft = frame_length, hop_length = hop_length)
    return np.squeeze(sbandwidth)


def spectral_rolloff(data, sampling_rate, frame_length = 2048, hop_length = 512):
    srolloff = librosa.feature.spectral_rolloff(y = data, sr = sampling_rate, n_fft = frame_length, hop_length = hop_length)
    return np.squeeze(srolloff)


def spectral_entropyy(data, sampling_rate):
    sentropy = spectral_entropy(x = data, sf = sampling_rate)
    return np.squeeze(sentropy)


def spectral_flux(data, sampling_rate):
    sflux = librosa.onset.onset_strength(y = data, sr = sampling_rate)
    return np.squeeze(sflux)

def feature_extraction(data, sampling_rate, frame_length = 2048, hop_length = 512):
    result = np.array([])
    result = np.hstack((result,
                        zero_crossing_rate(data, frame_length, hop_length),
                        root_mean_square(data, frame_length, hop_length),
                        mel_frequency_cepstral_coefficients(data, sampling_rate, frame_length, hop_length),
#                         chroma_stft(data, sampling_rate, frame_length, hop_length),
#                         melspectrogram(data, sampling_rate, frame_length, hop_length),
#                         spectral_centroid(data, sampling_rate, frame_length, hop_length),
#                         spectral_bandwidth(data, sampling_rate, frame_length, hop_length),
#                         spectral_rolloff(data, sampling_rate, frame_length, hop_length),
#                         spectral_entropyy(data, sampling_rate),
#                         spectral_flux(data, sampling_rate)
                     ))
    return result


def get_features(data, sampling_rate):
    target_sampling_rate = 22050
    if sampling_rate != target_sampling_rate:
        data = librosa.resample(data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
        sampling_rate = target_sampling_rate

    # No audio data augmentation.
    audio_1 = feature_extraction(data, sampling_rate)
#     audio_1.resize((1, 19921))
    audio = np.array(audio_1)

    # Inject Noise.
    noise_audio_1 = inject_noise(data, random = True)
    audio_2 =  feature_extraction(noise_audio_1, sampling_rate)
#     audio_2.resize((1, 19921))
    audio = np.vstack((audio, audio_2))
    
    # Pitching.
    pitch_audio_1 = pitching(data, sampling_rate, random = True)
    audio_3 = feature_extraction(pitch_audio_1, sampling_rate)
#     audio_3.resize((1, 19921))
    audio = np.vstack((audio, audio_3))
    
#     # Stretching.
#     stretch_audio_1 = stretching(data)
#     audio_4 = feature_extraction(stretch_audio_1, sampling_rate)
#     audio_4.resize((1, 19921))
#     audio = np.vstack((audio, audio_4))
    
    # Pitching and Inject Noise.
    pitch_audio_2 = pitching(data, sampling_rate, random = True)
    pitch_noise_audio_1 = inject_noise(pitch_audio_2, random = True)
    audio_5 = feature_extraction(pitch_noise_audio_1, sampling_rate)
#     audio_5.resize((1, 19921))
    audio = np.vstack((audio, audio_5))
    
#     # Stretching and Pitching.
#     stretch_audio_2 = stretching(data)
#     stretch_pitch_audio_1 = pitching(stretch_audio_2, sampling_rate, random = True)
#     audio_6 = feature_extraction(stretch_pitch_audio_1, sampling_rate)
#     audio_6.resize((1, 19921))
#     audio = np.vstack((audio, audio_6))
    
#     # Pitching, Inject Noise, and Stretching.
#     pitch_noise_stretch_audio_1 = pipeline(data, sampling_rate)
#     audio_7 =  feature_extraction(pitch_noise_stretch_audio_1, sampling_rate)
#     audio_7.resize((1, 19921))
#     audio = np.vstack((audio, audio_7))
    
    audio_features = audio

    return audio_features

# Duration and offset act as placeholders because there is no audio in start and the ending of each audio file is noramlly below three seconds.
def get_features_file(file_path, duration = 2.5, offset = 0.6):
    data, sampling_rate = librosa.load(path = file_path, duration = duration, offset = offset)

    # No audio data augmentation.
    audio_1 = feature_extraction(data, sampling_rate)
#     audio_1.resize((1, 19921))
    audio = np.array(audio_1)

    # Inject Noise.
    noise_audio_1 = inject_noise(data, random = True)
    audio_2 =  feature_extraction(noise_audio_1, sampling_rate)
#     audio_2.resize((1, 19921))
    audio = np.vstack((audio, audio_2))
    
    # Pitching.
    pitch_audio_1 = pitching(data, sampling_rate, random = True)
    audio_3 = feature_extraction(pitch_audio_1, sampling_rate)
#     audio_3.resize((1, 19921))
    audio = np.vstack((audio, audio_3))
    
#     # Stretching.
#     stretch_audio_1 = stretching(data)
#     audio_4 = feature_extraction(stretch_audio_1, sampling_rate)
#     audio_4.resize((1, 19921))
#     audio = np.vstack((audio, audio_4))
    
    # Pitching and Inject Noise.
    pitch_audio_2 = pitching(data, sampling_rate, random = True)
    pitch_noise_audio_1 = inject_noise(pitch_audio_2, random = True)
    audio_5 = feature_extraction(pitch_noise_audio_1, sampling_rate)
#     audio_5.resize((1, 19921))
    audio = np.vstack((audio, audio_5))
    
#     # Stretching and Pitching.
#     stretch_audio_2 = stretching(data)
#     stretch_pitch_audio_1 = pitching(stretch_audio_2, sampling_rate, random = True)
#     audio_6 = feature_extraction(stretch_pitch_audio_1, sampling_rate)
#     audio_6.resize((1, 19921))
#     audio = np.vstack((audio, audio_6))
    
#     # Pitching, Inject Noise, and Stretching.
#     pitch_noise_stretch_audio_1 = pipeline(data, sampling_rate)
#     audio_7 =  feature_extraction(pitch_noise_stretch_audio_1, sampling_rate)
#     audio_7.resize((1, 19921))
#     audio = np.vstack((audio, audio_7))
    
    audio_features = audio

    return audio_features