import numpy as np
import librosa

# Noise Injection.
def inject_noise(data, sampling_rate = 0.035, threshold = 0.075, random = False):
    if random:
        sampling_rate = np.random.random() * threshold
    noise_amplitude = sampling_rate * np.random.uniform() * np.amax(data)
    augmented_data = data + noise_amplitude * np.random.normal(size = data.shape[0])
    return augmented_data

# Pitching.
def pitching(data, sampling_rate, pitch_factor = 0.7,random = False):
    if random:
        pitch_factor= np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y = data, sr = sampling_rate, n_steps = pitch_factor)

# Stretching.
def stretching(data,r = 0.9):
    return librosa.effects.time_stretch(y = data, rate = r)

# Pipeline function that applies all the audio data augmentation functions we just built.
def pipeline(data, sampling_rate):
    data = pitching(data, sampling_rate, random = True)
    data = inject_noise(data, random = True)
    data = stretching(data)
    return data