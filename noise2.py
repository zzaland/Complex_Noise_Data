import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Add noise stream to reverberated stream
def add_noise_to_multichannel_signal(speech_file, noise_file, snr_db):
    # Load the speech and noise files
    speech, sr = librosa.load(speech_file, sr=None, mono=False)
    noise, _ = librosa.load(noise_file, sr=sr, mono=True)

    # Ensure same length
    if len(noise) < speech.shape[1]:
        noise = np.tile(noise, int(np.ceil(speech.shape[1] / len(noise))))
    noise = noise[:speech.shape[1]]

    # Calculate power of speech and noise
    speech_power = np.mean(speech ** 2, axis=1)
    noise_power = np.mean(noise ** 2)
    # Calculate the required noise power for the desired SNR
    desired_noise_power = speech_power / (10 ** (snr_db / 10))
    # Scale the noise to the desired noise power
    scaling_factors = np.sqrt(desired_noise_power / noise_power)
    # Add the noise to each channel of the speech signal
    noisy_speech = speech + (noise * scaling_factors[:, np.newaxis])
    return noisy_speech, sr


