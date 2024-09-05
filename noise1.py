import numpy as np
import librosa
import soundfile as sf

# add environmental noise and ego noise
def add_noise_to_multichannel_signal(Env_file, noise_file, snr_db):
    # Load the speech and noise files
    Env, sr = librosa.load(Env_file, sr=None, mono=False)
    noise, _ = librosa.load(noise_file, sr=sr, mono=False)

    # ensure the noise files have same length
    if len(noise) < Env.shape[1]:
        noise = np.tile(noise, int(np.ceil(Env.shape[1] / len(noise))))
    noise = noise[:Env.shape[1]]

    # calculate the power of the both noise
    Env_power = np.mean(Env ** 2, axis=1)
    noise_power = np.mean(noise ** 2)
    # calculate the required noise power for the desired SNR
    desired_noise_power = Env_power / (10 ** (snr_db / 10))
    # scale the noise to achieve the desired noise power
    scaling_factors = np.sqrt(desired_noise_power / noise_power)
    # Add the ego noise to Env noise
    noisy_stream = Env + (noise * scaling_factors[:, np.newaxis])

    return noisy_stream, sr

