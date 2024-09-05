from fileinput import filename
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

def convert_to_multichannel(single_channel_file, output_file, num_channels=4):
    # load single-channel audio file
    single_channel_signal, sr = librosa.load(single_channel_file, sr=None, mono=True)

    # Duplicate the single channel across the desired number of channels
    multichannel_signal = np.tile(single_channel_signal, (num_channels, 1))

    # Save the multichannel signal to a new file
    sf.write(output_file, multichannel_signal.T, sr)


