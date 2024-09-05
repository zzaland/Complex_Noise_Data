
from pathlib import Path
import soundfile as sf
import P0   as p0
import Reverberate as p3
import noise1 as n1
import noise2 as n2
#######################################################################################
# step 1: conversion of single channel data to multichannel if needed
dir_path = Path('openslrlibrispeech_asr/train-clean-100')

# file type for input file
files = list(dir_path.rglob('*.flac'))
max= 10 # maximum number of files to convert
num_channels=2 # number of output channels
num=0
for file in files:
    num=num+1
    single_channel_file = file;
    output_file = 'CND/clean/MC'+str(num)+'.wav' # output file name and location

    if num> max:
        break
    p0.convert_to_multichannel(single_channel_file, output_file, num_channels)

#################################################################################################
# step 2: creating reverberated speech

# room dimentions for impulse response can be changed from Reverberate.py by changing
# Source position, room size, and microphone position
dir_path = Path('CND/Clean') # source for clean speech input

# get a list of files in the directory
files = list(dir_path.glob('*.wav'))  # choose file type
RT= 0.7 # reverberation time in milliseconds


num = 0
for file in files:
    num += 1
    p3.apply_reverb_multichannel(file, f'CND/reverb/700/MC_reverb{num}.wav', rt60=RT)
    if num >= max:
        break

#################################################################################################
#Step 3: create a noise stream from different noise types

Noise1 = 'DRONE_001.wav' # sample noise type 1 ego noise from drone
Noise2 = 'ch02.wav' # Sample environmental noise
snr_db = 0  # the SNR value at which the two noises are mixed
            # change values if you want one type of noise to be dominant in the stream

noisy_stream, sr = n1.add_noise_to_multichannel_signal(Noise1, Noise2, snr_db)

# Save the noisy stream to a new file
sf.write('exp-1/noisy_stream2_SNR_0.wav', noisy_stream.T, sr)

#################################################################################################
#Step 4: adding Noise to reverberated speech
dir_path = Path('CND/reverb/700')
files = list(dir_path.glob('*.wav'))  # file types to be fetched from directory

num = 0
for file in files:
    num += 1
    speech_file = file
    noise_file = 'exp-1/noisy_stream2_SNR_0.wav'
    snr_db = 15  # Desired SNR in dB
    noisy_speech, sr = n2.add_noise_to_multichannel_signal(speech_file, noise_file, snr_db)
    # Save the noisy speech to a new file
    sf.write(f'CND/noisy/700/train/15/MC_noise{num}.wav', noisy_speech.T, sr)
#################################################################################################
print ('complex noisy data created successfully....')