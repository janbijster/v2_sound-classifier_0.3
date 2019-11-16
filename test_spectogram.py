from preprocessing_functions import extract_features, cut_spectrogram, pad_spectrogram
import matplotlib.pyplot as plt
import librosa, librosa.display
import numpy as np
import math

# params
test_audio_file = '/media/datashare/Projects/v2-sound/Sound2/data/audio/dog_bark/fold4-72829-3-0-0.wav'
test_audio_file = '/media/datashare/Projects/v2-sound/Sound2/data/audio/gun_shot/fold1-76093-6-1-0.wav'
#test_audio_file = '/media/datashare/Projects/v2-sound/Sound2/data/audio/siren/fold2-159747-8-0-9.wav'

default_sample_rate = 22050
default_sample_length_sec = 4
default_hop_size = 512
spectrogram_cutting_overlap = 0.5

# computed constants
default_spectrogram_length = math.ceil(default_sample_rate * default_sample_length_sec / default_hop_size)

# script
spectrogram = extract_features(test_audio_file)
spectrogram_length = spectrogram.shape[1]
if spectrogram_length < default_spectrogram_length:
    print('sample too short, wrap-padding right size...')
    spectrogram = pad_spectrogram(spectrogram, default_spectrogram_length, 'nearest')
plt.imshow(spectrogram)
plt.colorbar()
plt.show()