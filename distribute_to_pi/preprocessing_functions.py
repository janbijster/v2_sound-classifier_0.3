import librosa
import math
import numpy as np


def extract_features(audio_input, n_mfcc=40):
    # This functions discards the first 2 coefficients.
    # The reason for this is that these are respectively very low and very high
    # and constant throughout the sample, so they do not seem to contain much information.
    # See also point 5. of this answer:
    # https://dsp.stackexchange.com/questions/6499/help-calculating-understanding-the-mfccs-mel-frequency-cepstrum-coefficients#answer-6500
    try:
        if type(audio_input) is str:
            audio, sample_rate = librosa.load(audio_input, res_type='kaiser_best')
        elif type(audio_input) is np.ndarray:
            audio = audio_input
            sample_rate = 22050
        else:
            print('Unknown type {} as audio input. Either filename string r numpy array required'.format(type(audio_input)))
            
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        # average over left right channels
        if len(spectrogram.shape) == 3:
            print('stereo sample, averaging to mono...')
            spectrogram = np.mean(spectrogram, axis=0)
        # convert to db:
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        # scale to 0-1:
        spectrogram_db = 1 + spectrogram_db / 80
        
    except Exception as e:
        print('Error encountered while parsing file {}: {}'.format(audio_input, e))
        return None 
     
    return spectrogram_db[:,:]

def pad_spectrogram(spectrogram, required_size, padding_type='wrap'):
    return_spectrogram = np.copy(spectrogram)
    if padding_type == 'wrap':
        while return_spectrogram.shape[1] < required_size:
            return_spectrogram = np.hstack((return_spectrogram, spectrogram))
    elif padding_type == 'nearest':
        padding_values = spectrogram[:, -1].reshape((-1, 1))
        padding_array = np.repeat(padding_values, required_size - spectrogram.shape[1], axis=1)
        return_spectrogram = np.hstack((return_spectrogram, padding_array))
    else:
        print('Warning: unknown padding type {}'.format(padding_type))
        return_spectrogram = spectrogram
    return return_spectrogram[:, :required_size]

def cut_spectrogram(spectrogram, required_size, overlap_fraction):
    if spectrogram.shape[1] < required_size:
        print('Warning: sample of length {} too short to cut.'.format(spectrogram.shape[1]))
        return [spectrogram]
    overlap_length = round(overlap_fraction * required_size)
    hop_size = required_size - overlap_length
    num_cuts = math.ceil((spectrogram.shape[1] - required_size) / hop_size)
    print(num_cuts)
    spectrograms = []
    for i in range(num_cuts):
        start = i*hop_size
        end = start + required_size
        spectrograms.append(spectrogram[:, start:end])
    return spectrograms
