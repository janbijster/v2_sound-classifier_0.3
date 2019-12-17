import sounddevice as sd
import time
import numpy as np
import math
from matplotlib import pyplot as plt
from preprocessing_functions import extract_features
from keras.models import load_model
from keras import backend as K
from PIL import Image
from glob import glob
import os
import json
import argparse
import librosa
from PIL import Image
import sys
from utils import datetime_string, audio_volume
try:
    from pydub import AudioSegment
    can_use_pydub = True
except (FileNotFoundError, ModuleNotFoundError) as e:
    print('Either pydub, ffmpeg or libav not installed. Storing sounds as wav.')
    can_use_pydub = False


# constants and default arguments
sample_rate = 22050
sample_length_sec = 4
check_interval = 0.5
default_model_path = 'models/model-with-tire-screech.h5'
default_class_names_file = 'class_names.json'
default_volume_threshold = -30
default_sound_store_location = 'stored/'
trim_recording_start_seconds = 0.2 # trim recording inital click

# computed constants
num_samples = int(sample_length_sec * sample_rate)
trim_recording_start_samples = int(trim_recording_start_seconds * sample_rate)


# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=default_model_path,
    help='The filepath of the .h5 model for spectrogram classification. (default: {})'.format(default_model_path))
parser.add_argument('-c', '--classes', default=default_class_names_file,
    help='The filepath of the json file containing a list of class names. (default: {})'.format(default_class_names_file))
parser.add_argument('-s', '--store', default='n',
    help='Store the recordings that exceeded the volume threshold? ("y"/"n", default: "n")')
parser.add_argument('-t', '--threshold', default=default_volume_threshold,
    help='The volume threshold above which sounds will be stored if store is enabled. (number, default: {})'.format(default_volume_threshold))
parser.add_argument('-l', '--location', default=default_sound_store_location,
    help='The location where to store sounds if store is enabled. (default: {})'.format(default_sound_store_location))
args = parser.parse_args()
store_sounds = args.store.lower() == 'y'
# get class names
with open(args.classes) as f:
    class_names = json.load(f)


# functions
def start_recording(event, force=False):
    global store_sounds, store_session, class_names, args
    if not store_sounds or force:
        store_session = {}
        store_session['start_datetime'] = datetime_string()
        store_session['name'] = 'session_{}'.format(store_session['start_datetime'])
        store_session['folder'] = os.path.join(args.location, store_session['name'])
        store_session['json_filepath'] = os.path.join(store_session['folder'], 'index.json')
        store_session['volume_threshold'] = args.threshold
        store_session['class_names'] = class_names
        store_session['sounds'] = []

        if not os.path.exists(args.location):
            os.makedirs(args.location)
        if not os.path.exists(store_session['folder']):
            os.makedirs(store_session['folder'])
        
        store_sounds = True


def record_function(sd, num_samples, recordings):
    current_recording = sd.rec(num_samples + trim_recording_start_samples)
    sd.wait()
    current_recording = current_recording[trim_recording_start_samples:, :]
    # if there is already a recording, then store also the overlap with the previous one:
    if len(recordings) != 0:
        previous_recording = recordings[-1]
        overlap_samples = int(0.5 * num_samples)
        overlap_recording = np.vstack((previous_recording[overlap_samples:], current_recording[:overlap_samples]))
        recordings = [overlap_recording, current_recording]
    else:
        recordings = [current_recording]
    return recordings


# main
def main():
    # settings & initilaization
    sd.default.samplerate = sample_rate
    sd.default.channels = 1

    # For GPU training only
    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = False  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

    # load model
    model = load_model(args.model)
    # get class names
    with open(args.classes) as f:
        class_names = json.load(f)

    if store_sounds:
        start_recording(None, force=True)


    # start recording
    recordings = []

    while True:
        recordings = record_function(sd, num_samples, recordings)
        
        for recording_index, recording in enumerate(recordings):
            # process both the new recording and the overlap with the previous
            # get spectrogram:
            spectrogram = extract_features(recording.flatten())
            # resize to 128x128:
            spectrogram_array = np.array(Image.fromarray(spectrogram).resize((128, 128))).reshape((1, 128, 128, 1))
            
            predictions = model.predict_proba(spectrogram_array)[0].tolist()
            class_predictions_sorted = [(name, proba) for proba, name in sorted(zip(predictions, class_names), reverse=True)]
            
            volumes = audio_volume(recording.flatten())

            if store_sounds:
                if volumes.max() > args.threshold:
                    sound_info = {}
                    sound_info['predicted_class'] = class_predictions_sorted[0][0]
                    sound_info['probabilities'] = predictions
                    sound_info['predictions_sorted'] = class_predictions_sorted
                    sound_info['name'] = '{}_{}'.format(datetime_string(recording_index * sample_length_sec), sound_info['predicted_class'])
                    
                    # save sound file:
                    sound_file = os.path.join(store_session['folder'], sound_info['name'])
                    sound_info['audio_filepath'] = sound_file + '.wav'
                    librosa.output.write_wav(sound_info['audio_filepath'], recording, sample_rate)
                    # convert to mp3 if pydub (and ffmpeg or libav) is installed:
                    if can_use_pydub:
                        audio_segment = AudioSegment.from_file(sound_info['audio_filepath'], format='wav')
                        os.remove(sound_info['audio_filepath'])
                        sound_info['audio_filepath'] = sound_file + '.mp3'
                        audio_segment.export(sound_info['audio_filepath'], format='mp3')
                    
                    store_session['sounds'].append(sound_info)
                    with open(store_session['json_filepath'], 'w') as f:
                        json.dump(store_session, f, indent=4)

            # visualize only the new recording
            if recording_index == len(recordings) - 1:
                os.system('cls' if os.name == 'nt' else 'clear')
                top_k = 5
                print('#### Top {} ####'.format(top_k))
                for i in range(top_k):
                    print('{}) {:.0%}: {}'.format(i, class_predictions_sorted[i][1], class_predictions_sorted[i][0]))

                print('\nAverage volume: {:.0f}, peak volume: {:.0f} (recording threshold: {})'.format(
                    volumes.mean(), volumes.max(), args.threshold
                ))
                print('\nRecording.' if store_sounds else 'Not recording.')
                print('\n(Press ctrl+c to quit)')

if __name__ == "__main__":
    main()