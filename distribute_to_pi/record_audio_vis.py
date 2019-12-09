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
from visualization import initialize_visualizations, update_visualizations
from datetime import datetime
import librosa
from PIL import Image


# constants and default arguments
sample_rate = 22050
sample_length_sec = 4
check_interval = 0.5
default_model_path = 'models/model-with-tire-screech.h5'
default_class_names_file = 'class_names.json'
default_volume_threshold = -30
default_sound_store_location = 'stored/'
# computed constants
num_samples = int(sample_length_sec * sample_rate)
overlap_samples = int(0.5 * num_samples)


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
    help='The location where to store sounds if store is enaabled. (default: {})'.format(default_sound_store_location))
args = parser.parse_args()
store_sounds = args.store.lower() == 'y'


# functions
def num_nonzero_elements(arr):
    return len(np.nonzero(arr)[0])

def is_filled(arr):
    return num_nonzero_elements(arr) == len(arr)

def datetime_string():
    return datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def audio_volume (recording, smoothing_n=1000):
    volumes = 20 * np.log10(np.abs(recording) + 0.01)
    if smoothing_n > 0:
        volumes = running_mean(volumes, smoothing_n)
    return volumes.flatten()


# main
def main():
    # settings & initilaization
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    # plotting
    fig, axes = initialize_visualizations()

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


    # start recording
    previous_recording = None
    overlap_recording = None
    current_recording = None

    while True:
        previous_recording = current_recording
        current_recording = sd.rec(num_samples)
        sd.wait()

        recordings = [current_recording]
        if previous_recording is not None:
            overlap_recording = np.vstack((previous_recording[overlap_samples:], current_recording[:overlap_samples]))
            recordings = [overlap_recording, current_recording]
        
        for recording in recordings:
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
                    sound_info['name'] = '{}_{}'.format(datetime_string(), sound_info['predicted_class'])
                    sound_info['audio_filepath'] = os.path.join(store_session['folder'], '{}.wav'.format(sound_info['name']))
                    librosa.output.write_wav(sound_info['audio_filepath'], recording, sample_rate)
                    
                    store_session['sounds'].append(sound_info)
                    with open(store_session['json_filepath'], 'w') as f:
                        json.dump(store_session, f, indent=4)


        # visualize only the new recording
        update_visualizations(
            fig, axes,
            volumes, spectrogram_array,
            predictions, class_names,
            args.threshold, sample_length_sec
        )

        os.system('cls' if os.name == 'nt' else 'clear')
        print('#### Top 3 ####')
        for i in range(3):
            print('{}) {:.0%}: {}'.format(i, class_predictions_sorted[i][1], class_predictions_sorted[i][0]))

        print('')

if __name__ == "__main__":
    main()