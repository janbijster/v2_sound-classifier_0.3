import sounddevice as sd
import time
import numpy as np
import math
from matplotlib import pyplot as plt
from preprocessing_functions import extract_features
from keras.models import load_model
from PIL import Image
from glob import glob
import os
import json
from visualization import initialize_visualizations, update_visualizations

# params
default_sample_rate = 22050
default_sample_length_sec = 4
check_interval = 0.5
model_path = 'models/model-with-tire-screech.h5'
data_path = 'data/spectrograms'
class_names_file = 'class_names.json'

# computed constants
num_samples = int(default_sample_length_sec * default_sample_rate)
overlap_samples = int(0.5 * num_samples)

# functions
def num_nonzero_elements(arr):
    return len(np.nonzero(arr)[0])

def is_filled(arr):
    return num_nonzero_elements(arr) == len(arr)

# script
# settings & initilaization
sd.default.samplerate = default_sample_rate
sd.default.channels = 1
# plotting
fig, axes = initialize_visualizations()

# For GPU training only
if True:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
# load model
model = load_model(model_path)
# get class names
with open(class_names_file) as f:
    class_names = json.load(f)


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

        # process
        # get spectrogram:
        spectrogram = extract_features(recording.flatten())
        # resize to 128x128:
        X = np.array(Image.fromarray(spectrogram).resize((128, 128))).reshape((1, 128, 128, 1))
        
        predictions = model.predict_proba(X)[0].tolist()
        class_predictions_sorted = [(name, proba) for proba, name in sorted(zip(predictions, class_names), reverse=True)]

        update_visualizations(
            fig, axes,
            recording, X,
            predictions, class_names,
            0, default_sample_length_sec
        )

        os.system('cls' if os.name == 'nt' else 'clear')
        print('#### Top 3 ####')
        for i in range(3):
            print('{}) {:.0%}: {}'.format(i, class_predictions_sorted[i][1], class_predictions_sorted[i][0]))

        print('')

