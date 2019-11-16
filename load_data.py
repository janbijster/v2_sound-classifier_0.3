import os
from PIL import Image
import csv
import math
import numpy as np
from preprocessing_functions import extract_features, cut_spectrogram, pad_spectrogram

# params
data_folder_audio = './data/audio/'
data_folder_spectrograms = './data/spectrograms/'
data_meta_data_path = './data/metadata.csv'
padding_type = 'wrap'

default_sample_rate = 22050
default_sample_length_sec = 4
default_hop_size = 512
spectrogram_cutting_overlap = 0.5

# computed constants
default_spectrogram_length = math.ceil(default_sample_rate * default_sample_length_sec / default_hop_size)

# script
# load metadata if exists:
if os.path.exists(data_meta_data_path):
    with open(data_meta_data_path) as f:
        reader = csv.DictReader(f)
        meta_data = [row for row in reader]
else:
    meta_data = []

files_processed = set([s['audio_path'] for s in meta_data])

# create output folder
if not os.path.exists(data_folder_spectrograms):
	os.makedirs(data_folder_spectrograms)

# loop over categories
for dir_entry in [f for f in os.scandir(data_folder_audio) if f.is_dir()]:
    category = dir_entry.name

    output_category_folder = os.path.join(data_folder_spectrograms, category)
    if not os.path.exists(output_category_folder):
        os.makedirs(output_category_folder)

    for file_entry in [f for f in os.scandir(dir_entry.path) if not f.is_dir()]:
        if file_entry.path in files_processed:
            print('{} already processed...'.format(file_entry.path))
            continue

        try:
            filename_parts = file_entry.name.split('-')
            fold = filename_parts[0]
            id = '{}-{}-{}-{}'.format(*filename_parts[1:]).split('.')[0]
        except Exception as e:
            fold = 'unknown'
            id = file_entry.name.split('.')[0]

        spectrogram = extract_features(file_entry.path)
        if spectrogram is None:
            continue

        # normalize spectrogram to range 0-255 for saving as image:
        spectrogram = np.clip(spectrogram, 0, 1) * 255

        spectrogram_length = spectrogram.shape[1]
        if spectrogram_length < default_spectrogram_length:
            print('sample too short, wrap-padding right size...')
            spectrogram = pad_spectrogram(spectrogram, default_spectrogram_length, padding_type=padding_type)
        if spectrogram_length > default_spectrogram_length:
            print('sample too long, cutting in parts...')
            spectrograms = cut_spectrogram(spectrogram, default_spectrogram_length, spectrogram_cutting_overlap)
        else:
            spectrograms = [spectrogram]
        
        for i, item in enumerate(spectrograms):
            output_filename = '{}-{}-{}-{}.png'.format(category, fold, id, i)
            output_filepath = os.path.join(output_category_folder, output_filename)
            im = Image.fromarray(item)
            im.convert('RGB').save(output_filepath)
            meta_data_sample = {
                'category': category,
                'fold': fold,
                'id': id,
                'audio_path': file_entry.path,
                'cut_index': i,
                'spectrogram_shape': list(item.shape),
                'spectrogram_path': output_filepath
            }
            meta_data.append(meta_data_sample)

            # save intermediate metadata to allow stop and start
            with open(data_meta_data_path, 'w', newline='') as f:
                fieldnames = list(meta_data_sample.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(meta_data)