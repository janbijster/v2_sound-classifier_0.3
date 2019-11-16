# selects a random fraction of files and moves them to a different folder to act as a validation set
import os
import random
import shutil

# params
data_folder_training = './data/spectrograms/tire_screech'
data_folder_validation = './data/spectrograms-test/tire-screech'
fraction = 0.1

# script
if not os.path.exists(data_folder_validation):
	os.makedirs(data_folder_validation)

files = [f for f in os.scandir(data_folder_training) if not f.is_dir()]
num_files = round(fraction * len(files))
files_to_copy = random.sample(files, num_files)

for file_entry in files_to_copy:
    os.rename(file_entry.path, os.path.join(data_folder_validation, file_entry.name))