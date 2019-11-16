import os

# params
data_folder_spectrograms = './data/spectrograms-nearest/'
data_folder_validation = './data/spectrograms-nearest-test/'
validation_fold = 'fold10'

# script
if not os.path.exists(data_folder_validation):
	os.makedirs(data_folder_validation)

# loop over categories
for dir_entry in [f for f in os.scandir(data_folder_spectrograms) if f.is_dir()]:
    test_dir = os.path.join(data_folder_validation, dir_entry.name)
    if not os.path.exists(test_dir):
	    os.makedirs(test_dir)
    # loop over files
    for file_entry in [f for f in os.scandir(dir_entry.path) if not f.is_dir()]:
        if validation_fold in file_entry.name:
            os.rename(file_entry.path, os.path.join(test_dir, file_entry.name)) 