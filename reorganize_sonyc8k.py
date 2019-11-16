# script to reorganize the dataset, from folds to category dirs

import os, sys, csv, glob
import argparse
from shutil import copyfile


# default params
input_folder = '../v2_sound-classifier-0.2/data/sonyc-8k/UrbanSound8K/audio'
metadata_path = '../v2_sound-classifier-0.2/data/sonyc-8k/UrbanSound8K/metadata/UrbanSound8K.csv'

output_folder = './data/audio/'

# script
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--meta', help='Path to the metadata csv file containing slice_file_name and class columns. Default: {}'.format(metadata_path))
parser.add_argument('-i', '--input', help='Input data folder containing the folds. Default: {}'.format(input_folder))
parser.add_argument('-o', '--output', help='Output data folder. Default: {}'.format(output_folder))
args = parser.parse_args()

if args.meta:
    metadata_path = args.meta
if args.input:
	input_folder = args.input
if args.output:
	output_folder = args.output

if not os.path.exists(input_folder):
	raise ValueError('Input folder {} doesn\'t exist.'.format(input_folder))
if not os.path.exists(output_folder):
	os.makedirs(output_folder)

# read csv
class_dict = {}
with open(metadata_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_dict[row['slice_file_name'][:-4]] = row['class']

# traverse folder
for root, dirs, files in os.walk(input_folder):
    for folddirname in sorted(dirs):
        for file in glob.glob(os.path.join(input_folder, folddirname + '/*')):
            filename = os.path.split(file)[-1]
            output_filename = '{}-{}'.format(folddirname, filename)

            # lookup file:
            file_id = filename[:-4]
            try:
                class_name = class_dict[file_id]
            except KeyError:
                print('Couldn\'t find metadata row for file id {} in dir {}, skipping.'.format(file_id, dirname))
                continue

            output_dir = os.path.join(output_folder, class_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, output_filename)
            print('Copying {} to {}'.format(file, output_file))
            copyfile(file, output_file)