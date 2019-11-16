import argparse
from training_functions import train_model

# dimensions of our images.
img_width, img_height = 128, 128
# default data folder
train_data_dir = './data/spectrograms/'
validation_data_dir = './data/spectrograms-test/'
models_folder = './models'
model_name = 'model-with-tire-screech-10'

# script
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Training data folder. Default: {}'.format(train_data_dir))
parser.add_argument('-v', '--validation', help='Validation data folder. Default: {}'.format(validation_data_dir))
parser.add_argument('-n', '--name', help='Model name. Default: {}'.format(model_name))
args = parser.parse_args()
if args.train:
    train_data_dir = args.train
if args.validation:
    validation_data_dir = args.validation
if args.name:
    model_name = args.name

model = train_model(
    train_data_dir=train_data_dir,
    validation_data_dir=validation_data_dir,
    image_dimensions=(img_width, img_height),
    model_name=model_name,
    models_folder=models_folder
)