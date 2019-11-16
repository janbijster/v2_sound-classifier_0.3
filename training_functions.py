# copied from https://github.com/AvinashNath2/Image-Classification-using-Keras/blob/master/Small_Conv_Net.py

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import os


# For GPU training only
if True:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def create_model(img_width, img_height, num_categories):
    # Small Conv Net
    # a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_categories))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

    model.summary()
    return model


def train_model(train_data_dir, validation_data_dir, image_dimensions, model_name, models_folder, batch_size=32, num_epochs=50):
    ## parse and check arguments
    for dir in [train_data_dir, validation_data_dir]:
        if not os.path.exists(dir):
            raise FileNotFoundError('Invalid data path: {}'.format(dir))
    
    img_width, img_height = image_dimensions

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    ## preprocessing
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    datagen = ImageDataGenerator(rescale=1./255)

    # automagically retrieve images and their classes for train and validation sets
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    ## See if a model is already present, if so load it and continue training
    model_path = os.path.join(models_folder, model_name) + '.h5'
    if os.path.isfile(model_path):
        print('Found existing model, loading...')
        model = load_model(model_path)
    else:
        # if not, create new model
        print('No model with this name found. Creating new...')
        class_names = list(set([f.split('/')[0] for f in validation_generator.filenames]))
        model = create_model(img_width, img_height, len(class_names))
    
    ## Train
    train_samples = 2048
    validation_samples = 832

    # save the model after each epoch if improved
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=[checkpointer])

    return model