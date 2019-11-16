import os
import random
import shutil
from training_functions import train_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import confusion_matrix, hamming_loss
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np

# dimensions of our images.
img_width, img_height = 128, 128
# default data folder
train_data_dir = './data/spectrograms/'
validation_data_dir = './data/spectrograms-test/'
temp_data_dir = './data/spectrograms-temp/'

models_folder = './models'
model_name = 'model-fraction-{}'

sample_fractions = [0.2]

# script
if not os.path.exists(temp_data_dir):
    os.makedirs(temp_data_dir)

# prepare validation dataset for evaluation (stays the same for each fraction)
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255)
validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

num_batches = int(validation_generator.samples / batch_size)
class_names = list(validation_generator.class_indices.keys())

# now for each fraction...
losses = []
num_samples_per_fraction = []
for fraction in sample_fractions:
    num_training_samples = 0
    # populate temp folder with a random subset of the data:
    for dir_entry in [f for f in os.scandir(train_data_dir) if f.is_dir()]:
        
        temp_category_dir = os.path.join(temp_data_dir, dir_entry.name)
        # clear existing files
        if os.path.exists(temp_category_dir):
            shutil.rmtree(temp_category_dir)
        os.makedirs(temp_category_dir)

        # loop over files
        files = [f for f in os.scandir(dir_entry.path) if not f.is_dir()]
        num_files = round(fraction * len(files))
        files_to_copy = random.sample(files, num_files)

        for file_entry in files_to_copy:
            shutil.copy(file_entry.path, os.path.join(temp_category_dir, file_entry.name))
            num_training_samples += 1
    
    # train model on the data subset:

    _ = train_model(
        train_data_dir=temp_data_dir,
        validation_data_dir=validation_data_dir,
        image_dimensions=(img_width, img_height),
        model_name=model_name.format(fraction),
        models_folder=models_folder,
        num_epochs=max(round(fraction * 50), 15)
    )

    # load model (because latest model may have collapsed, not necessarily the best)
    model = load_model(os.path.join(models_folder, model_name.format(fraction) + '.h5'))

    # evaulate
    y_total = []
    y_pred_total = []

    for i in range(num_batches):
        [X, Y] = validation_generator.next()
        Y_pred = model.predict(X)
        # one hot encoding to class indices:
        y = np.argmax(Y, 1)
        y_pred = np.argmax(Y_pred, 1)
        y_total += list(y)
        y_pred_total += list(y_pred)

    # plot confusion matrix:
    conf_matrix = confusion_matrix(y_total, y_pred_total)

    # normalize:
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=class_names, yticklabels=class_names,
        title='Confusion matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    
    loss = hamming_loss(y_total, y_pred_total)
    title = 'Fraction: {}, total # of samples: {}, Hamming loss: {}'.format(fraction, num_training_samples, loss)
    print(title) 

    ax.set_title(title)
    plt.show()

    losses.append(loss)
    num_samples_per_fraction.append(num_training_samples)


plt.plot(num_samples_per_fraction, losses)
plt.title('Hamming loss vs # samples used')
plt.show()