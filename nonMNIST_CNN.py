# Zenen Treadwell -- Western University -- Student Number 250847963 -- CS4442
# Convolutional Neural Network to classify non-MNIST text data from images
# April 10th, 2019

import tensorflow as tf
import numpy as np
import random
import keras
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Loading image data paths into a pictures_file dictionary, grouped by letter
data_folder = "./data/notMNIST_small"
letters = os.listdir(data_folder)

pictures_file = {}
for letter in letters:
    images = [name for name in os.listdir("{}/{}/".format(data_folder,letter)) if (name[-4:] == ".png")]
    pictures_file[letter] = images

# Initializing data containers
X = []
labels = []
entries = 0

# Importing image data as a collection of 2D arrays
for letter in letters:
    print("\nImporting {}...".format(letter))
    for name in pictures_file[letter]:
        try:
            img = plt.imread(data_folder+"/{}/{}".format(letter, name))
            X.append(img)
            labels.append(letter)
            entries += 1
        except Exception as e:
            print("{} occured processing this file: {}".format(e, name))
    print("Finished")

print("All done!")

# Save image dimension
img_dim=len(X[0])

# Shape image array
X = np.asarray(X).reshape(entries, img_dim, img_dim, 1)

# Encode labels as one shot
labels = keras.utils.to_categorical(list(map(lambda x: ord(x) - ord('A'), labels))) 

# Shuffle values to avoid alphabetization
zipped = list(zip(X, labels))
np.random.shuffle(zipped)
X, labels = zip(*zipped)
X = np.asarray(X)
labels = np.asarray(labels)

# Function to display the first 15 entries in a dataset
def show_sample(X, title):
    num_rows, num_cols = 3, 5
    fig, axes = plt.subplots(num_rows, num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i, j].imshow(X[num_cols*i + j, :, :, 0], cmap='gray')
            axes[i, j].axis('off')
    fig.suptitle(title)
    plt.show()

show_sample(X, 'Raw Images from Dataset')

# This program will preprocess images and expand them by the following factor
RESIZE_FACTOR = 1.5
dim_out = int(img_dim * RESIZE_FACTOR)

# Image preprocessing step, taken from github tutorial:
# https://github.com/ageron/handson-ml/blob/master/13_convolutional_neural_networks
def prepare_image(image, target_width = dim_out, target_height = dim_out, max_zoom = 0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
        
    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)
    
    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    
    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Now, let's resize the image to the target dimensions.
    # The resize function of scikit-image will automatically transform the image to floats ranging from 0.0 to 1.0
    image = resize(image, (target_width, target_height))
    
    # Finally, let's ensure that the colors are represented as 32-bit floats:
    return image.astype(np.float32)


X_resized = []

# Using 512 Preprocessed images during development because my computer has no GPU, comment out next line for full run
#entries = 512
X = X[:entries]
labels = labels[:entries]

print("Preprocessing... this could take a while.\n")
for img in X:
    X_resized.append(prepare_image(img, dim_out, dim_out))

# Replace the dataset with the new resized version
X = np.asarray(X_resized).reshape(entries, dim_out, dim_out, 1)
show_sample(X, "Preprocessed Images")

# Splitting data into testing and training sets
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2)

# Model Generation using Keras based on https://www.kaggle.com/volperosso/simple-cnn-classifier-on-notmnist

## Hyperparameters ##

# SELU is used as the default activation function in accordance with this web article:
# https://towardsdatascience.com/gentle-introduction-to-selus-b19943068cd9

# A default Dropout value of 0.5 was chosen in accordance with this web article:
# https://towardsdatascience.com/a-walkthrough-of-convolutional-neural-network-7f474f91d7bd

# A value of 50 Epochs was chosen after finding that the initial value of 40 left room for improvement

# A batch size of 64 was chosen as the "standard" batch size for MNIST algorithms

# AdaMax was used as the optimization function alongside a loss function of categorical crossentropy
# in accordance with this paper: https://arxiv.org/pdf/1412.6980.pdf

## Overall Structure ##

# The following model consists of:
# One Input layer with built-in Dropout
# One Biased Convolution/Pooling block, consisting of 2 convolutional layers, one pooling layer and a dropout layer
# One Unbiased Convolution/Pooling block, consisting of 2 convolutional layers, one pooling layer and a dropout layer
# One Identification block, consisting of a data flattener and 2 fully connected layers implementing dropout.
# One Output layer, with 1 node for each category in the original dataset

# Details such as kernel size and total parameters are output using the model.summary() command
# Layer sizes and batch sizes are all exponentiations of 2 in accordance with CPU architecture

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Intake
image_input = Input(shape = X[0].shape, name="input")
path = Dropout(0.2, name='dropout')(image_input)

# Conv/Pooling Block 1
path = Conv2D(8, (3,3), activation='selu', name="CP1_Conv1")(path)
path = Conv2D(16, (3,3), activation='selu', name="CP1_Conv2")(path)
path = MaxPooling2D((2,2), name="CP1_Pool")(path)
path = Dropout(0.5, name="CP1_Drop")(path)

# Conv/Pooling Block 2
path = Conv2D(16, (3,3), use_bias=False, activation='selu', name="CP2_Conv1")(path)
path = Conv2D(32, (3,3), use_bias=False, activation='selu', name="CP2_Conv2")(path)
path = MaxPooling2D((2,2), name="CP2_Pool")(path)
path = Dropout(0.5, name="CP2_Drop")(path)

# Identification Block
path = Flatten(name='ID_Flatten')(path)
path = Dense(64, activation='selu', name="ID_Dense1")(path)
path = Dropout(0.5, name="ID_Drop")(path)
path = Dense(64, activation='selu', name="ID_Dense2")(path)
path = Dropout(0.5, name="ID_Drop2")(path)

# Output
output = Dense(len(letters), activation='softmax', name="output")(path)

# Generate the model
model = Model(image_input,output)
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
csv_training_logger = keras.callbacks.CSVLogger("training.csv", append=False)
model.fit(X_train, labels_train, epochs=45, batch_size=64, validation_data=[X_test, labels_test], callbacks=[csv_training_logger])

## Training process ##
# This model was difficult to train on due to its complexity and the limited processing power of my laptop. Therefore,
# I adjusted the kernel sizes to be half of the values suggested by the guide I used. Much of the testing was done using
# A small dataset of 100 processed images, but this resulted in low accuracy values and did not allow for much optimization.
# To compensate, I researched the optimal values and parameters for MNIST algorithms and implemented them in this program.
# The final testing run for this algorithm took 33 minutes to compute and resulted in a reported accuracy of 86.36%

# Displays the accuracy of the model over time
def graph_accuracy():
    data = np.genfromtxt('training.csv', delimiter=',')
    data = data[1:][:,1:]
    plt.plot(data[:,0]) # training accuracy
    plt.plot(data[:,2]) # testing accuracy
    plt.legend(['Training','Testing'])
    plt.title("Accuracy by Epoch")
    plt.xlabel("epoch")
    plt.ylim(0.0,1.0)
    plt.show()

#graph_accuracy()

# Evalute Accuracy
score = model.evaluate(X_test, labels_test, verbose=False)
print('Loss: {}'.format(score[0]))
print('Accuracy: {}%'.format(np.round(10000*score[1])/100))
