# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from dataset import Dataset
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, TimeDistributedDense, \
                              RepeatVector, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils.generic_utils import Progbar
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.recurrent import GRU, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.callbacks import ModelCheckpoint

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MAIN_DIR = "/media/aj/grab/ml/datasets/10k_cats"
DIRS = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
DIRS = [os.path.join(MAIN_DIR, subdir) for subdir in DIRS]

MODEL_IMAGE_HEIGHT = 128
MODEL_IMAGE_WIDTH = 128

def main():
    """Main method that reads the images, trains a model, then saves weights and predictions."""
    parser = argparse.ArgumentParser(description="Train a model to locate cat faces in images.")
    parser.add_argument("--path", required=True, help="Path to your 10k cats dataset directory")
    args = parser.parse_args()
    
    # initialize dataset
    
    # load images
    
    # convert to X, y
    
    # split train and val
    
    # create model
    
    # fit
    
    # save weights
    
    # save predictions on val set


def create_model_tiny(image_height, image_width, optimizer):
    """Creates the tiny version of the cat face locator model.
    This is useful for debugging, because it doesn't take as much theano compile time.
    
    Args:
        image_height: The height of the input images.
        image_width: The width of the input images.
        optimizer: Keras optimizer to use, e.g. Adam() or "sgd".
    Returns:
        Sequential
    """

    model = Sequential()
    
     # 3x64x64
    model.add(Convolution2D(4, 1 if GRAYSCALE else 3, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(4, 4, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # 4x15x15
    new_image_height = (((image_height - 2) / 2) - 2) / 2
    new_image_height = int(new_image_height)
    
    new_image_width = (((image_width - 2) / 2) - 2) / 2
    new_image_width = int(new_image_width)
    
    nb_last_kernels = 4
    flat_size = nb_last_kernels * new_image_height * new_image_width
    
    model.add(Flatten())
    
    model.add(Dense(flat_size, 64))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, 64))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    
    #model.add(Dense(64, 4))
    model.add(Dense(64, 8))
    model.add(Activation("sigmoid"))
    
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    return model

def create_model(image_height, image_width, optimizer):
    """Creates the cat face locator model.
    
    Args:
        image_height: The height of the input images.
        image_width: The width of the input images.
        optimizer: Keras optimizer to use, e.g. Adam() or "sgd".
    Returns:
        Sequential
    """
    
    model = Sequential()
    
     # Tensor size at this point (if 64x64 input): 3x128x128
    model.add(Convolution2D(32, 1 if GRAYSCALE else 3, 3, 3, border_mode="same"))
    model.add(LeakyReLU(0.33))
    model.add(Convolution2D(32, 32, 3, 3, border_mode="same"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Tensor size (...): 32x64x64
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Tensor size (...): 64x30x30
    model.add(Convolution2D(128, 64, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 128, 3, 3, border_mode="valid"))
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Tensor size (...): 128x13x13
    # calculate tensor size
    new_image_height = image_height / 2
    new_image_height = (new_image_height - 2 - 2) / 2
    new_image_height = (new_image_height - 2 - 2) / 2
    new_image_height = int(new_image_height)
    new_image_width = new_image_height
    
    nb_last_kernels = 128
    
    # Reshape to timesteps of LSTM and normalize
    model.add(Reshape(nb_last_kernels, new_image_height * new_image_width))
    model.add(BatchNormalization((nb_last_kernels, new_image_height * new_image_width)))
    
    # 1st LSTM layer
    model.add(LSTM(new_image_height * new_image_width, 256, return_sequences=True))
    
    # dropout, normalize
    model.add(BatchNormalization((nb_last_kernels, 256)))
    model.add(Dropout(0.5))
    
    # 2nd LSTM layer
    model.add(LSTM(256, 32, return_sequences=True))
    """
    model.add(Flatten())
    model.add(Dense(nb_last_kernels * 128, 1024))
    model.add(BatchNormalization((1024,)))
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    """
    
    # dropout, normalize and sigmoid to 4 outputs (box center (x, y) and scale (height/2, width/2),
    # all of these values are percentages of max (height/width))
    model.add(Flatten())
    model.add(BatchNormalization(nb_last_kernels * 32))
    model.add(Dropout(0.5))
    #model.add(Dense(nb_last_kernels * 32, 4))
    model.add(Dense(nb_last_kernels * 32, 8))
    #model.add(Dense(1024, 18))
    model.add(Activation("sigmoid"))
    
    # compile with mean squared error
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    
    return model

if __name__ == "__main__":
    main()
