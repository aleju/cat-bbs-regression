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

SPLIT = 0.1
MODEL_IMAGE_HEIGHT = 128
MODEL_IMAGE_WIDTH = 128
EPOCHS = 50
SAVE_WEIGHTS_FILEPATH = os.path.join(CURRENT_DIR, "cat_face_locator80x80.weights")
SAVE_WEIGHTS_CHECKPOINT_FILEPATH = os.path.join(CURRENT_DIR, "cat_face_locator80x80.best.weights")
AUGMENTATIONS = 5
SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_DIR = os.path.join(CURRENT_DIR, "predictions")

def main():
    """Main method that reads the images, trains a model, then saves weights and predictions."""
    parser = argparse.ArgumentParser(description="Train a model to locate cat faces in images.")
    parser.add_argument("--path", required=True, help="Path to your 10k cats dataset directory")
    args = parser.parse_args()
    
    subdir_names = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
    subdirs = [os.path.join(args.path, subdir) for subdir in subdir_names]
    
    # initialize dataset
    dataset = Dataset(subdirs)
    
    # load images and labels
    X, y = load_Xy()
    
    # split train and val
    nb_train = int(nb_images * (1 - SPLIT))
    nb_val = nb_images - nb_train
    X_train = X[0:nb_train, ...]
    y_train = y[0:nb_train, ...]
    X_val = X[nb_train:, ...]
    y_val = y[nb_train:, ...]
    
    # create model
    print("Creating model...")
    model = create_model(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, "mse", Adam())
    
    # fit
    checkpoint_cb = ModelCheckpoint(SAVE_WEIGHTS_CHECKPOINT_FILEPATH, verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=EPOCHS, validation_split=0.0,
              validation_data=(X_val, Y_val), show_accuracy=False,
              callbacks=[checkpoint_cb])
    
    # save weights
    print("Saving weights...")
    model.save_weights(SAVE_WEIGHTS_FILEPATH, overwrite=SAVE_AUTO_OVERWRITE)
    
    # save predictions on val set
    if SAVE_PREDICTIONS:
        print("Saving example predictions...")
        y_preds = predict_on_images(model, X_val)
        for img_idx, (y, x, half_height, half_width) in enumerate(y_preds):
            img_arr = draw_predicted_rectangle(X_val[img_idx], y, x, half_height, half_width)
            filepath = os.path.join(SAVE_EXAMPLES_DIR, "%d.png" % (img_idx,))
            misc.imsave(filepath, np.squeeze(img_arr))

def load_Xy():
    i = 0
    nb_images = len(dataset.fps) + len(dataset.fps) * AUGMENTATIONS
    X = np.zeros((nb_images, 3, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), dtype=np.float32)
    y = np.zeros((nb_images, 4), dtype=np.float32)
    for image in dataset.get_images():
        image.pad(PADDING) # todo
        augs = image.augment(AUGMENTATIONS, hflip=True, vflip=False, scale_to_percent=(0.9, 1.1), scale_axis_equally=False,
                             rotation_deg=10, shear_deg=0, translation_x_px=5, translation_y_px=5,
                             brightness_change=0.1, noise_mean=0.0, noise_std=0.05))
        
        for aug in [image] + augs:
            aug.unpad(PADDING)
            aug.resize(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
            X[i] = aug.to_array() / 255.0
            
            face_rect = aug.keypoints.get_rectangle()
            face_rect.normalize(aug) # todo in dataset.py auf image object umstellen
            center = face_rect.get_center()
            width = face_rect.get_width()
            height = face_rect.get_height()
            y[i] = [center.y, center.x, height, width]
            i += 1
    
    return X, y

def draw_predicted_rectangle(image_arr, y, x, half_height, half_width):
    image_arr = np.copy(image_arr) * 255
    image_arr = np.rollaxis(image_arr, 0, 3)
    keypoints = np.zeros((9*2,), dtype=np.uint16) # dummy keypoints
    image = ImageWithKeypoints(image_arr, Keypoints(keypoints))
    tl_y = y - half_height
    tl_x = x - half_width
    br_y = y + half_height
    br_x = x + half_width
    image.draw_rectangle(Rectangle(tl_y=tl_y, tl_y=tl_y, br_y=br_y, br_x=br_x))
    return image.to_array()

def create_model_tiny(image_height, image_width, loss, optimizer):
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
    model.compile(loss=loss, optimizer=optimizer)
    return model

def create_model(image_height, image_width, loss, optimizer):
    """Creates the cat face locator model.
    
    Args:
        image_height: The height of the input images.
        image_width: The width of the input images.
        optimizer: Keras optimizer to use, e.g. Adam() or "sgd".
    Returns:
        Sequential
    """
    
    model = Sequential()
    
     # Tensor size at this point (if 128x128 input): 3x128x128
    model.add(Convolution2D(32, 1 if GRAYSCALE else 3, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Convolution2D(32, 32, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Tensor size (...): 32x64x64
    model.add(Convolution2D(64, 32, 3, 3, border_mode="valid"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Convolution2D(64, 64, 3, 3, border_mode="valid"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Tensor size (...): 64x30x30
    model.add(Convolution2D(128, 64, 3, 3, border_mode="valid"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.33))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 128, 3, 3, border_mode="valid"))
    model.add(BatchNormalization())
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
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

if __name__ == "__main__":
    main()
