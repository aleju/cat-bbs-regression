# -*- coding: utf-8 -*-
"""
Trains a model to locate cat faces in images (assumes that the image contains a cat face).
"""
from __future__ import absolute_import, division, print_function
from dataset import Dataset, ImageWithKeypoints, Keypoints, Rectangle
import numpy as np
import argparse
import random
import os
from scipy import misc
from skimage import draw
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import Progbar

np.random.seed(42)
random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

MODEL_IMAGE_HEIGHT = 128
MODEL_IMAGE_WIDTH = 128
PADDING = 20
AUGMENTATIONS = 0
NB_LOAD_IMAGES = 2
SPLIT = 0.1
EPOCHS = 50
BATCH_SIZE = 64
SAVE_WEIGHTS_FILEPATH = os.path.join(CURRENT_DIR, "cat_face_locator.weights")
SAVE_WEIGHTS_CHECKPOINT_FILEPATH = os.path.join(CURRENT_DIR, "cat_face_locator.best.weights")
SAVE_PREDICTIONS = True
SAVE_PREDICTIONS_DIR = os.path.join(CURRENT_DIR, "predictions")

def main():
    """Main method that reads the images, trains a model, then saves weights and predictions."""
    parser = argparse.ArgumentParser(description="Train a model to locate cat faces in images.")
    parser.add_argument("--dataset", required=True, help="Path to your 10k cats dataset directory")
    args = parser.parse_args()

    subdir_names = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
    subdirs = [os.path.join(args.dataset, subdir) for subdir in subdir_names]

    # initialize dataset
    dataset = Dataset(subdirs)

    # load images and labels
    print("Loading images...")
    X, y = load_Xy(dataset, NB_LOAD_IMAGES, AUGMENTATIONS)

    # split train and val
    nb_images = X.shape[0]
    nb_train = int(nb_images * (1 - SPLIT))
    nb_val = nb_images - nb_train
    X_train = X[0:nb_train, ...]
    y_train = y[0:nb_train, ...]
    X_val = X[nb_train:, ...]
    y_val = y[nb_train:, ...]

    # create model
    print("Creating model...")
    model = create_model_tiny(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, "mse", Adam())

    # fit
    checkpoint_cb = ModelCheckpoint(SAVE_WEIGHTS_CHECKPOINT_FILEPATH, verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.0,
              validation_data=(X_val, y_val), show_accuracy=False,
              callbacks=[checkpoint_cb])

    # save weights
    print("Saving weights...")
    model.save_weights(SAVE_WEIGHTS_FILEPATH, overwrite=True)

    # save predictions on val set
    if SAVE_PREDICTIONS:
        print("Saving example predictions...")
        y_preds = model.predict(X_val, batch_size=BATCH_SIZE)
        for img_idx, (y, x, half_height, half_width) in enumerate(y_preds):
            img_arr = draw_predicted_rectangle(X_val[img_idx], y, x, half_height, half_width)
            filepath = os.path.join(SAVE_PREDICTIONS_DIR, "%d.png" % (img_idx,))
            misc.imsave(filepath, np.squeeze(img_arr))

def load_Xy(dataset, nb_load, nb_augmentations):
    """Loads X and y (examples with labels) for the dataset.
    Examples are images.
    Labels are the coordinates of the face rectangles with their half-heights and half-widths
    (each normalized to 0-1 with respect to the image dimensions.)

    Args:
        dataset            The Dataset object.
        nb_load            Intended number of images to load.
        nb_augmentations   Number of augmentations to perform.
    Returns:
        X (numpy array of shape (N, 3, height, width)),
        y (numpy array of shape (N, 4))
    """
    i = 0
    nb_load = min(nb_load, len(dataset.fps))
    nb_images = nb_load + nb_load * nb_augmentations
    X = np.zeros((nb_images, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3), dtype=np.float32)
    y = np.zeros((nb_images, 4), dtype=np.float32)

    for img_idx, image in enumerate(dataset.get_images()):
        print("Loading image %d of %d..." % (img_idx+1, nb_load))
        image.pad(PADDING)
        augs = image.augment(AUGMENTATIONS, hflip=True, vflip=False, scale_to_percent=(0.9, 1.1), scale_axis_equally=False,
                             rotation_deg=10, shear_deg=0, translation_x_px=5, translation_y_px=5,
                             brightness_change=0.1, noise_mean=0.0, noise_std=0.05)
        for aug in [image] + augs:
            aug.unpad(PADDING)
            aug.resize(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
            X[i] = aug.to_array() / 255.0
            face_rect = aug.keypoints.get_rectangle(aug)
            face_rect.normalize(aug)
            center = face_rect.get_center()
            width = face_rect.get_width()
            height = face_rect.get_height()
            y[i] = [center.y, center.x, height, width]
            i += 1

        if (img_idx + 1) >= nb_load:
            break

    X = np.rollaxis(X, 3, 1)

    return X, y

def unnormalize_prediction(y, x, half_height, half_width, img_height=MODEL_IMAGE_HEIGHT, img_width=MODEL_IMAGE_WIDTH):
    """Transforms a predictions from normalized (0 to 1) y, x, half-width,
    half-height to pixel values (top left y, top left x, bottom right y,
    bottom right x).
    Args:
        y: Normalized y coordinate of rectangle center.
        x: Normalized x coordinate of rectangle center.
        half_height: Normalized height of rectangle.
        half_width: Normalized width of rectangle.
        img_height: Height of the image to use while unnormalizing.
        img_width: Width of the image to use while unnormalizing.
    Returns:
        (top left y in px, top left x in px, bottom right y in px,
        bottom right x in px)
    """
    # calculate x, y of corners in pixels
    tl_y = int((y - half_height) * img_height)
    tl_x = int((x - half_width) * img_width)
    br_y = int((y + half_height) * img_height)
    br_x = int((x + half_width) * img_width)

    # make sure that x and y coordinates are within image boundaries
    tl_y = clip(0, tl_y, img_height-2)
    tl_x = clip(0, tl_x, img_width-2)
    br_y = clip(0, br_y, img_height-1)
    br_x = clip(0, br_x, img_width-1)

    # make sure that top left corner is really top left of bottom right values
    if tl_y > br_y:
        tl_y, br_y = br_y, tl_y
    if tl_x > br_x:
        tl_x, br_x = br_x, tl_x

    # make sure that the area covered is at least 1px,
    # move preferably the top left corner
    # but dont move it outside of the image
    if tl_y == br_y:
        if tl_y == 0:
            br_y += 1
        else:
            tl_y -= 1

    if tl_x == br_x:
        if tl_x == 0:
            br_x += 1
        else:
            tl_x -= 1

    return tl_y, tl_x, br_y, br_x

def draw_predicted_rectangle(image_arr, y, x, half_height, half_width):
    """Draws a rectangle onto the image at the provided coordinates.
    Args:
        image_arr: Numpy array of the image.
        y: y-coordinate of the rectangle (normalized to 0-1).
        x: x-coordinate of the rectangle (normalized to 0-1).
        half_height: Half of the height of the rectangle (normalized to 0-1).
        half_width: Half of the width of the rectangle (normalized to 0-1).
    Returns:
        Modified image (numpy array)
    """
    assert image_arr.shape[0] == 3, str(image_arr.shape)
    height = image_arr.shape[1]
    width = image_arr.shape[2]
    tl_y, tl_x, br_y, br_x = unnormalize_prediction(y, x, half_height, half_width, img_height=height, img_width=width)
    image_arr = np.copy(image_arr) * 255
    image_arr = np.rollaxis(image_arr, 0, 3)
    return draw_rectangle(image_arr, tl_y, tl_x, br_y, br_x)

def draw_rectangle(img, tl_y, tl_x, br_y, br_x):
    """Draws a rectangle onto an image.
    Args:
        img: The image as a numpy array of shape (row, col, channel).
        tl_y: Top left y coordinate as pixel.
        tl_x: Top left x coordinate as pixel.
        br_y: Top left y coordinate as pixel.
        br_x: Top left x coordinate as pixel.
    Returns:
        image with rectangle
    """
    assert img.shape[2] == 3, img.shape[2]
    img = np.copy(img)
    lines = [
        (tl_y, tl_x, tl_y, br_x), # top left to top right
        (tl_y, br_x, br_y, br_x), # top right to bottom right
        (br_y, br_x, br_y, tl_x), # bottom right to bottom left
        (br_y, tl_x, tl_y, tl_x)  # bottom left to top left
    ]
    for y0, x0, y1, x1 in lines:
        rr, cc, val = draw.line_aa(y0, x0, y1, x1)
        img[rr, cc, 0] = val * 255

    return img

def clip(lower, val, upper):
    """Clips a value. For lower bound L, upper bound U and value V it
    makes sure that L <= V <= U holds.
    Args:
        lower: Lower boundary (including)
        val: The value to clip
        upper: Upper boundary (including)
    Returns:
        value within bounds
    """
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val

def create_model_tiny(image_height, image_width, loss, optimizer):
    """Creates the tiny version of the cat face locator model.
    This is useful for debugging, because it doesn't take as much theano compile time.

    Args:
        image_height: The height of the input images.
        image_width: The width of the input images.
        loss: Keras loss function (name or object), e.g. "mse".
        optimizer: Keras optimizer to use, e.g. Adam() or "sgd".
    Returns:
        Sequential
    """

    model = Sequential()

     # 3x128x128
    model.add(Convolution2D(4, 3, 3, border_mode="same", input_shape=(3, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    # 4x64x64
    model.add(Convolution2D(8, 3, 3, border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # 8x32x32
    model.add(Convolution2D(16, 3, 3, border_mode="same"))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # 16x16x16 = 4096
    model.add(Flatten())

    model.add(Dense(64))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(4))
    model.add(Activation("sigmoid"))

    print("Compiling...")
    model.compile(loss=loss, optimizer=optimizer)
    return model

def create_model(image_height, image_width, loss, optimizer):
    """Creates the cat face locator model.

    Args:
        image_height: The height of the input images.
        image_width: The width of the input images.
        loss: Keras loss function (name or object), e.g. "mse".
        optimizer: Keras optimizer to use, e.g. Adam() or "sgd".
    Returns:
        Sequential
    """

    model = Sequential()

     # 3x128x128
    model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(3, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # 32x64x64
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # 64x32x32
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # 128x16x16
    model.add(Convolution2D(256, 5, 5, border_mode="same"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # 256x8x8 = 16384
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(4))
    model.add(Activation("sigmoid"))

    # compile with mean squared error
    print("Compiling...")
    model.compile(loss=loss, optimizer=optimizer)

    return model

if __name__ == "__main__":
    main()
