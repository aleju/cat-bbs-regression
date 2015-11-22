# -*- coding: utf-8 -*-
"""
This file is used to train a convnet to recognize cat faces in images.
Training is done with the "10k cats" dataset.
The trained convnet expects every image to contain at least one cat.
The trained convnet will only mark one face if there are multiple cats.

How to train
------------
1. Download the 10k cats dataset from archive.org
2. Extract it somewhere. It should contain the subdirectories CAT_00, CAT_01, ..., CAT_06.
3. Change the constant "MAIN_DIR" further below to the directory of your 10k cats dataset.
4. Start training with:
    python train_cat_face_locator.py

It will train by default for 50 epochs. You can later continue training for another 50 epochs with
    python train_cat_face_locator.py --load="cat_face_locator.weights"
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import re
import os
import gzip
import sys
import random
import numpy as np
import csv
import math
from scipy import misc
from skimage import color
from skimage import transform as tf
from ImageAugmenter import create_aug_matrices, apply_aug_matrices
import numpy as np
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
from ImageAugmenter import ImageAugmenter
import argparse
from saveload import load_weights_seq

os.sys.setrecursionlimit(10000)
np.random.seed(42)
random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MAIN_DIR = "/media/aj/grab/ml/datasets/10k_cats"
DIRS = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
DIRS = [os.path.join(MAIN_DIR, subdir) for subdir in DIRS]
VISUALIZE = False

MODEL_IMAGE_HEIGHT = 80
MODEL_IMAGE_WIDTH = 80
GRAYSCALE = False
SAVE_EXAMPLES = True
SAVE_EXAMPLES_DIR = os.path.join(CURRENT_DIR, "out_val_predictions_angle")
SAVE_TRAINING_IMAGES = True
SAVE_TRAINING_IMAGES_DIR = os.path.join(CURRENT_DIR, "out_train_angle")
SAVE_WEIGHTS_FILEPATH = os.path.join(CURRENT_DIR, "cat_face_locator_angle80x80.weights")
SAVE_WEIGHTS_CHECKPOINT_FILEPATH = os.path.join(CURRENT_DIR, "cat_face_locator_angle80x80.best.weights")
SAVE_AUTO_OVERWRITE = True

#N_TRAIN = 9400 # true training count is N_TRAIN + (N_TRAIN * N_AUGMENTATIONS_TRAIN)
#N_VAL = 512
N_TRAIN = 9400
N_VAL = 512
N_AUGMENTATIONS_TRAIN = 5
N_AUGMENTATIONS_VAL = 0
EPOCHS = 100

def main():
    """Main function. Trains the model, saves the weights, save some example images with marked
    faces."""
    parser = argparse.ArgumentParser(description="Generate data for a word")
    parser.add_argument("--load", required=False, help="File to load weights from")
    args = parser.parse_args()
    
    # numbers here will be increased by augmentations,
    # i.e. at 5 augmentations per image 9400 + 5*9400 (original + augmented)
    X_train, Y_train = get_examples_coords(N_TRAIN, augmentations=N_AUGMENTATIONS_TRAIN)
    X_val, Y_val = get_examples_coords(N_VAL, start_at=N_TRAIN, augmentations=N_AUGMENTATIONS_VAL)
    print("Collected examples:")
    print("<X_train> shape: %s, size: %.4f MB, dtype:%s" % (str(X_train.shape), (X_train.nbytes / 1024 / 1024), str(X_train.dtype)))
    print("<X_val> shape: %s, size: %.4f MB, dtype:%s" % (str(X_val.shape), (X_val.nbytes / 1024 / 1024), str(X_val.dtype)))
    print("<Y_train> shape: %s, size: %.4f MB, dtype:%s" % (str(Y_train.shape), (Y_train.nbytes / 1024 / 1024), str(Y_train.dtype)))
    print("<Y_val> shape: %s, size: %.4f MB, dtype:%s" % (str(Y_val.shape), (Y_val.nbytes / 1024 / 1024), str(Y_val.dtype)))
    
    # debug: show training images with marked faces
    """
    for i, image in enumerate(X_train):
        print("-----------_")
        print(image.shape)
        print(Y_train[i])
        tl_y, tl_x, br_y, br_x = center_scale_to_pixels(image,
                                                        Y_train[i][0], Y_train[i][1],
                                                        Y_train[i][2], Y_train[i][3])
        marked_image = draw_rectangle(image*255,
                                      tl_x, br_x, tl_y, br_y, (255,),
                                      channel_is_first_axis=True)
        misc.imshow(np.squeeze(marked_image))
    """
    
    print("Saving marked training images...")
    for i, image in enumerate(X_train):
        # channel is first axis, therefore (shape[1], shape[2])
        #coords = unnormalize_coords(Y_train[i][4:], (image.shape[1], image.shape[2]))
        angle = Y_train[i][4]
        #print("[Debug Loop]")
        #print("coords:", coords)
        #print("Y_train[i]:", Y_train[i])
        #print("image.shape:", image.shape)
        #print("[/Debug Loop]")
        #print("angle", angle, angle*180)
        image = np.copy(image) * 255
        #image = draw_coordinates(image, coords, channel_is_first_axis=True)
        rect_coords = center_scale_to_pixels(image, Y_train[i][0], Y_train[i][1], Y_train[i][2], Y_train[i][3], channel_is_first_axis=True)
        #print("loop rect_coords:", rect_coords, "/", Y_train[i][0:4])
        image = draw_rectangle(image, rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3], channel_is_first_axis=True)
        image = draw_angle(image, int(Y_train[i][0]*image.shape[1]), int(Y_train[i][1]*image.shape[2]), angle*180, channel_is_first_axis=True)
        #misc.imshow(np.squeeze(image))
        if SAVE_TRAINING_IMAGES and i < 100:
            misc.imsave(os.path.join(SAVE_TRAINING_IMAGES_DIR, "%d.png" % (i,)), np.squeeze(image))
    
    print("Creating model...")
    model = create_model(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, Adam())
    
    if args.load:
        print("Loading weights...")
        load_weights_seq(model, args.load)

    checkpoint_cb = ModelCheckpoint(SAVE_WEIGHTS_CHECKPOINT_FILEPATH, verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=EPOCHS, validation_split=0.0,
              validation_data=(X_val, Y_val), show_accuracy=False,
              callbacks=[checkpoint_cb])

    print("Saving weights...")
    model.save_weights(SAVE_WEIGHTS_FILEPATH, overwrite=SAVE_AUTO_OVERWRITE)

    if SAVE_EXAMPLES:
        print("Saving examples (predictions)...")
        y_preds = predict_on_images(model, X_val)
        #for img_idx, (tl_y, tl_x, br_y, br_x) in enumerate(y_preds):
        for img_idx, (rect_coords, angle_deg) in enumerate(y_preds):
            image = np.copy(X_val[img_idx]) * 255
            image = np.rollaxis(image, 0, 3)
            if img_idx < 100:
                print("pred %d" % (img_idx,), "shape:", image.shape, "rect_coords:", rect_coords, "angle_deg:", angle_deg)
            #image = draw_coordinates(image, coords, channel_is_first_axis=False)
            rect_center_y = int((rect_coords[0]+rect_coords[2])/2)
            rect_center_x = int((rect_coords[1]+rect_coords[3])/2)
            image = draw_angle(image, rect_center_y, rect_center_x, angle_deg)
            image = draw_rectangle(image, rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3], channel_is_first_axis=False)
            misc.imsave(os.path.join(SAVE_EXAMPLES_DIR, "%d.png" % (img_idx,)),
                        np.squeeze(image))

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
    model.add(Dense(64, 5))
    model.add(Activation("tanh"))
    
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
    model.add(Dense(nb_last_kernels * 32, 5))
    #model.add(Dense(1024, 18))
    model.add(Activation("tanh"))
    
    # compile with mean squared error
    print("Compiling...")
    model.compile(loss="mse", optimizer=optimizer)
    
    return model

def get_all_filepaths(fp_dirs):
    """Returns the filepaths to all images in the provided directories.
    Args:
        fp_dirs: Lists of paths to directories.
    Returns:
        List of filepaths to images
    """
    result_img = []
    
    for fp_dir in fp_dirs:
        fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
        fps = [os.path.join(fp_dir, f) for f in fps]
        fps_img = [fp for fp in fps if re.match(r".*\.jpg$", fp)]
        #fps_coords = [fp for fp in fps if re.match(r".*\.jpg\.cat$", fp)]
        result_img.extend(fps_img)

    return result_img

def get_image_with_rectangle(image_filepath, coords_filepath):
    """Returns a squared training image together with the face box's center (X, Y) coordinates and
    scale (height, width) values.
    
    The values x, y, height and width are percentages relative to the squared training image
    sizes (height, width). They can be fed into a network with sigmoid outputs.
    
    Args:
        image_filepath: Filepath to the cat image.
        coords_filepath: Filepath to the coordinates file, which contains the coordinates of
            the cat face, as provided in the 10k cats dataset.
    Returns:
        image, (X, Y), (H, W),
        where
        image is a numpy array of the squared image (i.e. it has been increased in
            height/width to be squared and then resized to (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)),
        X is the x percentage coordinate of the center of the face box (percentage means relative
            to the max x value, i.e. 0.5 would mean "at half of the x axis")
        Y is analogous to X
        H is the half-height of the face box (also percentage), i.e. go that much to the right/left
            and you hit the border of the face box
        W is analogous to H, just the width.
    """
    
    fp_img = image_filepath
    fp_img_cat = coords_filepath
    if not os.path.isfile(fp_img) or not os.path.isfile(fp_img_cat):
        print("[WARNING] Either '%s' or '%s' could not be found" % (fp_img, fp_img_cat))
        return None, (None, None), (None, None), None
        #return None, None
    
    # read the image
    filename = fp_img[fp_img.rfind("/")+1:]
    image = misc.imread(fp_img, flatten=GRAYSCALE)
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    # read the coordinates of eyes, mouth etc.
    coords_raw = open(fp_img_cat, "r").readlines()[0].strip().split(" ")
    coords_raw = [abs(int(coord)) for coord in coords_raw]
    coords = []
    for i in range(0, len(coords_raw[1:]) - 1, 2): # first element is the number of coords
        x = clip(0, coords_raw[i+1], image_width-1)
        y = clip(0, coords_raw[i+2], image_height-1)
        pair = (y, x)
        coords.append(pair)
    
    # estimate face's center from coordinates
    # important coords: 0 = left eye, 1 = right eye, 2 = nose
    face_center_x = (coords[0][1] + coords[1][1] + coords[2][1]) / 3
    face_center_y = (coords[0][0] + coords[1][0] + coords[2][0]) / 3
    face_center = (int(face_center_y), int(face_center_x))
    
    # get coordinates of rectangle
    rect_tl_y, rect_tl_x, rect_br_y, rect_br_x = coordinates_to_rectangle(coords, image, method=3)
    
    # show all rectangles if VISUALIZE is activated
    if VISUALIZE:
        show_rectangles(image, coords)
    # -----------------------

    # pad the image around the borders so that it is square
    # we have to adjust the face rectangle coordinates accordingly
    image, (pad_top, pad_right, pad_bottom, pad_left) = square_image(image)
    image_height_square = image.shape[0]
    image_width_square = image.shape[1]
    rect_tl_x += pad_left
    rect_br_x += pad_left
    rect_tl_y += pad_top
    rect_br_y += pad_top

    #img = np.copy(image)
    #img = draw_rectangle(img, rect_tl_y, rect_tl_x, rect_br_y, rect_br_x)
    #misc.imshow(img)
    #print("original coords rect A:", rect_tl_y, rect_tl_x, rect_br_y, rect_br_x)
    #print("normalized:", rect_tl_y/image_height_square, rect_tl_x/image_width_square, rect_br_y/image_height_square, rect_br_x/image_width_square)
    
    # add paddings to coordinates and normalize them (0-1) to height/width
    coords_square = []
    for (y, x) in coords:
        y = (y + pad_top) / image_height_square
        x = (x + pad_left) / image_width_square
        coords_square.append((y, x))
    # -------------

    # calculate center and scales of the face box
    rect_height = rect_br_y - rect_tl_y
    rect_width = rect_br_x - rect_tl_x
    rect_scale_y = rect_height / 2 # half-height
    rect_scale_x = rect_width / 2 # half-width
    rect_center_y = rect_tl_y + rect_scale_y
    rect_center_x = rect_tl_x + rect_scale_x

    # resize image to 32x32
    image = misc.imresize(image, (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))

    tl_y, tl_x, br_y, br_x = center_scale_to_pixels(image, rect_center_y/image_height_square, rect_center_x/image_width_square, rect_scale_y/image_height_square, rect_scale_x/image_width_square)
    #img = np.copy(image)
    #img = draw_rectangle(img, tl_y, tl_x, br_y, br_x)
    #misc.imshow(img)
    #print("original coords rect B:", center_scale_to_pixels(image, rect_center_y/image_height_square, rect_center_x/image_width_square, rect_scale_y/image_height_square, rect_scale_x/image_width_square))
    #print("normalized:", rect_center_y/image_height_square, rect_center_x/image_width_square, rect_scale_y/image_height_square, rect_scale_x/image_width_square)

    # add one channel if in grayscale mode (otherwise shape would just be (height, width))
    if GRAYSCALE:
        image_tmp = np.zeros((image.shape[0], image.shape[1], 1))
        image_tmp[:, :, 0] = image
        image = image_tmp
    
    # image, (X,Y), (half-height, half-width), coords_square
    # all percent values relative to max x/y
    #return image, (rect_center_y/image_height_square, rect_center_x/image_width_square), (rect_scale_y/image_height_square, rect_scale_x/image_width_square), np.array(coords_square)
    return image, \
           (rect_center_y/image_height_square, rect_center_x/image_width_square), \
           (rect_scale_y/image_height_square, rect_scale_x/image_width_square), \
           np.array(coords_square)

def show_rectangles(image, coords):
    image_marked = np.copy(image)
    
    # mark all coordinates with dots
    for coord in coords:
        if GRAYSCALE:
            image_marked[coord[0], coord[1]] = 255
        else:
            image_marked[coord[0], coord[1], 0] = 255
    
    min_x, max_x, min_y, max_y = coordinates_to_rectangle(coords, image, method=0)
    min_x_fcenter, max_x_fcenter, min_y_fcenter, max_y_fcenter = coordinates_to_rectangle(coords, image_marked, method=1)
    min_x_half, max_x_half, min_y_half, max_y_half = coordinates_to_rectangle(coords, image_marked, method=2)
    min_x_merge, max_x_merge, min_y_merge, max_y_merge = coordinates_to_rectangle(coords, image_marked, method=3)
    min_x_merge_sq, max_x_merge_sq, min_y_merge_sq, max_y_merge_sq = coordinates_to_rectangle(coords, image_marked, method=4)
    
    # visualize 1st rectangle in green
    image_marked = draw_rectangle(image_marked, min_x, max_x, min_y, max_y, (0, 255, 0))
    
    # visualize 2nd rectangle in blue
    image_marked = draw_rectangle(image_marked,
                                  min_x_fcenter, max_x_fcenter,
                                  min_y_fcenter, max_y_fcenter,
                                  (0, 0, 255))
    
    # visualize 3rd rectangle in red
    image_marked = draw_rectangle(image_marked,
                                  min_x_half, max_x_half,
                                  min_y_half, max_y_half,
                                  (255, 0, 0))
    
    # visualize 4th rectangle in yellow
    image_marked = draw_rectangle(image_marked,
                                  min_x_merge, max_x_merge,
                                  min_y_merge, max_y_merge,
                                  (255, 255, 0))
        
    # visualize 5th rectangle in cyan
    image_marked = draw_rectangle(image_marked,
                                  min_x_merge_sq, max_x_merge_sq,
                                  min_y_merge_sq, max_y_merge_sq,
                                  (0, 255, 255))

    misc.imshow(image_marked)

def coordinates_to_rectangle(coords, image, method=4):
    #
    # Note: We define several rectangles for the face box here based on the provided coordinates.
    # Only the 5th version is currently used, as it looks best. The other ones are only generated
    # for easier debug outputs, doesnt cost much to generate them (except for ugly code).
    # 
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    face_center_x = (coords[0][1] + coords[1][1] + coords[2][1]) / 3
    face_center_y = (coords[0][0] + coords[1][0] + coords[2][0]) / 3
    face_center = (int(face_center_y), int(face_center_x))
    
    if method == 0:
        # 1st rectangle: bounding box around provided coordinates
        min_x = min([x for (y, x) in coords])
        min_y = min([y for (y, x) in coords])
        max_x = max([x for (y, x) in coords])
        max_y = max([y for (y, x) in coords])
        
        return (min_y, min_x, max_y, max_x)
    elif method == 1:
        # 2nd rectangle: the same rectangle as 1st, but translated to the center of the face
        
        min_y, min_x, max_y, max_x = coordinates_to_rectangle(coords, image, method=0)
        rectangle_center = (min_y + (max_y - min_y)/2, min_x + (max_x - min_x)/2)
        rectangle_center = (int(rectangle_center[0]), int(rectangle_center[1]))
        center_diff = (face_center[0] - rectangle_center[0], face_center[1] - rectangle_center[1])
        
        #min_x_fcenter = max(0, min_x + center_diff[0])
        min_x_fcenter = max(0, min_x + center_diff[1])
        #min_y_fcenter = max(0, min_y + center_diff[1])
        min_y_fcenter = max(0, min_y + center_diff[0])
        #max_x_fcenter = min(image_width-1, max_x + center_diff[0])
        max_x_fcenter = min(image_width-1, max_x + center_diff[1])
        #max_y_fcenter = min(image_height-1, max_y + center_diff[1])
        max_y_fcenter = min(image_height-1, max_y + center_diff[0])
        
        return (min_y_fcenter, min_x_fcenter, max_y_fcenter, max_x_fcenter)
    elif method == 2:
        # 3rd rectangle: the same rectangle as 1st, but translated _half-way_ towards the center of
        # the face
        
        min_y, min_x, max_y, max_x = coordinates_to_rectangle(coords, image, method=0)
        rectangle_center = (min_y + (max_y - min_y)/2, min_x + (max_x - min_x)/2)
        rectangle_center = (int(rectangle_center[0]), int(rectangle_center[1]))
        center_diff = (face_center[0] - rectangle_center[0], face_center[1] - rectangle_center[1])
        
        #min_x_half = max(0, int(min_x + center_diff[0]/2))
        min_x_half = max(0, int(min_x + center_diff[1]/2))
        #min_y_half = max(0, int(min_y + center_diff[1]/2))
        min_y_half = max(0, int(min_y + center_diff[0]/2))
        #max_x_half = min(image_width-1, int(max_x + center_diff[0]/2))
        max_x_half = min(image_width-1, int(max_x + center_diff[1]/2))
        #max_y_half = min(image_height-1, int(max_y + center_diff[1]/2))
        max_y_half = min(image_height-1, int(max_y + center_diff[0]/2))
        
        return (min_y_half, min_x_half, max_y_half, max_x_half)
    elif method == 3:
        # 4th rectangle: a merge between 1st and 3rd rectangle, essentially a bounding box around
        # the corners of both rectangles
        
        min_y, min_x, max_y, max_x = coordinates_to_rectangle(coords, image, method=0)
        min_y_half, min_x_half, max_y_half, max_x_half = coordinates_to_rectangle(coords, image, method=2)
        
        min_x_merge = max(0, min(min_x, min_x_half))
        min_y_merge = max(0, min(min_y, min_y_half))
        max_x_merge = min(image_width-1, max(max_x, max_x_half))
        max_y_merge = min(image_height-1, max(max_y, max_y_half))
        
        return (min_y_merge, min_x_merge, max_y_merge, max_x_merge)
    elif method == 4:
        # 5th rectangle: like 4th, but decreased in rows/columns to be squared
        # We will use only this rectangle.
        
        min_y_merge, min_x_merge, max_y_merge, max_x_merge = coordinates_to_rectangle(coords, image, method=3)
        
        """
        rec4_height = max_y_merge - min_y_merge
        rec4_width = max_x_merge - min_x_merge
        min_x_merge_sq = min_x_merge
        max_x_merge_sq = max_x_merge
        min_y_merge_sq = min_y_merge
        max_y_merge_sq = max_y_merge
        
        if rec4_height == rec4_width:
            pass
        elif rec4_height > rec4_width:
            diff = rec4_height - rec4_width
            remove_top = math.floor(diff / 2)
            remove_bottom = math.floor(diff / 2)
            if diff % 2 != 0:
                remove_top += 1
            min_y_merge_sq += int(remove_top)
            max_y_merge_sq -= int(remove_bottom)
        elif rec4_width > rec4_height:
            diff = rec4_width - rec4_height
            remove_left = math.floor(diff / 2)
            remove_right = math.floor(diff / 2)
            if diff % 2 != 0:
                remove_left += 1
            min_x_merge_sq += int(remove_left)
            max_x_merge_sq -= int(remove_right)
        """
        tl_y, tl_x, br_y, br_x = square_rectangle(image, min_y_merge, min_x_merge, max_y_merge, max_x_merge)
        
        #return (min_y_merge_sq, min_x_merge_sq, max_y_merge_sq, max_x_merge_sq)
        return (tl_y, tl_x, br_y, br_x)
    else:
        raise Exception("Unknown rectangle generation method %d chosen." % (method,))

def square_rectangle(image, tl_y, tl_x, br_y, br_x, channel_is_first_axis=False):
    if channel_is_first_axis:
        assert image.shape[0] in [1, 3]
        img_height = image.shape[1]
        img_width = image.shape[2]
    else:
        assert image.shape[2] in [1, 3]
        img_height = image.shape[0]
        img_width = image.shape[1]
    
    height = br_y - tl_y
    width = br_x - tl_x
    
    # extend by adding cols / rows until borders of image are reached
    i = 0
    while width < height and br_x < img_width and tl_x > 0:
        if i % 2 == 0:
            tl_x -= 1
        else:
            br_x += 1
        width += 1
    
    while height < width and br_y < img_height and tl_y > 0:
        if i % 2 == 0:
            tl_y -= 1
        else:
            br_y += 1
        height += 1
    
    # remove cols / rows until rectangle is squared
    # this part was written at a different time, which is why the removal works differently,
    # it does however the exactle same thing (move yx coordinates of topleft/bottemright corners)
    if height > width:
        diff = height - width
        remove_top = math.floor(diff / 2)
        remove_bottom = math.floor(diff / 2)
        if diff % 2 != 0:
            remove_top += 1
        tl_y += int(remove_top)
        br_y -= int(remove_bottom)
    elif width > height:
        diff = width - height
        remove_left = math.floor(diff / 2)
        remove_right = math.floor(diff / 2)
        if diff % 2 != 0:
            remove_left += 1
        tl_x += int(remove_left)
        br_x -= int(remove_right)
    
    return tl_y, tl_x, br_y, br_x

"""
def coordinates_to_rectangle(coords, image):
    print("coordinates_to_rectangle shape", image.shape)
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    face_center_x = (coords[0][1] + coords[1][1] + coords[2][1]) / 3
    face_center_y = (coords[0][0] + coords[1][0] + coords[2][0]) / 3
    face_center = (int(face_center_x), int(face_center_y))
    
    # 1st rectangle: convex hull (as rectangle) around provided coordinates
    min_x = min([coord[1] for coord in coords])
    min_y = min([coord[0] for coord in coords])
    max_x = max([coord[1] for coord in coords])
    max_y = max([coord[0] for coord in coords])
    rectangle_center = (min_x + (max_x - min_x)/2, min_y + (max_y - min_y)/2)
    rectangle_center = (int(rectangle_center[0]), int(rectangle_center[1]))
    center_diff = (face_center[0] - rectangle_center[0], face_center[1] - rectangle_center[1])
    
    # 2nd rectangle: the same rectangle as 1st, but translated to the center of the face
    min_x_fcenter = max(0, min_x + center_diff[0])
    min_y_fcenter = max(0, min_y + center_diff[1])
    max_x_fcenter = min(image_width-1, max_x + center_diff[0])
    max_y_fcenter = min(image_height-1, max_y + center_diff[1])
    
    # 3rd rectangle: the same rectangle as 1st, but translated _half-way_ towards the center of
    # the face
    min_x_half = max(0, int(min_x + center_diff[0]/2))
    min_y_half = max(0, int(min_y + center_diff[1]/2))
    max_x_half = min(image_width-1, int(max_x + center_diff[0]/2))
    max_y_half = min(image_height-1, int(max_y + center_diff[1]/2))
    
    # 4th rectangle: a merge between 1st and 3rd rectangle, essentially a bounding box around
    # the corners of both rectangles
    min_x_merge = max(0, min(min_x, min_x_half))
    min_y_merge = max(0, min(min_y, min_y_half))
    max_x_merge = min(image_width-1, max(max_x, max_x_half))
    max_y_merge = min(image_height-1, max(max_y, max_y_half))
    
    # 5th rectangle: like 4th, but decreased in rows/columns to be squared
    # We will use only this rectangle.
    rec4_height = max_y_merge - min_y_merge
    rec4_width = max_x_merge - min_x_merge
    min_x_merge_sq = min_x_merge
    max_x_merge_sq = max_x_merge
    min_y_merge_sq = min_y_merge
    max_y_merge_sq = max_y_merge
    
    if rec4_height == rec4_width:
        pass
    elif rec4_height > rec4_width:
        diff = rec4_height - rec4_width
        remove_top = math.floor(diff / 2)
        remove_bottom = math.floor(diff / 2)
        if diff % 2 != 0:
            remove_top += 1
        min_y_merge_sq += int(remove_top)
        max_y_merge_sq -= int(remove_bottom)
    elif rec4_width > rec4_height:
        diff = rec4_width - rec4_height
        remove_left = math.floor(diff / 2)
        remove_right = math.floor(diff / 2)
        if diff % 2 != 0:
            remove_left += 1
        min_x_merge_sq += int(remove_left)
        max_x_merge_sq -= int(remove_right)

    return (min_y_merge_sq, min_x_merge_sq, max_y_merge_sq, max_x_merge_sq)
"""

def square_image(image):
    """Pads an image to make it squared (i.e. same height and width).
    Will pad the width if width < height, else the height.
    
    Args:
        image: Numpy array of the image (H, W, C)
    Returns:
        image, squared (numpy array)
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    idx = 0
    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0
    
    # -------------------
    # count how many columns and rows we have to add (at left/right or top/bottom)
    # loops here are inefficient, but easy to read
    
    # columns
    while image_width < image_height:
        # alternate between adding a columns left or right
        if idx % 2 == 0:
            pad_left += 1
        else:
            pad_right += 1
        image_width += 1
        idx += 1
    
    # rows
    idx = 0
    while image_height < image_width:
        # alternate top/bottom
        if idx % 2 == 0:
            pad_top += 1
        else:
            pad_bottom += 1
        image_height += 1
        idx += 1
    
    # -------------------
    
    # pad the image with columns / rows as counted above
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if GRAYSCALE:
            image = np.pad(image,
                           ((pad_top, pad_bottom), (pad_left, pad_right)),
                           #mode=str("median"))
                           mode=str("constant"))
        else:
            image = np.pad(image,
                           ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                           #mode=str("median"))
                           mode=str("constant"))

    return image, (pad_top, pad_right, pad_bottom, pad_left)

#def get_examples_rectangles(count, start_at=0, augmentations=0):
    """Returns X and Y examples to train/test on.
    Args:
        count: Maximum number of different images to return (this will be increased by the
               augmentation number, i.e. count=1 with augmentations=10 will return 10+1 examples).
        start_at: Start index of the first example to return.
        augmentations: How often each image will be augmented.
    Returns:
        (X, Y)
        with X being a tensor of images
        and Y being in array of rows [center x, center y, height/2, width/2] of each face rectangle.
    """
"""
    # low strength augmentation because we will not change the coordinates, so the image
    # should be kept mostly the same
    ia = ImageAugmenter(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH,
                        channel_is_first_axis=False,
                        hflip=False, vflip=False,
                        scale_to_percent=(0.95, 1.05), scale_axis_equally=True,
                        rotation_deg=5, shear_deg=2,
                        translation_x_px=1, translation_y_px=1)
    
    images_filepaths = get_all_filepaths(DIRS)
    images = []
    labels = []
    for image_filepath in images_filepaths[start_at:start_at+count]:
        coords_filepath = "%s.cat" % (image_filepath,)
        image, (center_y, center_x), (scale_y, scale_x) = get_image_with_rectangle(image_filepath,
                                                                                   coords_filepath)
        # get_image_with_rectangle returns None if the coordinates file was not found,
        # which is the case for one image in 10k cats dataset
        if image is not None:
            images.append(image / 255) # project pixel values to 0-1
            y = [center_y, center_x, scale_y, scale_x]
            labels.append(y)
            
            if augmentations > 0:
                images_aug = []
                for i in range(augmentations):
                    if i % 2 == 0:
                        images_aug.append(np.fliplr(image))
                        labels.append((center_y, 1-center_x, scale_y, scale_x))
                    else:
                        images_aug.append(image)
                        labels.append(y)
                # also projects pixel values to 0-1
                images_aug = ia.augment_batch(np.array(images_aug, dtype=np.uint8))
                images.extend(images_aug)
    
    images = np.array(images, dtype=np.float32)
    images = np.rollaxis(images, 3, 1)
    
    return images, np.array(labels, dtype=np.float32)
"""

def get_examples_coords(count, start_at=0, augmentations=0):
    """Returns X and Y examples to train/test on.
    Args:
        count: Maximum number of different images to return (this will be increased by the
               augmentation number, i.e. count=1 with augmentations=10 will return 10+1 examples).
        start_at: Start index of the first example to return.
        augmentations: How often each image will be augmented.
    Returns:
        (X, Y)
        with X being a tensor of images
        and Y being in array of rows, each containing 18 values, i.e. 9 coordinates of form (y, x),
        with all coordinates being normalized to the height (y) or width (x) of the squared image.
    """
    # low strength augmentation because we will not change the coordinates, so the image
    # should be kept mostly the same
    """
    ia = ImageAugmenter(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH,
                        channel_is_first_axis=False,
                        hflip=False, vflip=False,
                        scale_to_percent=(0.95, 1.05), scale_axis_equally=True,
                        rotation_deg=5, shear_deg=2,
                        translation_x_px=1, translation_y_px=1)
    """
    
    target_count = count + (count * augmentations)
    images_filepaths = get_all_filepaths(DIRS)
    if GRAYSCALE:
        images = np.zeros((target_count, 1, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), dtype=np.float32)
    else:
        images = np.zeros((target_count, 3, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), dtype=np.float32)
    labels = np.zeros((target_count, 2 + 2 + 1), dtype=np.float32)
    img_idx = 0
    for fp_idx, image_filepath in enumerate(images_filepaths[start_at:start_at+count]):
        coords_filepath = "%s.cat" % (image_filepath,)
        #image, (center_y, center_x), (scale_y, scale_x), coords = get_image_with_rectangle(image_filepath, coords_filepath)
        image, (center_y, center_x), (scale_y, scale_x), coords = get_image_with_rectangle(image_filepath, coords_filepath)
        # get_image_with_rectangle returns None if the coordinates file was not found,
        # which is the case for one image in 10k cats dataset
        if image is None:
            for i in range(1 + augmentations):
                rnd = random.randint(0, img_idx)
                images[img_idx] = images[rnd]
                labels[img_idx] = labels[rnd]
                img_idx += 1
        else:
            if fp_idx % 100 == 0:
                print("%d of %d images..." % (img_idx, target_count))
            coords_arr = np.array(coords)
            coords_flat = np.array(coords).flatten()
            
            label_prototype = np.zeros((2 + 2 + 1,), dtype=np.float32)
            label_prototype[0] = center_y
            label_prototype[1] = center_x
            label_prototype[2] = scale_y
            label_prototype[3] = scale_x
            label_orig = np.copy(label_prototype)
            label_orig[4] = angle_between_eyes(coords_flat[0], coords_flat[1], coords_flat[2], coords_flat[3])
            #label_orig[4:4+4] = coords_flat[0:4]
            
            #img = np.copy(image)
            #rect_tl_y, rect_tl_x, rect_br_y, rect_br_x = center_scale_to_pixels(img, label_orig[0], label_orig[1], label_orig[2], label_orig[3])
            #img = draw_rectangle(img, rect_tl_y, rect_tl_x, rect_br_y, rect_br_x, channel_is_first_axis=False)
            #print("[get_examples_coords] orig + face")
            #print(label_orig)
            #misc.imshow(img)
            
            #images.append(image / 255) # project pixel values to 0-1
            images[img_idx] = np.rollaxis(image, 2, 0) / 255
            labels[img_idx] = label_orig
            img_idx += 1
            
            #augmentations = 0
           # print("[get_examples_coords] Adding original coords:", coords_flat)
            if augmentations > 0:
                rectangle = np.array([center_y * image.shape[0], center_x * image.shape[1], scale_y, scale_x])
                images_aug, rectangle_aug, coords_aug = augment(image, rectangle, unnormalize_coords(coords_arr, image.shape), augmentations, normalize=True)
                for i in range(augmentations):
                    images[img_idx] = np.rollaxis(images_aug[i], 2, 0)
                    label_aug = np.zeros((2 + 2 + 1,), dtype=np.float32)
                    label_aug[0:4] = rectangle_aug[i]
                    #label_aug[4:4+4] = coords_aug[i].flatten()[0:4]
                    label_aug[4] = angle_between_eyes(coords_aug[i][0][0], coords_aug[i][0][1], coords_aug[i][1][0], coords_aug[i][1][1])
                    labels[img_idx] = label_aug
                    img_idx += 1
            """
            if augmentations > 0:
                images_aug = []
                for i in range(augmentations):
                    if i % 2 == 0:
                        images_aug.append(np.fliplr(image))
                        # flip coordinates, i.e. (1 - x)
                        label = np.copy(coords_flat)
                        for i in range(1, len(label), 2):
                            label[i] = 1 - label[i]
                        labels.append(label)
                    else:
                        images_aug.append(image)
                        labels.append(np.copy(coords_flat))
                # also projects pixel values to 0-1
                images_aug = ia.augment_batch(np.array(images_aug, dtype=np.uint8))
                images.extend(images_aug)
            """
    #print(images)
    #images_arr = np.array(images, dtype=np.float32)
    #images = np.rollaxis(images_arr, 3, 1)
    
    #print("[get_examples_coords] labels out:", np.array(labels))
    return images, labels


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            v = 0.0
        else:
            v = np.pi
    else:
        v = angle
    
    if v2_u[0] < 0:
        return -v
    else:
        return v

def angle_between_eyes(left_y, left_x, right_y, right_x, normalize=True):
    left_eye = np.array([left_y, left_x])
    right_eye = np.array([right_y, right_x])
    # conversion to int is here necessary, otherwise eyes_vector cant have negative values
    eyes_vector = right_eye.astype(np.float32) - left_eye.astype(np.float32)
    x_axis_vector = np.array([0, 1])
    angle = angle_between(x_axis_vector, eyes_vector)
    angle_deg = math.degrees(angle)
    
    #print("angle", angle, "angle_deg", angle_deg, "eyes_vector", eyes_vector, "left_y", left_y, "left_x", left_x, "right_y", right_y, "right_x", right_x)
    assert -180 <= angle_deg <= 180, angle_deg
    if normalize:
        return angle_deg / 180
    else:
        return angle_deg

def center_scale_to_pixels(image, center_y, center_x, scale_y, scale_x, channel_is_first_axis=False):
    """Converts the face rectangle position from [center x, center y, height/2, width/2] (all
    relative to max x/y) to [topleft x, topleft y, bottomright x, bottomright y].
    Args:
        image: numpy array of the image of the face rectangle.
        center_y: Y-coordinate of the face rectangle center, relative to max y.
        center_x: X-coordinate of the face rectangle center, relative to max x.
        scale_y: Half-height of the face rectangle, relative to max height.
        scale_x: Half-width of the face rectangle, relative to max width.
    Returns:
        (topleft x, topleft y, bottomright x, bottomright y),
        als exact pixel values (not relative to max values).
    """
    #img_height =  MODEL_IMAGE_HEIGHT
    #img_width = MODEL_IMAGE_WIDTH
    if channel_is_first_axis:
        img_height = image.shape[1]
        img_width = image.shape[2]
    else:
        img_height = image.shape[0]
        img_width = image.shape[1]
    
    y, x, height_half, width_half = (int(center_y*img_height),
                                     int(center_x*img_height),
                                     int(scale_y*img_width),
                                     int(scale_x*img_width))
    tl_y = y - height_half
    tl_x = x - width_half
    br_y = y + height_half
    br_x = x + width_half
    
    # not outside of the image bounds
    tl_y = min(max(tl_y, 0), img_height-1)
    tl_x = min(max(tl_x, 0), img_width-1)
    br_y = min(max(br_y, 0), img_height-1)
    br_x = min(max(br_x, 0), img_width-1)
    
    # at least an area of 1 pixel
    if tl_y == br_y:
        if tl_y > 0:
            tl_y -= 1
        else:
            br_y += 1
    if tl_x == br_x:
        if tl_x > 0:
            tl_x -= 1
        else:
            br_x += 1
    
    return tl_y, tl_x, br_y, br_x

def unnormalize_coords(coords, shape):
    coords = np.array(coords)
    assert len(coords.shape) in [1, 2], np.array(coords).shape
    assert isinstance(shape, tuple), type(shape)
    
    height = shape[0]
    width = shape[1]
    
    if len(coords.shape) == 1:
        result = []
        
        for i in range(0, len(coords)):
            if i % 2 == 0:
                v = int(coords[i] * height)
                v = clip(0, v, height-1)
                result.append(v)
            else:
                v = int(coords[i] * width)
                v = clip(0, v, width-1)
                result.append(v)
        return np.array(result)
    else:
        result = []
        for (y, x) in coords:
            y = int(y * height)
            y = clip(0, y, height-1)
            x = int(x * width)
            x = clip(0, x, width-1)
            result.append([y, x])
    
        return np.array(result)

def clip(minval, val, maxval):
    if val < minval:
        return minval
    elif val > maxval:
        return maxval
    else:
        return val

def draw_rectangle(image, min_y, min_x, max_y, max_x, color_tuple=None, channel_is_first_axis=False):
    """Draws a rectangle at given coordinates into the image.
    Args:
        image: The image onto which to paint (will be changed in-place).
        min_x: Rectangle's top left x value.
        max_x: Rectangle's bottom right x value.
        min_y: Rectangle's top left y value.
        max_y: Rectangle's bottom right y value.
    Returns:
        image
    """
    if color_tuple is None:
        if GRAYSCALE:
            color_tuple = 255
        else:
            color_tuple = (255,0,0)
    
    if len(color_tuple) > 0 and GRAYSCALE:
        print("[WARNING] got 3-channel color tuple in draw_rectangle(), " \
              "but grayscale is active.", color_tuple)
        color_tuple = 255
    
    if channel_is_first_axis:
        image = np.rollaxis(image, 0, 3)
    
    for x in range(min_x, max_x+1):
        image[min_y, x, ...] = color_tuple
        image[max_y, x, ...] = color_tuple
    for y in range(min_y, max_y+1):
        image[y, min_x, ...] = color_tuple
        image[y, max_x, ...] = color_tuple
    
    if channel_is_first_axis:
        image = np.rollaxis(image, 2, 0)
    
    return image

def draw_angle(image, start_y, start_x, angle_deg, length=12, channel_is_first_axis=False, color_tuple=None):
    image = np.copy(image)
    
    if channel_is_first_axis:
        image = np.rollaxis(image, 0, 3)
    height = image.shape[0]
    width = image.shape[1]
    
    if color_tuple is None:
        if GRAYSCALE:
            color_tuple = 255
        else:
            color_tuple = (255,0,0)
    
    #(y_cart, x_cart) = pol2cart(1, angle_deg)
    #angle_vec = np.array([y_cart, x_cart]) - np.array([start_y, start_x])
    for i in range(length):
        y_cart, x_cart = pol2cart(i, math.radians(angle_deg))
        #print("[draw_angle] marking x=+%d, y=+%d from start x=%d and y=%d" % (x_cart, y_cart, start_x, start_y))
        image[clip(0, start_y+y_cart, height-1), clip(0, start_x+i, height-1), ...] = color_tuple

    if channel_is_first_axis:
        image = np.rollaxis(image, 2, 0)
    
    return image

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (y, x)

def draw_coordinates(image, coords_flat, channel_is_first_axis=False):
    coords = coords_flat
    assert len(np.array(coords).shape) == 1, np.array(coords).shape
    
    if channel_is_first_axis:
        image = np.rollaxis(image, 0, 3)
    
    color = True
    if len(image.shape) < 3 or image.shape[2] == 1:
        color = False
    
    height = image.shape[0]
    width = image.shape[1]
    
    #for (y, x) in coords:
    for i in range(0, len(coords), 2):
        y = coords[i]
        x = coords[i+1]
        if color:
            image[y, clip(0,x-1,width-1):clip(0,x+2,width-1), ...] = (255, 0, 0)
            image[clip(0,y-1,height-1):clip(0,y+2,height-1), x, ...] = (255, 0, 0)
            #image[y, x, 0] = 255
        else:
            image[y, clip(0,x-1,width-1):clip(0,x+2,width-1), ...] = 255
            image[clip(0,y-1,height-1):clip(0,y+2,height-1), x, ...] = 255
            #image[y, x, ...] = 255

    if channel_is_first_axis:
        image = np.rollaxis(image, 2, 0)

    return image

def predict_on_images(model, images):
    """Predicts the coordinates of face rectangles for given images.
    Args:
        model: The neural net model.
        images: Numpy array of images.
    Returns:
        List of coordinate-arrays
    """
    y_preds_model = model.predict(images, batch_size=128)
    # from (img idx, channel, height, width)
    # to   (img idx, height, width, channel)
    # as expected by unnormalize_coords
    images = np.rollaxis(images, 1, 4)
    
    y_preds = []
    for i, y_pred in enumerate(y_preds_model):
        rect_coords = center_scale_to_pixels(images[i], y_pred[0], y_pred[1], y_pred[2], y_pred[3])
        #coords = unnormalize_coords(y_pred[4:], images[i].shape)
        angle = y_pred[4]
        y_preds.append((rect_coords, angle*180))
    return y_preds

def augment(image, rectangle, coords_arr, nb_augmentations, normalize=False):
    assert image.dtype == 'uint8', image.dtype
    assert len(image.shape) == 3
    assert image.shape[2] in [1, 3]
    assert len(coords_arr.shape) == 2
    assert coords_arr.shape[0] == 9
    assert coords_arr.shape[1] == 2
    
    height = image.shape[0]
    width = image.shape[1]
    images = np.zeros((nb_augmentations, image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
    rectangle_flip = np.zeros((nb_augmentations, rectangle.shape[0]), dtype=rectangle.dtype)
    coords_arr_flip = np.zeros((nb_augmentations, coords_arr.shape[0], coords_arr.shape[1]), dtype=coords_arr.dtype)
    for i in range(nb_augmentations):
        coords_arr_flip[i] = np.copy(coords_arr)
        rectangle_flip[i] = np.copy(rectangle)
        
        if i % 2 == 0:
            images[i] = np.fliplr(np.copy(image))
            rectangle_flip[i][1] = (width-1) - rectangle_flip[i][1] # center x of rectangle
            assert 0 <= rectangle_flip[i][1] < width
            # flip x axis coordinates
            for j in range(coords_arr.shape[0]):
                coords_arr_flip[i][j][1] = (width-1) - coords_arr_flip[i][j][1]
                assert 0 <= coords_arr_flip[i][j][1] < width
            # switch left and right eye coords
            # after previous 1-x they are at the right positions, just the wrong way around
            # todo: other coords should probably be flipped to (eg left and right ear tip)
            coords_arr_flip[i][0][0], coords_arr_flip[i][1][0] = coords_arr_flip[i][1][0], coords_arr_flip[i][0][0]
            coords_arr_flip[i][0][1], coords_arr_flip[i][1][1] = coords_arr_flip[i][1][1], coords_arr_flip[i][0][1]
        else:
            images[i] = np.copy(image)
    
    #images = np.repeat(image, nb_augmentations, axis=0)
    #print("augment shapes", images.shape, image.shape)
    
    matrices = create_aug_matrices(nb_augmentations, width, height,
                                   scale_to_percent=(0.9, 1.1),
                                   scale_axis_equally=False,
                                   rotation_deg=10,
                                   shear_deg=2,
                                   translation_x_px=5,
                                   translation_y_px=5)
    
    images_aug = apply_aug_matrices(images, matrices, transform_channels_equally=True,
                                    channel_is_first_axis=False, random_order=False)
    rectangle_aug = np.zeros(rectangle_flip.shape)
    coords_aug = np.zeros(coords_arr_flip.shape)
    for i in range(nb_augmentations):
        rectangle_aug[i][0:2] = warp_coordinates([rectangle_flip[i, 0:2]], matrices[i], image)
        rectangle_aug[i][2:] = rectangle_flip[i, 2:]
        coords_aug[i] = warp_coordinates(coords_arr_flip[i], matrices[i], image)
    
    if normalize:
        rectangle_aug[:, 0] /= image.shape[0] # center y
        rectangle_aug[:, 1] /= image.shape[1] # center x
        coords_aug[:, :, 0] /= image.shape[0]
        coords_aug[:, :, 1] /= image.shape[1]
    
    #print("[augment] coords_aug OUT:", coords_aug)
    
    return images_aug, rectangle_aug, coords_aug

def warp_coordinates(coords_arr, matrix, image):
    #print("[warp_coordinates] IN: ", coords_arr)
    # create new image with N channels for N coordinates
    # mark each coordinate's pixel in the respective channel
    # rotate
    # read out new coordinates (after rotation)
    mode = "constant"
    cval = 0.0
    interpolation_order = 1
    
    image_coords = np.zeros((len(coords_arr), image.shape[0], image.shape[1]), dtype=np.uint8)
    for i, (y, x) in enumerate(coords_arr):
        image_coords[i][y][x] = 255
    
    new_coords_arr = []
    last_coord = None
    for i in range(len(coords_arr)):
        image_coords_aug = tf.warp(image_coords[i], matrix, mode=mode, cval=cval, order=interpolation_order)
        maxindex = np.argmax(image_coords_aug)
        if maxindex == 0:
            # max value at index 0 => coordinate most likely ended up outside of image after
            # augmentation; so try to replace it with another coordinate that ended up inside
            # the image. If that fails, pick the old value (before augmentation).
            #if last_coord is not None:
            #    (y, x) = last_coord
            #else:
            (y, x) = coords_arr[i]
            #print("Note: coordinate %d is outside of the image after augmentation" \
            #      "(original coords: %.4f, %.4f, picked: %.4f, %.4f)" % (i, coords_arr[i][0], coords_arr[i][1], y, x))
        else:
            (y, x) = np.unravel_index(maxindex, image_coords_aug.shape)
            last_coord = (y, x)
            #print("coord", i, maxindex, y, x)
            #misc.imshow(image_coords[i])
            #misc.imshow(image_coords_rot)
        new_coords_arr.append((y, x))
    #print("[warp_coordinates] OUT: ", np.array(new_coords_arr))
    return np.array(new_coords_arr)

if __name__ == "__main__":
    main()
