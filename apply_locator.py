# -*- coding: utf-8 -*-
"""
File to apply the trained ConvNet model to a number of images.
It will use the ConvNet to locate cat faces in the images and mark them.
It is expected that each image contains exactly one cat (i.e. a face will be
extracted out of each image, even if there is no cat).
If an image contains multiple cats, only one face will be extracted.

Usage:
    python train.py
    python apply_locator.py
"""
from __future__ import division, print_function
from dataset import Dataset
import os
import numpy as np
import argparse
import random
import re
from scipy import misc
from scipy import ndimage
from train import MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, BATCH_SIZE, \
                  SAVE_WEIGHTS_FILEPATH, create_model, create_model_tiny, \
                  draw_predicted_rectangle
from keras.optimizers import Adam
from ImageAugmenter import ImageAugmenter

np.random.seed(42)
random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
WRITE_TO_DIR = os.path.join(CURRENT_DIR, "apply_locator_output")
# scale (height, width) of each saved image
OUT_SCALE = 64

def main():
    """
    Main function.
    Does the following step by step:
    * Load images (from which to extract cat faces) from SOURCE_DIR
    * Initialize model (as trained via train_cat_face_locator.py)
    * Loads and prepares images for the model.
    * Uses trained model to predict locations of cat faces.
    * Projects face coordinates onto original images
    * Marks faces in original images.
    * Saves each marked image.
    """
    parser = argparse.ArgumentParser(description="Apply a trained cat face locator model to images.")
    parser.add_argument("--dataset", required=True, help="Path to the images directory.")
    parser.add_argument("--weights", required=False, default="cat_face_locator.best.weights", help="Filepath to the weights of the model.")
    parser.add_argument("--output", required=False, default="apply_model_output", help="Filepath to the directory in which to save the output.")
    args = parser.parse_args()

    # load images
    dataset = Dataset([args.dataset])
    filenames = [os.path.basename(fp) for fp in dataset.fps] # will be used during saving
    nb_images = len(dataset.fps)
    X = np.zeros((nb_images, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3), dtype=np.float32)
    paths = []
    for i, (fp, image) in enumerate(zip(dataset.fps, dataset.get_images())):
        #image.square()
        image.resize(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
        X[i] = image.image_arr / 255.0
        paths.append(fp)
    X = np.rollaxis(X, 3, 1)

    # create model
    model = create_model_tiny(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, "mse", Adam())
    model.load_weights(args.weights)

    # predict positions of faces
    preds = model.predict(X, batch_size=BATCH_SIZE)

    # Calculate exact coordinates of faces on original images
    """
    coords = []
    for idx, (y, x, half_height, half_width) in enumerate(preds):
        orig_height = X[idx].shape[1]
        orig_width = X[idx].shape[2]
        tl_y, tl_x, br_y, br_x = unnormalize_prediction(y, x, half_height, half_width, img_height=orig_height, img_width=orig_width)

        coords.append((tl_y, tl_x, br_y, br_x))

    # Save images
    print("Saving images...")
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coords):
        print(X.shape, X[idx].shape)
        img = draw_rectangle(X[idx], tl_y, tl_x, br_y, br_x)
        filepath = os.path.join(WRITE_TO_DIR, filenames[idx])
        misc.imsave(filepath, img)
    """

    print("Saving images...")
    for idx, (y, x, half_height, half_width) in enumerate(preds):
        img = draw_predicted_rectangle(X[idx], y, x, half_height, half_width)
        filepath = os.path.join(WRITE_TO_DIR, filenames[idx])
        misc.imsave(filepath, img)

def get_images2(dirs):
    """Collects all images in given directories.
    Args:
        dirs: List of directories.
    Returns:
        List of images (numpy arrays).
    """
    paths = get_image_paths(dirs)
    result = []
    for path in paths:
        # neccessary to use ndimage instead of misc.imread, because there are black and white
        # images and they always get flattened by misc.imread (losing their color dimension)
        image = ndimage.imread(path, mode="RGB")
        result.append(image)
    return result, paths

def get_image_paths2(dirs):
    """Collects filepaths of all images in given directories.
    Args:
        dirs: List of directories.
    Returns:
        List of filepaths
    """
    result = []
    for fp_dir in dirs:
        fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
        fps = [os.path.join(fp_dir, f) for f in fps]
        fps_img = [fp for fp in fps if re.match(r".*\.(?:jpg|jpeg|png|gif)$", fp)]
        result.extend(fps_img)
    return result

if __name__ == "__main__":
    main()
