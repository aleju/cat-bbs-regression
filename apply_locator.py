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
import re
import numpy as np
import argparse
import random
from scipy import ndimage
from scipy import misc
from train import MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, BATCH_SIZE, \
                  SAVE_WEIGHTS_CHECKPOINT_FILEPATH, create_model, create_model_tiny, \
                  draw_predicted_rectangle
from keras.optimizers import Adam

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
    parser = argparse.ArgumentParser(description="Apply a trained cat face locator " \
                                                  "model to images.")
    parser.add_argument("--dataset", required=True, help="Path to the images directory.")
    parser.add_argument("--weights", required=False, default=SAVE_WEIGHTS_CHECKPOINT_FILEPATH,
                        help="Filepath to the weights of the model.")
    parser.add_argument("--output", required=False, default="apply_model_output",
                        help="Filepath to the directory in which to save the output.")
    args = parser.parse_args()

    # load images
    filepaths = get_image_filepaths([args.dataset])
    filenames = [os.path.basename(fp) for fp in filepaths] # will be used during saving
    nb_images = len(filepaths)
    X = np.zeros((nb_images, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3), dtype=np.float32)
    for i, fp in enumerate(filepaths):
        image = ndimage.imread(fp, mode="RGB")
        image = misc.imresize(image, (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))
        X[i] = image / 255.0
    X = np.rollaxis(X, 3, 1)

    # assure that dataset is not empty
    assert X.shape[0] > 0, X.shape

    # create model
    model = create_model(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, "mse", Adam())
    model.load_weights(args.weights)

    # predict positions of faces
    preds = model.predict(X, batch_size=BATCH_SIZE)

    # Draw predicted rectangles and save
    print("Saving images...")
    for idx, (y, x, half_height, half_width) in enumerate(preds):
        img = draw_predicted_rectangle(X[idx], y, x, half_height, half_width)
        filepath = os.path.join(WRITE_TO_DIR, filenames[idx])
        misc.imsave(filepath, img)

def get_image_filepaths(dirs):
    """Loads filepaths of images from dataset.
    Args:
        dirs    List of directories as strings
    Returns:
        List of strings (filepaths)"""
    result_img = []
    for fp_dir in dirs:
        fps = [f for f in os.listdir(fp_dir) if os.path.isfile(os.path.join(fp_dir, f))]
        fps = [os.path.join(fp_dir, f) for f in fps]
        fps_img = [fp for fp in fps if re.match(r".*\.jpg$", fp)]
        result_img.extend(fps_img)
    return result_img

if __name__ == "__main__":
    main()
