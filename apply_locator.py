# -*- coding: utf-8 -*-
"""
File to apply the trained ConvNet model to a number of images.
It will use the ConvNet to locate the cat faces in the images, extract these faces,
augment them (i.e. rotate, translate...) and then save them (i.e. only the faces).
It is expected that each image contains a cat (i.e. a face will be extracted of each image, even
if there is no cat).
If an image contains multiple cats, only one face will be extracted.

Usage:
    python train_cat_face_locator.py
    python apply_locator.py

Note:
    You should change the constants 'SOURCE_DIR' and 'TARGET_DIR' to your settings.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import random
import re
from scipy import misc
from scipy import ndimage
from cat_face_locator import MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, GRAYSCALE,
                             SAVE_WEIGHTS_FILEPATH, create_model, create_model_tiny,
                             predict_on_images, square_image, visualize_rectangle
from saveload import load_weights_seq
from keras.optimizers import Adam
from ImageAugmenter import ImageAugmenter

os.sys.setrecursionlimit(10000)
np.random.seed(42)
random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
WRITE_TO_DIR = os.path.join(CURRENT_DIR, "apply_locator_output")

# faces with a total area below this value will not be saved
MINIMUM_AREA = 8 * 8

# scale (height, width) of each saved image
OUT_SCALE = 64

def main():
    """
    Main function.
    Does the following step by step:
    * Load images (from which to extract cat faces) from SOURCE_DIR
    * Initialize model (as trained via train_cat_face_locator.py)
    * Prepares images for the model (i.e. shrinks them, squares them)
    * Lets model locate cat faces in the images
    * Projects face coordinates onto original images
    * Squares the face rectangles (as we want to get square images at the end)
    * Extracts faces from images with some pixels of padding around theM
    * Augments each face image several times
    * Removes the padding from each face image
    * Resizes each face image to OUT_SCALE (height, width)
    * Saves each face image (unaugmented + augmented images)
    """
    parser = argparse.ArgumentParser(description="Apply a trained cat face locator model images.")
    parser.add_argument("--images", required=True, help="Path to the images directory.")
    parser.add_argument("--weights", required=False, default="cat_face_locator.best.weights", help="Filepath to the weights of the model.")
    parser.add_argument("--output", required=False, default="apply_model_output", help="Filepath to the directory in which to save the output.")
    args = parser.parse_args()

    # --------------
    # load images
    # --------------
    dataset = Dataset([args.images])
    filenames = [os.basename(fp) for fp in dataset.fps] # will be used during saving
    nb_images = len(dataset.fps)
    X = np.zeros((nb_images, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3), dtype=np.float32)
    paths = []
    for i, fp, image in enumerate(zip(dataset.fps, dataset.get_images())):
        #image.square()
        image.resize(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
        X[i] = image.image_arr / 255.0
        paths.append(fp)
    X = np.rollaxis(X, 3, 1)

    # --------------
    # create model
    # --------------
    #model = create_model_tiny(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, Adam())
    model = create_model(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, "mse", Adam())
    model.load_weights(args.weights)

    # --------------
    # predict positions of faces
    # --------------
    preds = predict_on_images(model, images_padded)

    # --------------
    # Calculate exact coordinates of faces on original images
    # --------------
    coords = []
    for idx, (y, x, half_height, half_width) in enumerate(preds):
        orig_height = X[idx].shape[2]
        orig_width = X[idx].shape[3]
        tl_y, tl_x, br_y, br_x = unnormalize_prediction(y, x, half_height, half_width, img_height=orig_height, img_width=orig_width)
        """keypoints = np.zeros((9*2,), dtype=np.uint16)
        keypoints[0:4] = [tl_y, tl_x, br_y, br_x]
        image = ImageWithKeypoints(X[idx], Keypoints(keypoints))
        image.resize()"""

        coords.append((tl_y, tl_x, br_y, br_x))

    # --------------
    # Save images
    # --------------
    print("Saving images...")
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coords):
        img = draw_rectangle(X[idx], tl_y, tl_x, br_y, br_x)
        filepath = os.path.join(WRITE_TO_DIR, filenames[idx])
        misc.imsave(filepath, img)

def draw_rectangle(img, tl_y, tl_x, br_y, br_x):
    img = np.copy(img)
    lines = [
        (tl_y, tl_x, tl_y, br_x), # top left to top right
        (tl_y, br_x, br_y, br_x), # top right to bottom right
        (br_y, br_x, br_y, tl_x), # bottom right to bottom left
        (br_y, tl_x, tl_y, tl_x)  # bottom left to top left
    ]
    for y0, x0, y1, x1 in lines:
        rr, cc, val = draw.line_aa(y0, x0, y1, y1)
        img[:, rr, cc] = val * 255

    return img

def get_images(dirs):
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

def get_image_paths(dirs):
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
