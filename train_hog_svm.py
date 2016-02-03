"""
Script to train an SVM classifier to locate cat faces in images.
The classifiers training data consists of HOGs of crops of images (windows).
It has to predict 1 (contains cat) whenever the fraction of pixels that
show a cat (in a window) is above a threshold (20 percent by default).
Usage:
    python train_hog_svm.py --dataset="/path/to/images-directory"
"""
from __future__ import absolute_import, division, print_function
from dataset import Dataset
import numpy as np
import argparse
import random
import os
from skimage import color
from skimage.feature import hog
from sklearn.svm import SVC

np.random.seed(42)
random.seed(42)

SPLIT = 0.1
MODEL_IMAGE_HEIGHT = 512
MODEL_IMAGE_WIDTH = 512
CROP_HEIGHT = 128
CROP_WIDTH = 128
NB_LOAD_IMAGES = 60000
CAT_FRACTION_THRESHOLD = 0.4
PADDING = 20

def main():
    """Load images, train classifier, score classifier."""
    parser = argparse.ArgumentParser(description="Train an SVM model to locate cat faces in images.")
    parser.add_argument("--dataset", required=True, help="Path to your 10k cats dataset directory")
    args = parser.parse_args()

    # initialize dataset
    subdir_names = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
    subdirs = [os.path.join(args.dataset, subdir) for subdir in subdir_names]
    dataset = Dataset(subdirs)

    # load images and labels
    print("Loading images...")
    X, y = load_xy(dataset, NB_LOAD_IMAGES, 0)
    assert X.dtype == np.float32
    assert np.max(X) <= 1.0
    assert np.min(X) >= 0.0

    # split train and val
    nb_images = X.shape[0]
    nb_train = int(nb_images * (1 - SPLIT))
    X_train = X[0:nb_train, ...]
    y_train = y[0:nb_train, ...]
    X_val = X[nb_train:, ...]
    y_val = y[nb_train:, ...]
    print("%d of %d values in y_train are 1, %d of %d values in y_val" % (np.count_nonzero(y_train), y_train.shape[0], np.count_nonzero(y_val), y_val.shape[0]))

    print("Training...")
    svc = SVC(C=0.001)
    svc.fit(X_train, y_train)

    print("Predictions...")
    preds = svc.predict(X_val)
    for i in range(preds.shape[0]):
        print(i, preds[i])

    print("Scoring...")
    acc = svc.score(X_val, y_val)
    print("accuracy = %.4f" % (acc))

def load_xy(dataset, nb_crops_max, nb_augmentations):
    """Loads X and y (examples with labels) for the dataset.
    Examples are HOGs of 32x32 crops of 256x256 images.
    Labels are 0/1 whether the crop contains a cat.

    Args:
        dataset            The Dataset object.
        nb_load            Intended number of images to load.
        nb_augmentations   Number of augmentations to perform.
    Returns:
        X (numpy array of shape (N, 32*32=1024)),
        y (numpy array of shape (N, 1))
    """
    nb_crops_per_image = (MODEL_IMAGE_HEIGHT // CROP_HEIGHT) * (MODEL_IMAGE_WIDTH // CROP_WIDTH)
    nb_crops_possible = len(dataset.fps) * nb_crops_per_image
    nb_crops_max = min(nb_crops_max, nb_crops_possible)

    X = np.zeros((nb_crops_max, 8100), dtype=np.float32)
    y = np.zeros((nb_crops_max,), dtype=np.float32)

    #print("nb_crops_per_image=", nb_crops_per_image, "nb_load=", nb_load, "nb_crops=", nb_crops)
    nb_crops_added = 0
    for i, (crop, face_factor) in enumerate(get_examples_with_labels(dataset, nb_crops_max, nb_augmentations)):
        if nb_crops_added % 100 == 0:
            print("Crop %d of %d" % (nb_crops_added+1, nb_crops_max))
        crop_hog = hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), normalise=True)
        #crop_hog, vis = hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), normalise=True, visualise=True)
        #from scipy import misc
        #misc.imshow(crop)
        #misc.imshow(vis)
        crop_hog[crop_hog < 0] = 0 # the hog values can rarely end up slightly below 0
        is_cat = True if face_factor >= CAT_FRACTION_THRESHOLD else False
        #if is_cat or random.random() < 0.25:
        X[nb_crops_added] = crop_hog
        y[nb_crops_added] = 1 if is_cat else 0
        nb_crops_added += 1

    return X, y

def get_examples_with_labels(dataset, nb_crops_max, nb_augmentations,
                             model_image_height=MODEL_IMAGE_HEIGHT,
                             model_image_width=MODEL_IMAGE_WIDTH,
                             crop_height=CROP_HEIGHT,
                             crop_width=CROP_WIDTH,
                             padding=PADDING,
                             drop_nonface_prob=0.0):
    """Loads X and y (examples with labels) for the dataset.
    Examples are HOGs of 32x32 crops of 256x256 images.
    Labels are 0/1 whether the crop contains a cat.

    Args:
        dataset            The Dataset object.
        nb_load            Intended number of images to load.
        nb_augmentations   Number of augmentations to perform.
    Returns:
        X (numpy array of shape (N, 32*32=1024)),
        y (numpy array of shape (N, 1))
    """
    nb_crops_loaded = 0
    for img_idx, image in enumerate(dataset.get_images()):
        image.resize(model_image_height, model_image_width)
        image.pad(padding)
        augs = image.augment(nb_augmentations, hflip=True, vflip=False,
                             scale_to_percent=(0.9, 1.1), scale_axis_equally=False,
                             rotation_deg=10, shear_deg=0, translation_x_px=5, translation_y_px=5,
                             brightness_change=0.1, noise_mean=0.0, noise_std=0.05)
        for aug in [image] + augs:
            aug.unpad(padding)
            for crop, face_factor in image_to_crops(aug, crop_height, crop_width):
                if drop_nonface_prob == 0.0 or random.random() > drop_nonface_prob:
                    yield crop, face_factor
                    nb_crops_loaded += 1

                if nb_crops_loaded >= nb_crops_max:
                    break
            if nb_crops_loaded >= nb_crops_max:
                break
        if nb_crops_loaded >= nb_crops_max:
            break


def image_to_crops(img, crop_height, crop_width):
    """Extracts all NxM grayscale crops (patches) from a given image.
    Args:
        img         The image to crop
        crop_height Height of the crop
        crop_width  Width of the crop
    Returns:
        Generator of (crop as numpy array, cat fraction as float 0.0 to 1.0)
    """
    img_arr = color.rgb2gray(img.to_array())
    img_face = np.zeros(img_arr.shape, dtype=np.bool_)

    face_rect = img.keypoints.get_rectangle(img)
    rect_tl = face_rect.tl_y
    img_face[face_rect.tl_y:face_rect.br_y+1, face_rect.tl_x:face_rect.br_x+1] = 1

    height = img_arr.shape[0]
    width = img_arr.shape[1]
    nb_crops_y = height // crop_height
    nb_crops_x = width // crop_width
    nb_crops = nb_crops_y * nb_crops_x

    crop_tl_y = 0
    for grid_y in range(nb_crops_y):
        crop_tl_x = 0
        for grid_x in range(nb_crops_x):
            crop_br_y = crop_tl_y + crop_height
            crop_br_x = crop_tl_x + crop_width

            img_arr_crop = img_arr[crop_tl_y:crop_br_y, crop_tl_x:crop_br_x]
            img_face_crop = img_face[crop_tl_y:crop_br_y, crop_tl_x:crop_br_x]
            face_px = np.count_nonzero(img_face_crop)
            face_factor = face_px / (crop_height * crop_width)

            yield img_arr_crop, face_factor

            crop_tl_x += crop_width
        crop_tl_y += crop_height

if __name__ == "__main__":
    main()
