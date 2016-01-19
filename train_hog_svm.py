from __future__ import absolute_import, division, print_function
from dataset import Dataset
import numpy as np
import argparse
import random
import os

np.random.seed(42)
random.seed(42)

MODEL_IMAGE_HEIGHT = 256
MODEL_IMAGE_WIDTH = 256
CROP_HEIGHT = 32
CROP_WIDTH = 32
NB_LOAD_IMAGES = 5000

def main():
    # initialize dataset
    subdir_names = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
    subdirs = [os.path.join(args.dataset, subdir) for subdir in subdir_names]
    dataset = Dataset(subdirs)

    # load images and labels
    print("Loading images...")
    X, y = load_xy(dataset, NB_LOAD_IMAGES)

    # split train and val
    nb_images = X.shape[0]
    nb_train = int(nb_images * (1 - SPLIT))
    X_train = X[0:nb_train, ...]
    y_train = y[0:nb_train, ...]
    X_val = X[nb_train:, ...]
    y_val = y[nb_train:, ...]

def load_xy(dataset, nb_load, nb_augmentations):
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
    i = 0
    nb_crops_per_image = floor(MODEL_IMAGE_HEIGHT / CROP_HEIGHT) * floor(MODEL_IMAGE_WIDTH / CROP_WIDTH)
    nb_load = min(nb_load, len(dataset.fps) * nb_crops_per_image)
    nb_images = nb_load + nb_load * nb_augmentations
    X = np.zeros((nb_images, CROP_HEIGHT, CROP_WIDTH), dtype=np.float32)
    y = np.zeros((nb_images, 1), dtype=np.float32)

    for img_idx, image in enumerate(dataset.get_images()):
        if img_idx % 100 == 0:
            print("Loading image %d of %d..." % (img_idx+1, nb_load))
        image.resize(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
        image.pad(PADDING)
        augs = image.augment(nb_augmentations, hflip=True, vflip=False,
                             scale_to_percent=(0.9, 1.1), scale_axis_equally=False,
                             rotation_deg=10, shear_deg=0, translation_x_px=5, translation_y_px=5,
                             brightness_change=0.1, noise_mean=0.0, noise_std=0.05)
        for aug in [image] + augs:
            aug.unpad(PADDING)
            for crop, face_factor in create_crops(aug):
                crop_hog = hog(crop, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), normalize=True, feature_vector=True,
                               visualise=False)
                X[i] = crop_hog
                y[i] = 1 if face_factor >= 0.2 else 0
                i += 1

                if (i + 1) >= nb_images:
                    break

    X = np.rollaxis(X, 3, 1)

    return X, y

def create_crops(img):
    from skimage import color

    img_arr = color.rgb2gray(img.to_array())
    img_face = np.zeros(img_arr.shape, dtype=np.boolean)

    face_rect = img.keypoints.get_rectangle(img)
    rect_tl = face_rect.tl_y
    img_face[face_rect.tl_y:face_rect.br_y+1, face_rect.tl_x:face_rect.br_x+1] = 1

    nb_crops_y = floor(MODEL_IMAGE_HEIGHT / CROP_HEIGHT)
    nb_crops_x = floor(MODEL_IMAGE_WIDTH / CROP_WIDTH)
    nb_crops = nb_crops_y * nb_crops_x

    for i in range(nb_crops):
        grid_y = i // nb_crops_x
        grid_x = i % nb_crops_x

        crop_tl_y = MODEL_IMAGE_HEIGHT * (CROP_HEIGHT * grid_y)
        crop_br_y = MODEL_IMAGE_HEIGHT * (CROP_HEIGHT * (grid_y + 1))
        crop_tl_x = MODEL_IMAGE_WIDTH * (CROP_WIDTH * grid_x)
        crop_br_x = MODEL_IMAGE_WIDTH * (CROP_WIDTH * (grid_y + 1))

        img_arr_crop = img_arr[crop_tl_y:crop_br_y, crop_tl_x:crop_br_x]
        img_face_crop = img_face[crop_tl_y:crop_br_y, crop_tl_x:crop_br_x]
        face_px = np.nonzero(img_face_crop)
        face_factor = face_px / (CROP_HEIGHT * CROP_WIDTH)

        yield img_arr_crop, face_factor

if __name__ == "__main__":
    main()
