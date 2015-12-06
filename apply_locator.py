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
    nb_images = len(dataset.fps)
    X = np.zeros((nb_images, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3), dtype=np.float32)
    paths = []
    for i, fp, image in enumerate(zip(dataset.fps, dataset.get_images())):
        image.square()
        image.resize(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
        X[i] = image.image_arr / 255.0
        paths.append(fp)
    X = np.rollaxis(X, 3, 1)

    # --------------
    # create model
    # --------------
    #model = create_model_tiny(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, Adam())
    model = create_model(MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, Adam())
    model.load_weights(args.weights)

    # --------------
    # predict positions of faces
    # --------------
    coordinates_predictions = predict_on_images(model, images_padded)

    print("[Predicted positions]", coordinates_predictions[0])
    """
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coordinates_predictions):
        marked_image = visualize_rectangle(images_padded[idx]*255, tl_x, br_x, tl_y, br_y, \
                                           (255,), channel_is_first_axis=True)
        misc.imshow(marked_image)
    """

    # --------------
    # project coordinates from small padded images to full-sized original images (without padding)
    # --------------
    coordinates_orig = []
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coordinates_predictions):
        pad_top, pad_right, pad_bottom, pad_left = paddings[idx]
        height_full = images[idx].shape[0] + pad_top + pad_bottom
        width_full = images[idx].shape[1] + pad_right + pad_left
        height_orig = images[idx].shape[0]
        width_orig = images[idx].shape[1]

        tl_y_perc = tl_y / MODEL_IMAGE_HEIGHT
        tl_x_perc = tl_x / MODEL_IMAGE_WIDTH
        br_y_perc = br_y / MODEL_IMAGE_HEIGHT
        br_x_perc = br_x / MODEL_IMAGE_WIDTH

        # coordinates on full sized squared image version
        tl_y_full = int(tl_y_perc * height_full)
        tl_x_full = int(tl_x_perc * width_full)
        br_y_full = int(br_y_perc * height_full)
        br_x_full = int(br_x_perc * width_full)

        # remove paddings to get coordinates on original images
        tl_y_orig = tl_y_full - pad_top
        tl_x_orig = tl_x_full - pad_left
        br_y_orig = br_y_full - pad_top
        br_x_orig = br_x_full - pad_left

        # fix broken coordinates
        # anything below 0
        # anything above image height (y) or width (x)
        # anything where top left >= bottom right
        tl_y_orig = min(max(tl_y_orig, 0), height_orig)
        tl_x_orig = min(max(tl_x_orig, 0), width_orig)
        br_y_orig = min(max(br_y_orig, 0), height_orig)
        br_x_orig = min(max(br_x_orig, 0), width_orig)

        if tl_y_orig >= br_y_orig:
            tl_y_orig = br_y_orig - 1
        if tl_x_orig >= br_x_orig:
            tl_x_orig = br_x_orig - 1

        coordinates_orig.append((tl_y_orig, tl_x_orig, br_y_orig, br_x_orig))

    """
    # project face coordinates to original image sizes
    coordinates_orig = []
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coordinates_nopad):
        height_orig = images[idx].shape[0]
        width_orig = images[idx].shape[1]

        tl_y_perc = tl_y / MODEL_IMAGE_HEIGHT
        tl_x_perc = tl_x / MODEL_IMAGE_WIDTH
        br_y_perc = br_y / MODEL_IMAGE_HEIGHT
        br_x_perc = br_x / MODEL_IMAGE_WIDTH

        tl_y_orig = int(tl_y_perc * height_orig)
        tl_x_orig = int(tl_x_perc * width_orig)
        br_y_orig = int(br_y_perc * height_orig)
        br_x_orig = int(br_x_perc * width_orig)

        coordinates_orig.append((tl_y_orig, tl_x_orig, br_y_orig, br_x_orig))

    print("[Coordinates on original image]", coordinates_orig[0])

    # remove padding from predicted face coordinates
    # tl = top left, br = bottom right
    coordinates_nopad = []
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coordinates_predictions):
        pad_top, pad_right, pad_bottom, pad_left = paddings[idx]
        tl_y_nopad = tl_y - pad_top
        tl_x_nopad = tl_x - pad_left
        br_y_nopad = br_y - pad_top
        br_x_nopad = br_x - pad_left
        tpl = (tl_y_nopad, tl_x_nopad, br_y_nopad, br_x_nopad)
        tpl_fixed = [max(coord, 0) for coord in tpl]
        if tpl_fixed[0] >= tpl_fixed[2]:
            tpl_fixed[2] += 1
        elif tpl_fixed[1] >= tpl_fixed[3]:
            tpl_fixed[3] += 1
        tpl_fixed = tuple(tpl_fixed)

        if tpl != tpl_fixed:
            print("[WARNING] Predicted coordinate below 0 after padding-removel. Bad prediction." \
                  " (In image %d, coordinates nopad: %s, coordinates pred: %s)" \
                  % (idx, tpl, coordinates_predictions[idx]))

        coordinates_nopad.append(tpl_fixed)
    """

    print("[Removed padding from predicted coordinates]", coordinates_orig[0])

    # --------------
    # square faces
    # --------------
    coordinates_orig_square = []
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coordinates_orig):
        height = br_y - tl_y
        width = br_x - tl_x
        i = 0
        # we remove here instead of adding rows/cols, because that way we wont exceed the
        # image maximum sizes
        while height > width:
            if i % 2 == 0:
                tl_y += 1
            else:
                br_y -= 1
            height -= 1
            i += 1
        while width > height:
            if i % 2 == 0:
                tl_x += 1
            else:
                br_x -= 1
            width -= 1
            i += 1
        print("New height:", (br_y-tl_y), "New width:", (br_x-tl_x))
        coordinates_orig_square.append((tl_y, tl_x, br_y, br_x))

    print("[Squared face coordinates]", coordinates_orig_square[0])

    # --------------
    # pad faces
    # --------------
    # extract "padded" faces, where the padding is part of the original image
    # (N pixels around the face)
    # After doing that, we can augment the "padded" faces, then remove the padding and have less
    # augmentation damage (i.e. areas that would otherwise be black will now be filled with parts
    # of the original image)
    faces_padded = []
    for idx, (tl_y, tl_x, br_y, br_x) in enumerate(coordinates_orig_square):
        image = images[idx]
        # we pad the whole image by N pixels so that we can savely extract an area of N pixels
        # around the face
        image_padded = np.pad(image, ((AUGMENTATION_PADDING, AUGMENTATION_PADDING), \
                                      (AUGMENTATION_PADDING, AUGMENTATION_PADDING), \
                                      (0, 0)), mode=str("median"))
        face_padded = image_padded[tl_y:br_y+2*AUGMENTATION_PADDING, \
                                   tl_x:br_x+2*AUGMENTATION_PADDING, \
                                   ...]
        faces_padded.append(face_padded)

    print("[Extracted face with padding]")
    misc.imshow(faces_padded[0])

    # --------------
    # augment and save images
    # --------------
    for idx, face_padded in enumerate(faces_padded):
        # these should be the same values for all images
        image_height = face_padded.shape[0]
        image_width = face_padded.shape[1]
        print("[specs of padded face] height", image_height, "width", image_width)

        # augment the padded images
        ia = ImageAugmenter(image_width, image_height,
                            channel_is_first_axis=False,
                            hflip=True, vflip=False,
                            scale_to_percent=(0.90, 1.10), scale_axis_equally=True,
                            rotation_deg=45, shear_deg=0,
                            translation_x_px=8, translation_y_px=8)
        images_aug = np.zeros((AUGMENTATION_ITERATIONS, image_height, image_width, 3),
                              dtype=np.uint8)
        for i in range(AUGMENTATION_ITERATIONS):
            images_aug[i, ...] = face_padded
        print("images_aug.shape", images_aug.shape)
        images_aug = ia.augment_batch(images_aug)

        # randomly change brightness of whole images
        for idx_aug, image_aug in enumerate(images_aug):
            by_percent = random.uniform(0.90, 1.10)
            images_aug[idx_aug] = np.clip(image_aug * by_percent, 0.0, 1.0)
        print("images_aug.shape [0]:", images_aug.shape)

        # add gaussian noise
        # skipped, because that could be added easily in torch as a layer
        #images_aug = images_aug + np.random.normal(0.0, 0.05, images_aug.shape)

        # remove the padding
        images_aug = images_aug[:,
                                AUGMENTATION_PADDING:-AUGMENTATION_PADDING,
                                AUGMENTATION_PADDING:-AUGMENTATION_PADDING,
                                ...]
        print("images_aug.shape [1]:", images_aug.shape)

        # add the unaugmented image
        images_aug = np.vstack((images_aug, \
                                [face_padded[AUGMENTATION_PADDING:-AUGMENTATION_PADDING, \
                                             AUGMENTATION_PADDING:-AUGMENTATION_PADDING, \
                                             ...]]))

        print("images_aug.shape [2]:", images_aug.shape)

        # save images
        for i, image_aug in enumerate(images_aug):
            if image_aug.shape[0] * image_aug.shape[1] < MINIMUM_AREA:
                print("Ignoring image %d / %d because it is too small (area of %d vs min. %d)" \
                       % (idx, i, image_aug.shape[0] * image_aug.shape[1], MINIMUM_AREA))
            else:
                image_resized = misc.imresize(image_aug, (OUT_SCALE, OUT_SCALE))
                filename_aug = "%s_%d.jpg" % (images_filenames[idx].replace(".jpg", ""), i)
                #misc.imshow(image_resized)
                misc.imsave(os.path.join(TARGET_DIR, filename_aug), image_resized)

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
