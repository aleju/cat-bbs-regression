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
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from train_hog_svm import get_crops_with_labels

np.random.seed(42)
random.seed(42)

MODEL_IMAGE_HEIGHT = 512
MODEL_IMAGE_WIDTH = 512
CROP_HEIGHT = 64
CROP_WIDTH = 64
NB_CROPS = 50000
NB_VALIDATION = 2048
NB_AUGMENTATIONS = 0
NB_CROPS_PER_IMAGE = 10 # max
CAT_FRACTION_THRESHOLD = 0.8
NB_COMPONENTS = 256 # pca components

def main():
    """Load images, train classifier, score classifier."""
    parser = argparse.ArgumentParser(description="Train an SVM model to locate cat faces in images based on PCA representations.")
    parser.add_argument("--dataset", required=True, help="Path to your 10k cats dataset directory")
    args = parser.parse_args()

    # initialize dataset
    subdir_names = ["CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04", "CAT_05", "CAT_06"]
    subdirs = [os.path.join(args.dataset, subdir) for subdir in subdir_names]
    dataset = Dataset(subdirs)

    # load images and labels
    print("Loading examples...")
    X, y = load_xy(dataset, NB_CROPS, NB_AUGMENTATIONS, NB_COMPONENTS)
    y[y >= CAT_FRACTION_THRESHOLD] = 1
    y[y < CAT_FRACTION_THRESHOLD] = 0
    assert X.dtype == np.float32
    print("X min:", np.min(X))
    print("X max:", np.max(X))
    print("y min=%.2f, max=%.2f, avg=%.2f, iavg=%.2f, median=%.2f" % (np.min(y), np.max(y), np.average(y), 1-np.average(y), np.median(y)))

    # split train and val
    X_val, X_train = X[0:NB_VALIDATION, ...], X[NB_VALIDATION:, ...]
    y_val, y_train = y[0:NB_VALIDATION, ...], y[NB_VALIDATION:, ...]
    print("%d of %d values in y_train are 1, %d of %d values in y_val" % (np.count_nonzero(y_train), y_train.shape[0], np.count_nonzero(y_val), y_val.shape[0]))
    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    print("Training...")
    # class_weight="balanced" for sklearn 0.18+
    svc = SVC(C=1000000, class_weight="auto")
    svc.fit(X_train, y_train)
    print("Found %d support vectors" % (len(svc.support_vectors_)))

    print("Predictions...")
    preds = svc.predict(X_val)
    for i in range(preds.shape[0]):
        print("%d: pred=%.2f, label=%.2f" % (i, preds[i], y_val[i]))

    print("Scoring...")
    acc = svc.score(X_val, y_val)
    print("accuracy = %.4f" % (acc))

def load_xy(dataset, nb_crops_max, nb_augmentations, nb_components):
    images, y = load_crops(dataset, nb_crops_max, nb_augmentations)
    images = images.reshape(images.shape[0], CROP_HEIGHT * CROP_WIDTH)
    pca = PCA(nb_components)
    X = pca.fit_transform(images)
    X = scale_linear_bycolumn(X)
    return X, y

def load_crops(dataset, nb_crops_max, nb_augmentations):
    X = []
    y = []

    examples = get_crops_with_labels(dataset, nb_crops_max, nb_augmentations,
                                     nb_crops_per_image=NB_CROPS_PER_IMAGE,
                                     model_image_height=MODEL_IMAGE_HEIGHT,
                                     model_image_width=MODEL_IMAGE_WIDTH,
                                     crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)

    for i, (crop, face_factor) in enumerate(examples):
        if i % 100 == 0:
            print("Crop %d of %d" % (i+1, nb_crops_max))

        X.append(crop)
        y.append(face_factor)

    # all entries in X/Y same length
    assert len(set([len(row) for row in X])) == 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

def scale_linear_bycolumn(rawpoints, low=0.0, high=1.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

if __name__ == "__main__":
    main()
