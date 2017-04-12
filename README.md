# About

Simple convolutional neural network to detect cat bounding boxes in images.
The system is restricted to one bounding box per image, which is localized using regression (i.e. directly predicting the bounding box coordinates).
The model consists of 7 convolutional layers and 2 fully connected layers (including output layer).

# Dependencies

* python 2.7 (only tested with that version)
* keras (tested in v1.06)
* scipy
* numpy
* scikit-image

# Usage

* Download the [10k cats dataset](https://web.archive.org/web/20150520175645/http://137.189.35.203/WebUI/CatDatabase/catData.html) and extract it, e.g. into directory `/foo/bar/10k-cats`. That directory should contain the subdirectories `CAT_00`, `CAT_01`, etc.
  * Dataset is available at 
* Train the model using `train_convnet.py --dataset="/foo/bar/10k-cats"`.
* Apply the model using `train_convnet.py --dataset="/foo/bar/directory-with-cat-images"`.

# Images

Example results:

![Located cat face](images/1-39-125521249_b1318298ec_n.jpg?raw=true "Located cat face")
![Located cat face](images/1-56-180653960_21cf28e0b3_n.jpg?raw=true "Located cat face")
![Located cat face](images/1-61-213767259_11c8550a0e_n.jpg?raw=true "Located cat face")
![Located cat face](images/1-266-19922494159_78303f8b16_n.jpg?raw=true "Located cat face")
![Located cat face](images/3-2287-2088100404_c0112197e3_n.jpg?raw=true "Located cat face")
![Located cat face](images/3-2831-10902603864_4993c4aa1a_n.jpg?raw=true "Located cat face")
