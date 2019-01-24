import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras_preprocessing.image import load_img, img_to_array
from tqdm import tqdm_notebook
from skimage.transform import resize
from generator import *

# Set some parameters
# im_width = 128
# im_height = 128
border = 5




# Get and resize train images and masks
def get_data(path, im_width, im_height, train=True):
    ids = next(os.walk(path + "images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_width, im_height, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, color_mode = "grayscale"))
            mask = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255  #TODO Why?
    print('Done!')
    if train:
        return X, y
    else:
        return X


def get_train_generator(path, batch_size):
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    return trainGenerator(batch_size, path, 'images', 'masks', data_gen_args)



def get_test_generator(path, batch_size):
    return testGenerator(path,batch_size)
