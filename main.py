import os
from data_load import get_data, get_test_generator, get_train_generator
import model
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from plot import plotting, plot_sample


# %matplotlib inline

from sklearn.model_selection import train_test_split

im_width = 256
im_height = 256
root_path = os.getcwd()
path_train = root_path+'\\input\\train\\'
path_test = root_path+'\\input\\test\\'





X, y = get_data(path_train, im_width, im_height, train=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)
train_gen = get_train_generator(path_train, 8)
test_gen = get_test_generator(path_test,2)

def train():
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        # ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-bottles_256.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    unet = model.get_unet_v2(im_height,im_width)
    results = unet.fit(X_train, y_train,  epochs=100, callbacks=callbacks, validation_data=(X_valid, y_valid))
    plotting(results)


def train_wth_gen():
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        # ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-bottles_gen.h5', verbose=1,monitor='loss', save_best_only=True)
    ]
    unet = model.get_unet_v2(im_height, im_width)
    results = unet.fit_generator(train_gen, steps_per_epoch=10, epochs=5, callbacks=callbacks)
    # plotting(results)

def predict():

    unet = model.get_unet_v2(im_height,im_width)
    unet.load_weights('model-bottles_gen.h5')

    preds_train = unet.predict(X_train, verbose=1)
    preds_val = unet.predict(X_valid, verbose=1)

    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)

    plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=None)

    # testGene = testGenerator(path_test,1)
    # results = unet.predict_generator(testGene, 1, verbose=1)
    # saveResult(path_test, results)


def predict_wth_generator():

    unet = model.get_unet_v2(im_height, im_width)
    unet.load_weights('model-bottles_gen.h5')
    results = unet.predict_generator(test_gen, 2, verbose=1)
    print("")

# train_wth_gen()
# train()
# predict()
predict_wth_generator()\
