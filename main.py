import os
from data_load import get_data
import model
import numpy as np
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from plot import plotting, plot_sample
from keras.optimizers import Adam

# %matplotlib inline

from sklearn.model_selection import train_test_split

im_width = 128
im_height = 128
root_path = os.getcwd()
path_train = root_path+'\\input\\train\\'
path_test = root_path+'\\input\\test\\'


X, y = get_data(path_train, train=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)


def train():
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-bottles.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    input_img = Input((im_height, im_width, 1), name='img')
    unet = model.get_unet( input_img, n_filters=16, dropout=0.05, batchnorm=True)
    unet.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    # model.summary()
    results = unet.fit(X_train, y_train, batch_size=8, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))

    plotting(results)
    return unet

def predict():
    input_img = Input((im_height, im_width, 1), name='img')
    unet = model.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    unet.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    unet.load_weights('model-bottles.h5')
    # unet.evaluate(X_valid, y_valid, verbose=1)
    preds_train = unet.predict(X_train, verbose=1)
    preds_val = unet.predict(X_valid, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    # plot_sample(X_train, y_train, preds_train, preds_train_t, ix=None)
    plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=None)


# train()
predict()