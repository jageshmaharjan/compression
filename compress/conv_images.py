from keras.layers import Input,Dense, Conv2D, MaxPooling2D,UpSampling2D
from keras.models import Sequential, Model
# from keras.callbacks import TensorBoard, ModelCheckpoint

from imutils import paths
import numpy as np
import argparse
import cv2
import os
from os import listdir


def image_to_feature_vector(img, size=(96,96)):         # Avg Dim 1798,1260
    return cv2.resize(img, size).flatten()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='data file path', type=str)
    args = parser.parse_args()
    path = args.path
    # path = '/home/jugs/PycharmProjects/compression/face/'

    onlyfiles = [f for f in listdir(path)]

    data = []
    for f in onlyfiles:
        fp = os.path.join(path, f)
        x = cv2.imread(fp)
        x = image_to_feature_vector(x)
        x = x.astype('float32') / 255
        x = np.reshape(x, (96, 96, 3))
        data.append(x)

    x_train = np.reshape(data[:5000], (5000, 96, 96, 3))
    x_test = np.reshape(data[5001:], (1936, 96, 96, 3))

    input_image = Input(shape=(96, 96, 3))

    x = Conv2D(12, (3, 3), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # x = Conv2D(48, (3,3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_image, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    autoencoder.summary()

    autoencoder.fit(x_train, x_train, epochs=50, batch_size=128,
                    shuffle=True, validation_data=(x_test, x_test))
    print(x)
