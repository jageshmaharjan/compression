from keras.layers import Input,Dense, Conv2D, MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as k
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import cv2
import os
from os import listdir

def image_to_feature_vector(img, size=(512,342)):         # Avg Dim 1798,1260;  size=(350,525)
    return cv2.resize(img, size).flatten()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='data file path', type=str)
    args = parser.parse_args()
    path = args.path

    onlyfiles = [f for f in listdir(path)]
    print(len(onlyfiles))

    data = []
    for f in onlyfiles[:500]:
        fp = os.path.join(path, f)
        x = cv2.imread(fp,1)
        x = image_to_feature_vector(x)
        x = x.astype('float32') / 255
        x = np.reshape(x, (342,512, 3))
        data.append(x)

    print(len(data))
    x_train = np.reshape(data[:399], (399, 342,512, 3))
    x_test = np.reshape(data[400:480], (80, 342,512, 3))
    x_val1id = np.reshape(data[481:500], (19, 342,512, 3))
    print(x_train[1].shape)

    input_img = Input(shape=(342,512,3))

    x = Conv2D(8, (3, 2), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((3, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3,4), activation='relu', padding='same')(x)

    encoded = MaxPooling2D((3, 4), padding='valid', name='last_enc')(x)

    x = UpSampling2D((3,4))(encoded)
    x = Conv2D(64, (3, 4), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((3, 2))(x)
    x = Conv2D(3, (3, 2), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(3, (2, 2), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(3, (3, 3), activation='relu')(x)
    # x = UpSampling2D((2, 2))(x)

    decoded = x #Conv2D(3, (2, 2), activation='sigmoid', padding='same', name='final_dec')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    autoencoder.summary()

    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(7,7,32))

    decoder_layer = autoencoder.layers[-4]
    decoder = Model(encoded_input, decoder_layer(encoded_input) )

    autoencoder.fit(x_train, x_train, epochs=2, batch_size=10,
                        shuffle=True, validation_data=(x_test, x_test))

    decoded_imgs = autoencoder.predict(x_test)