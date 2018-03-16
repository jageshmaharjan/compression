from keras.layers import Input,Dense, Conv2D, MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as k
from keras.callbacks import TensorBoard, ModelCheckpoint
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
    parser.add_argument('--path_1', help='data file path', type=str)
    parser.add_argument('--path_2', help='data file path', type=str)
    args = parser.parse_args()
    path_1 = args.path_1
    path_2 = args.path_2

    onlyfiles_train = [f for f in listdir(path_1)]
    print(len(onlyfiles_train))
    train_data = []
    for f in onlyfiles_train:
        fp = os.path.join(path_1, f)
        x = cv2.imread(fp,1)
        x = image_to_feature_vector(x)
        x = x.astype('float32') / 255
        x = np.reshape(x, (342,512, 3))
        train_data.append(x)

    onlyfiles_test = [f for f in listdir(path_2)]
    print(len(onlyfiles_test ))
    test_data = []
    for f in onlyfiles_test:
        fp = os.path.join(path_2, f)
        x = cv2.imread(fp,1)
        x = image_to_feature_vector(x)
        x = x.astype('float32') / 255
        x = np.reshape(x, (342,512, 3))
        test_data.append(x)

    x_train = np.reshape(train_data[:], (len(train_data), 342,512, 3))
    x_test = np.reshape(test_data[:], (len(test_data), 342,512, 3))
    # x_val1id = np.reshape(data[481:500], (19, 342,512, 3))
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


    tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    cp = ModelCheckpoint(filepath='./models', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    autoencoder.fit(x_train, x_train, epochs=2, batch_size=10,
                        shuffle=True, validation_data=(x_test, x_test),
                    callbacks=[cp,tb])

    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(342,512, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(342,512, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('pro_img.png')
