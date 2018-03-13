from keras.layers import Input,Dense, Conv2D, MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as k
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test), 28,28,1))
print(x_train.shape)
print(x_test.shape)

input_img = Input(shape=(28,28,1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(1, (3,3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(1, (3,3), activation='relu', padding='same')(encoded)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
autoencoder.summary()

tb = TensorBoard(log_dir='')

autoencoder.fit(x_train,x_train, epochs=50, batch_size=128,
               shuffle=True, validation_data=(x_test,x_test),
               callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])