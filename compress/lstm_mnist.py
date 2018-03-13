from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input,Dense, Conv2D, MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as k
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 784, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 784, 1))

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     ax = plt.subplot(1, n, i+1)
#     plt.imshow(x_train[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

timesteps = 784
input_dim = 1
latent_dim = 32

inputs = Input(shape=(timesteps, input_dim))

encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

autoencoder.summary()

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('lstm_mnist.png')