import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

np.save("mnist_x_train", x_train)
np.save("mnist_x_test", x_test)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoding_dim = 2

encoder_input_layer = Input(shape=(784,), name="encoder_input")
e = Dense(256, activation="relu")(encoder_input_layer)
e = Dense(64, activation="relu")(e)
e = Dense(16, activation="relu")(e)
encoder_output_layer = Dense(encoding_dim, activation="relu", name="encoder_output")(e)

encoder = Model(encoder_input_layer, encoder_output_layer, name="encoder")

print(encoder.summary())

decoder_input_layer = Input(shape=(encoding_dim,), name="decoder_input")
d = Dense(16, activation="relu")(decoder_input_layer)
d = Dense(64, activation="relu")(d)
d = Dense(256, activation="relu")(d)
decoder_output_layer = Dense(784, activation="sigmoid", name="decoder_output")(d)

decoder = Model(decoder_input_layer, decoder_output_layer, name="decoder")
print(decoder.summary())

ae_input_layer = encoder_input_layer
ae_output_layer = decoder(encoder(encoder_input_layer))

autoencoder = Model(ae_input_layer, ae_output_layer, name="autoencoder")
print(autoencoder.summary())
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)
autoencoder.save("AE.h5")


encoder_ae = Model(autoencoder.input, autoencoder.get_layer("encoder").output)
encoder_ae.compile(optimizer="adam", loss="binary_crossentropy")
encoder_ae.save("AE_encoder.h5")


decoder_ae = Model(
    autoencoder.get_layer("decoder").input, autoencoder.get_layer("decoder").output
)
decoder_ae.compile(optimizer="adam", loss="binary_crossentropy")
decoder_ae.save("AE_decoder.h5")


model = load_model("AE.h5")
model.evaluate(x_test, x_test)
decoded_imgs = model.predict(x_test)

n = 10
plt.figure(figsize=(10, 2))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

model = load_model("AE.h5")
tf.saved_model.save(model, "tmp_model_AE")
model = load_model("AE_encoder.h5")
tf.saved_model.save(model, "tmp_model_AE_encoder")
model = load_model("AE_decoder.h5")
tf.saved_model.save(model, "tmp_model_AE_decoder")
