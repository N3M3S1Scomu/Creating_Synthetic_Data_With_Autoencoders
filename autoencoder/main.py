
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

################### DATASET ##########################
(x_train,_),(x_test,_)=mnist.load_data()

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

x_train=x_train.reshape(len(x_train),np.product(x_train.shape[1:]))
x_test=x_test.reshape(len(x_test),np.product(x_test.shape[1:]))

print(x_train.shape)
print(x_test.shape)

################### MODEL ##########################
encode_dims=32 # encoder boyutu
input_img=keras.Input(shape=(784)) # mnist dataset boyutu = 28x28

encoded=layers.Dense(encode_dims,activation="relu")(input_img) # encoder katmanı
decoded=layers.Dense(784,activation="sigmoid")(encoded)

autoencoder=keras.Model(input_img,decoded)

encoder=keras.Model(input_img,encoded)
encoded_input = keras.Input(shape=(encode_dims,))

decoded_layer=autoencoder.layers[-1]
decoder=keras.Model(encoded_input,decoded_layer(encoded_input))

autoencoder.compile(optimizer="adam",loss="binary_crossentropy")

autoencoder.fit(x_train,x_train,epochs=10,batch_size=256,validation_data=(x_test,x_test))

################## PREDICTION #######################
encoded_img=encoder.predict(x_test)
decoded_img=decoder.predict(encoded_img)

plt.figure()
plt.title("syntetic data")
plt.imshow(decoded_img[1].reshape(28,28))

plt.figure()
plt.title("original data")
plt.imshow(x_test[1].reshape(28,28))

plt.show()

"""
n=10

plt.figure(figsize=(18,5))

for i in range(n):
    ax=plt.subplot(2,n,i+1)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax2=plt.subplot(3,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()

plt.show()
"""

######################## OUTPUT ########################
"""
Epoch 1/10
235/235 [==============================] - 3s 9ms/step - loss: 0.2737 - val_loss: 0.1829
Epoch 2/10
235/235 [==============================] - 2s 9ms/step - loss: 0.1659 - val_loss: 0.1508
Epoch 3/10
235/235 [==============================] - 2s 8ms/step - loss: 0.1425 - val_loss: 0.1329
Epoch 4/10
235/235 [==============================] - 2s 8ms/step - loss: 0.1280 - val_loss: 0.1212
Epoch 5/10
235/235 [==============================] - 2s 7ms/step - loss: 0.1181 - val_loss: 0.1128
Epoch 6/10
235/235 [==============================] - 2s 6ms/step - loss: 0.1111 - val_loss: 0.1069
Epoch 7/10
235/235 [==============================] - 2s 6ms/step - loss: 0.1060 - val_loss: 0.1027
Epoch 8/10
235/235 [==============================] - 1s 6ms/step - loss: 0.1022 - val_loss: 0.0994
Epoch 9/10
235/235 [==============================] - 2s 7ms/step - loss: 0.0995 - val_loss: 0.0971
Epoch 10/10
235/235 [==============================] - 2s 8ms/step - loss: 0.0976 - val_loss: 0.0956
313/313 [==============================] - 1s 1ms/step
313/313 [==============================] - 0s 1ms/step
"""
