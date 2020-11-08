
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from matplotlib import image
print("")

## Seeding
#np.random.seed(42)
#tf.random.set_seed(42)

## Loading the MNIST dataset and then normalizing the images.
#dataset = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = dataset.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

gmas = image.imread("/home/roman/gma.png")
angs = image.imread("/home/roman/ang.png")
cnss = image.imread("/home/roman/cnsimply.png")
ughs = image.imread("/home/roman/ugh.png")
wins = image.imread("/home/roman/win.png")

grandma = np.load("/home/roman/Pictures/grandma_1.npy")
angry   = np.load("/home/roman/Pictures/angry_1.npy")
cns     = np.load("/home/roman/Pictures/cns_1.npy")
ugh     = np.load("/home/roman/Pictures/ugh_1.npy")
winter  = np.load("/home/roman/Pictures/winter_1.npy")

#x_train = grandma

#x_train = np.vstack( (grandma,angry,cns,ugh,winter) )
x_train = np.vstack( (grandma,cns,winter) )

#dx = ndimage.sobel(x_train[:,:,:], 1)
#dy = ndimage.sobel(x_train[:,:,:], 2)

#x_train = np.hypot(dx, dy)
#x_train /= np.max(x_train)

#print(np.max(x_train_2))

#exit()

#x_train_2 = grandma[:,0:32,0:32]

#x_1 = x_train[:,0:56,0:56]
#x_2 = x_train[:,4:60,0:56]
#x_3 = x_train[:,8:64,0:56]
#
#x_4 = x_train[:,0:56,4:60]
#x_5 = x_train[:,4:60,4:60]
#x_6 = x_train[:,8:64,4:60]
#
#x_7 = x_train[:,0:56,8:64]
#x_8 = x_train[:,4:60,8:64]
#x_9 = x_train[:,8:64,8:64]
#
#x_train_2 = np.vstack( (x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9) )

#random_indices = np.random.choice(x_train.shape[0], size=10, replace=False)


#x_test_2  = x_train[random_indices,:,:]
#x_train_2 = x_train[np.delete(np.arange(x_train.shape[0]), random_indices),:,:]

#print(x_train_2.shape)
#print(x_test_2.shape)

#exit()

H = 64
W = 64
C = 1

## Flattening the images.

gmas_gray = np.mean(gmas,axis=2)
#gmas_mag = np.hypot(ndimage.sobel(gmas_gray[:,:], 0), ndimage.sobel(gmas_gray[:,:], 1))
#gmas_mag /= np.max(gmas_mag)
gmas_tr = np.vstack( (np.tile(np.reshape( gmas_gray, (-1, H * W * C)), (grandma.shape[0]//2,1)),
                     np.tile(np.reshape( np.flip(gmas_gray,axis=1), (-1, H * W * C)), (grandma.shape[0]//2,1))) )

#angs_gray = np.mean(angs,axis=2)
#angs_mag = np.hypot(ndimage.sobel(angs_gray[:,:], 0), ndimage.sobel(angs_gray[:,:], 1))
#angs_mag /= np.max(angs_mag)
#angs_tr = np.vstack( (np.tile(np.reshape( angs_gray, (-1, H * W * C)), (angry.shape[0]//2,1)),
#                     np.tile(np.reshape( np.flip(angs_gray,axis=1), (-1, H * W * C)), (angry.shape[0]//2,1)) ))

cnss_gray = np.mean(cnss,axis=2)
#cnss_mag = np.hypot(ndimage.sobel(cnss_gray[:,:], 0), ndimage.sobel(cnss_gray[:,:], 1))
#cnss_mag /= np.max(cnss_mag)
cnss_tr = np.vstack( (np.tile(np.reshape( cnss_gray, (-1, H * W * C)), (cns.shape[0]//2,1)),
                     np.tile(np.reshape( np.flip(cnss_gray,axis=1), (-1, H * W * C)), (cns.shape[0]//2,1)) ))

ughs_gray = np.mean(ughs,axis=2)
#ughs_mag = np.hypot(ndimage.sobel(ughs_gray[:,:], 0), ndimage.sobel(ughs_gray[:,:], 1))
#ughs_mag /= np.max(ughs_mag)
ughs_tr = np.vstack( (np.tile(np.reshape( ughs_gray, (-1, H * W * C)), (ugh.shape[0]//2,1)),
                     np.tile(np.reshape( np.flip(ughs_gray,axis=1), (-1, H * W * C)), (ugh.shape[0]//2,1)) ))

wins_gray = np.mean(wins,axis=2)
#wins_mag = np.hypot(ndimage.sobel(wins_gray[:,:], 0), ndimage.sobel(wins_gray[:,:], 1))
#wins_mag /= np.max(wins_mag)
wins_tr = np.vstack( (np.tile(np.reshape( wins_gray, (-1, H * W * C)), (winter.shape[0]//2,1)),
                     np.tile(np.reshape( np.flip(wins_gray,axis=1), (-1, H * W * C)), (winter.shape[0]//2,1)) ))

#gma_te = np.tile(np.reshape( np.mean(gma,axis=2), (-1, H * W * C)), (x_test_2.shape[0],1))

#y_train = np.vstack( (gmas_tr,angs_tr,cnss_tr,ughs_tr,wins_tr) )
y_train = np.vstack( (gmas_tr,cnss_tr,wins_tr) )
x_train = np.reshape(x_train, (-1, H * W * C))
#x_test = np.reshape(x_test_2, (-1, H * W * C))
print(x_train.shape, y_train.shape)
#print(gma_tr.shape, gma_te.shape)

## Latent space
latent_dim = 256

emb_name = 'embedding'

## Building the autoencoder
inputs = Input(shape=(H*W*C,))
e1 = Dense(2048, activation="relu")(inputs)
e2 = Dense(1024, activation="relu")(e1)
h = Dense(latent_dim, activation="relu", name=emb_name)(e2)
d4 = Dense(1024, activation="relu")(h)
d5 = Dense(2048, activation="relu")(d4)
outputs = Dense(H*W*C, activation="sigmoid")(d5)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
autoencoder.summary()


## Training the autoencoder
autoencoder.fit(
    x_train,
    y_train,
    epochs=3000,
    batch_size=2048,
    shuffle=False#,
#    validation_data=(x_test, gma_te)
)

random_indices = np.random.choice(x_train.shape[0], size=10, replace=False)

xxx = x_train[random_indices,:]
test_pred_y = autoencoder.predict(xxx)

autoencoder.save("autoencoder_model")

#intermediate_layer_model = Model(inputs=autoencoder.input,
#                                 outputs=autoencoder.get_layer(emb_name).output)
#intermediate_output = intermediate_layer_model.predict(x_test)

#np.save("fle.npy",intermediate_output)

n = 10  ## how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ## display original
    ax = plt.subplot(2, n, i + 1)
    ax.set_title("Original Image")
    plt.imshow(xxx[i].reshape(H, W))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ## display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    ax.set_title("Predicted Image")
    plt.imshow(test_pred_y[i].reshape(H, W))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("results/deep_autoencoder.png")
