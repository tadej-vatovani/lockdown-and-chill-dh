import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from matplotlib import image


tst3 = image.imread("/home/roman/tst2.png") 
tst4 = image.imread("/home/roman/ugh.png") 
tst5 = image.imread("/home/roman/cnsimply.png") 
#tst6 = image.imread("/home/roman/gma.png") 

tst_stack = np.stack( [tst3, tst4, tst5], axis=0)
tst_gray = np.mean(tst_stack,axis=3)

#tst_dx = ndimage.sobel(tst_gray[:,:,:], 1)
#tst_dy = ndimage.sobel(tst_gray[:,:,:], 2)

#tst_mag = np.hypot(tst_dx, tst_dy)
#tst_mag /= np.max(tst_mag)


emb_name = 'embedding'

autoencoder = tf.keras.models.load_model("autoencoder_model")

intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(emb_name).output)


H = 64
W = 64
C = 1

grandma = np.load("/home/roman/Pictures/grandma_1.npy")
angry   = np.load("/home/roman/Pictures/angry_1.npy")
cns     = np.load("/home/roman/Pictures/cns_1.npy")
ugh     = np.load("/home/roman/Pictures/ugh_1.npy")
winter  = np.load("/home/roman/Pictures/winter_1.npy")

labels = ['grandma', 'cns', 'winter']
np_labels = np.hstack( (np.zeros(grandma.shape[0]), 
                        np.ones(cns.shape[0]), np.ones(winter.shape[0])*2))#, np.ones(winter.shape[0])*4) )

#np_labels = np.hstack( (np.zeros(grandma.shape[0]), np.ones(cns.shape[0]), np.ones(ugh.shape[0])*2) )

x_train = np.vstack( (grandma,cns,winter) )
print(x_train.shape)

#dx = ndimage.sobel(x_train[:,:,:], 1)
#dy = ndimage.sobel(x_train[:,:,:], 2)

#x_train = np.hypot(dx, dy)
#x_train /= np.max(x_train)

#print(np.max(x_train_2))

#exit()

#x_train_2 = grandma[:,0:32,0:32]

#random_indices = np.random.choice(x_train.shape[0], size=50, replace=False)
#x_test_2  = x_train[random_indices,:,:]

#train_random_indices = np.delete(np.arange(x_train.shape[0]), random_indices)

#x_train_2 = x_train[train_random_indices,:,:]
#np_labels_2 = np_labels[random_indices]

x_train_3 = np.reshape(x_train, (-1, H * W * C))
#x_test_3 = np.reshape(x_test, (-1, H * W * C))
tst_3 = np.reshape(tst_gray, (-1, H * W * C))

embedding = intermediate_layer_model.predict(x_train_3)

#test_embedding = intermediate_layer_model.predict(x_test_3)

tst_embedding = intermediate_layer_model.predict(tst_3)

nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(embedding)
#dist, ind = nbrs.kneighbors(test_embedding)
dist2, ind2 = nbrs.kneighbors(tst_embedding)

label_indices = np.median(np_labels[ind2[:,:]],axis=1)
for i in label_indices:
    print( labels[int(round(i))] )

#print(dist)
#print(ind)


print(tst_gray.shape)
print(tst_3.shape)

test_pred_y = autoencoder.predict(tst_3)

n = 3  ## how many digits we will display
plt.figure(figsize=(12, 4))
for i in range(n):
    ## display original
    ax = plt.subplot(2, n, i + 1)
    ax.set_title("Original Image")
    plt.imshow(tst_3[i].reshape(H, W))
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
plt.savefig("results/deep_predictor.png")
