import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.misc
import tensorflow as tf

RAW_IMAGE_SHAPE = (150, 250, 3) #RBG channel of a landscape oriented card
REDUCED_IMAGE_SHAPE = (30, 50, 3)
PIXEL_DEPTH = 255.0

def normalizeImage(img):
    return (img-PIXEL_DEPTH/2)/PIXEL_DEPTH

def reformat(img):
  return img.reshape((-1, np.prod(REDUCED_IMAGE_SHAPE))).astype(np.float32)

def getPredictionColor(sm):
    return ["RED", "PURPLE", "GREEN"][np.argmax(sm)]

weights = tf.Variable(tf.truncated_normal([np.prod(REDUCED_IMAGE_SHAPE), 3]))
biases = tf.Variable(tf.zeros([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./color-model")
  raw = ndimage.imread(sys.argv[1]).astype(float)
  reduced = scipy.misc.imresize(raw, REDUCED_IMAGE_SHAPE)
  img = normalizeImage(reduced)
  prediction = tf.nn.softmax(tf.matmul(reformat(img), weights) + biases).eval()
  print getPredictionColor(prediction);
