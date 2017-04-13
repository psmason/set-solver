import os
import random
import SocketServer

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.misc

import tensorflow as tf

from six.moves import cPickle as pickle

PIXEL_DEPTH         = 255.0
TRAINING_DIR        = "./training"
RAW_IMAGE_SHAPE     = (150, 250, 3) #RBG channel of a landscape oriented card
REDUCED_IMAGE_SHAPE = (30, 50, 3)
NUM_COLORS          = 3

def getFileColor(s):
    # assumes color is the first token of the file name
    return s.split("-")[0]

def organizeByColor(files):
    organized = {
        "RED" : [],
        "GREEN" : [],
        "PURPLE" : [],
    }

    for f in files:
        color = getFileColor(f)
        organized[color].append(f)
    return organized

def normalizeImage(img):
    return (img-PIXEL_DEPTH/2)/PIXEL_DEPTH

def loadAndNormalize(organizedFiles):
    classification = {
        "RED" : 0,
        "PURPLE" : 1,
        "GREEN" : 2,
    }

    features = np.ndarray(shape=(len(files),) + REDUCED_IMAGE_SHAPE,
                      dtype=np.float32)
    labels = np.ndarray(shape=(len(files),), dtype=np.int32)

    i = 0
    for color in organizedFiles.iterkeys():
        for f in organizedFiles[color]:
            f = os.path.join(TRAINING_DIR, f)
            raw = ndimage.imread(f).astype(float)
            reduced = scipy.misc.imresize(raw, REDUCED_IMAGE_SHAPE)
            features[i, :, :, :] = normalizeImage(reduced)
            labels[i] = classification[color]
            i += 1
    return features, labels

def splitData(features, labels):
    indices = np.random.permutation(features.shape[0])

    # 80/10/10 training/validation/test split
    trainingSize = int(len(indices)*0.8)
    validationSize = int(len(indices)*0.1)
    testSize = len(indices)-trainingSize-validationSize

    rtn = {
        "training" : {
            "features" : features[indices[:trainingSize], :, :, :],
            "labels"   : labels[indices[:trainingSize]],
        },
        "validation" : {
            "features" : features[indices[trainingSize:trainingSize+validationSize], :, :, :],
            "labels"   : labels[indices[trainingSize:trainingSize+validationSize]],
        },
        "test" : {
            "features" : features[indices[trainingSize+validationSize:], :, :, :],
            "labels"   : labels[indices[trainingSize+validationSize:]],
        },
    }

    return rtn

# Loading the features
files = os.listdir(TRAINING_DIR)
print "files found:", len(files)
organized = organizeByColor(files)
features, labels = loadAndNormalize(organized)
split = splitData(features, labels)

# Train model
def nextTrainingBatch(data, batchSize):
    indices = random.sample(range(data["features"].shape[0]), batchSize)
    return {
        "features" : data["features"][indices, :],
        "labels"   : data["labels"][indices],
    }

def formatFeatures(features):
    return features.reshape((-1, np.prod(REDUCED_IMAGE_SHAPE))).astype(np.float32)

def formatLabels(labels):
    return np.arange(NUM_COLORS) == labels[:,None].astype(np.float32)

session = tf.InteractiveSession()
serializedModel = tf.placeholder(tf.string, name="set_color_model")
featureConfigs = {"x" : tf.FixedLenFeature(shape=[np.prod(REDUCED_IMAGE_SHAPE)],
                                           dtype=tf.float32)}
parsedModel = tf.parse_example(serializedModel, featureConfigs)
x  = tf.identity(parsedModel['x'], name='x')
y_ = tf.placeholder('float', shape=[None, 3])
w  = tf.Variable(tf.truncated_normal([np.prod(REDUCED_IMAGE_SHAPE), 3]))
b  = tf.Variable(tf.zeros([3]))

y = tf.identity(tf.matmul(x, w) + b, name='y')
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    + 0.01*0.5*(tf.nn.l2_loss(w))
) 

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

session.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
for i in range(5000):
    batch = nextTrainingBatch(split["training"], 400)
    optimizer.run(feed_dict={
        x  : formatFeatures(batch["features"]),
        y_ : formatLabels(batch["labels"]),
    })
    if 0 == i%100:
        print 'checkpoint training accuracy %g' % session.run(
            accuracy, feed_dict={
                x  : formatFeatures(split["training"]["features"]),
                y_ : formatLabels(split["training"]["labels"]),
            })                  
        print 'checkpoint validation accuracy %g' % session.run(
            accuracy, feed_dict={
                x  : formatFeatures(split["validation"]["features"]),
                y_ : formatLabels(split["validation"]["labels"]),
            })                  

print 'training accuracy %g' % session.run(
    accuracy, feed_dict={
        x  : formatFeatures(split["test"]["features"]),
        y_ : formatLabels(split["test"]["labels"]),
    })

# Start the classification server
import cv2
class MyTCPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        while True:
            data = self.request.recv(1024*1024)
            if not data:
                break
            array = np.frombuffer(data, dtype=np.uint8)
            raw = cv2.imdecode(array, cv2.IMREAD_COLOR)
            normalized = normalizeImage(scipy.misc.imresize(raw, REDUCED_IMAGE_SHAPE))
            normalized = normalized.reshape((-1, np.prod(REDUCED_IMAGE_SHAPE))).astype(np.float32)
            prediction =  session.run(
                tf.argmax(y, 1), feed_dict={x: normalized}
            )
            response = ["RED", "PURPLE","GREEN"][prediction]
            self.request.sendall(response)

server = SocketServer.TCPServer(("localhost", 9000), MyTCPHandler)
print "starting model service"
server.serve_forever()
