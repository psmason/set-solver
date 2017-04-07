import os
import random
import SocketServer

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.misc

import tensorflow as tf

from six.moves import cPickle as pickle

PIXEL_DEPTH = 255.0
TRAINING_DIR = "./training"
ORGANIZED_DIR = "./organized"
RAW_IMAGE_SHAPE = (150, 250, 3) #RBG channel of a landscape oriented card
REDUCED_IMAGE_SHAPE = (30, 50, 3)

files = os.listdir(TRAINING_DIR)
print "files found:", len(files)

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

def pickleColor(organized, color):
    files = organized[color]
    dataset = np.ndarray(shape=(len(files),) + REDUCED_IMAGE_SHAPE,
                         dtype=np.float32)
    for i in xrange(len(files)):
        f = os.path.join(TRAINING_DIR, files[i])
        raw = ndimage.imread(f).astype(float)
        reduced = scipy.misc.imresize(raw, REDUCED_IMAGE_SHAPE)
        dataset[i, :, :, :] = normalizeImage(reduced)

    print color, "mean:", np.mean(dataset)
    print color, "std:", np.std(dataset)
        
    print color, "dataset shape:", dataset.shape
    outputFile = os.path.join(ORGANIZED_DIR, color + ".pickle")
    with open(outputFile, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def splitDatasets():
    # 80/10/10 for training/validation/test sets
    rtnTrainingData   = np.ndarray(shape=(0,)+REDUCED_IMAGE_SHAPE, dtype=np.float32)
    rtnValidationData = np.ndarray(shape=(0,)+REDUCED_IMAGE_SHAPE, dtype=np.float32)
    rtnTestData       = np.ndarray(shape=(0,)+REDUCED_IMAGE_SHAPE, dtype=np.float32)

    rtnTrainingLabels   = np.ndarray(shape=(0), dtype=np.int32)
    rtnValidationLabels = np.ndarray(shape=(0), dtype=np.int32)
    rtnTestLabels       = np.ndarray(shape=(0), dtype=np.int32)
    
    for _, (color, label) in enumerate([("RED", 0), ("PURPLE", 1), ("GREEN", 2)]):
        with open(os.path.join(ORGANIZED_DIR, color + ".pickle"), 'rb') as f:
            data = pickle.load(f)
            np.random.shuffle(data)

            trainingSize = int(0.8*data.shape[0])
            rtnTrainingData = np.concatenate([rtnTrainingData,
                                              data[0:trainingSize, :, :, :]])
            rtnTrainingLabels = np.concatenate([rtnTrainingLabels, np.full(trainingSize,
                                                                           label,
                                                                           dtype=np.int32)])
            
            validationSize = int(0.1*data.shape[0])
            rtnValidationData = np.concatenate([rtnValidationData,
                                                data[trainingSize:trainingSize+validationSize, :, :, :]])
            rtnValidationLabels = np.concatenate([rtnValidationLabels,
                                                  np.full(validationSize,
                                                          label,
                                                          dtype=np.int32)])

            testSize = data.shape[0]-trainingSize-validationSize
            rtnTestData = np.concatenate([rtnTestData,
                                          data[trainingSize+validationSize:, :, :, :]])
            rtnTestLabels = np.concatenate([rtnTestLabels,
                                            np.full(testSize,
                                                    label,
                                                    dtype=np.int32)])

    return rtnTrainingData, rtnValidationData, rtnTestData, rtnTrainingLabels, rtnValidationLabels, rtnTestLabels

def nextTrainingBatch(data, labels, batchSize):
    indices = random.sample(range(data.shape[0]), batchSize)
    return {
        "data" : data[indices, :],
        "labels" : labels[indices],
    }

organized = organizeByColor(files)
print "RED count:", len(organized["RED"])
print "PURPLE count:", len(organized["PURPLE"])
print "GREEN count:", len(organized["GREEN"])

pickleColor(organized, "RED")
pickleColor(organized, "PURPLE")
pickleColor(organized, "GREEN")

trainingData, validationData, testData, trainingLabels, validationLabels, testLabels = splitDatasets()
print trainingData.shape, validationData.shape, testData.shape
print trainingLabels.shape, validationLabels.shape, testLabels.shape

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, np.prod(REDUCED_IMAGE_SHAPE))).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(3) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(trainingData, trainingLabels)
valid_dataset, valid_labels = reformat(validationData, validationLabels)
test_dataset, test_labels   = reformat(testData, testLabels)

# Train model
session = tf.InteractiveSession()
serializedModel = tf.placeholder(tf.string, name="set_color_model")
featureConfigs = {"x" : tf.FixedLenFeature(shape=[np.prod(REDUCED_IMAGE_SHAPE)],
                                           dtype=tf.float32)}
parsedModel = tf.parse_example(serializedModel, featureConfigs)
x  = tf.identity(parsedModel['x'], name='x')
y_ = tf.placeholder('float', shape=[None, 3])
w  = tf.Variable(tf.truncated_normal([np.prod(REDUCED_IMAGE_SHAPE), 3]))
b  = tf.Variable(tf.zeros([3]))

session.run(tf.global_variables_initializer())
y = tf.identity(tf.matmul(x, w) + b, name='y')
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
)

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
for i in range(500):
    batch = nextTrainingBatch(train_dataset, train_labels, 200)
    optimizer.run(feed_dict={x: batch["data"], y_: batch["labels"]})
    if 0 == i%100:
        print 'checkpoint accuracy %g' % session.run(
            accuracy, feed_dict={x: valid_dataset,
                                 y_: valid_labels})                  

print 'training accuracy %g' % session.run(
    accuracy, feed_dict={x: test_dataset,
                         y_: test_labels})

import cv2
class MyTCPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        # receive 1MB        
        data = self.request.recv(1024*1024)
        raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        normalized = normalizeImage(scipy.misc.imresize(raw, REDUCED_IMAGE_SHAPE))
        normalized = normalized.reshape((-1, np.prod(REDUCED_IMAGE_SHAPE))).astype(np.float32)
        prediction =  session.run(
            tf.argmax(y, 1), feed_dict={x: normalized}
        )
        self.request.sendall(["RED", "PURPLE","GREEN"][prediction])

if __name__ == "__main__":
    server = SocketServer.TCPServer(("localhost", 9000), MyTCPHandler)

    print "starting model service"
    server.serve_forever()
