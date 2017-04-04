import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.misc

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

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
            trainingData = data[0:trainingSize, :, :, :]
            trainingLabels = np.ndarray(trainingSize, dtype=np.int32)
            trainingLabels[:] = label

            validationSize = int(0.1*data.shape[0])
            validationData = data[trainingSize:trainingSize+validationSize, :, :, :]
            validationLabels = np.ndarray(validationSize, dtype=np.int32)
            validationLabels[:] = label

            testSize = data.shape[0]-trainingSize-validationSize
            testData = data[trainingSize+validationSize:, :, :, :]
            testLabels = np.ndarray(testSize, dtype=np.int32)
            testLabels[:] = label    

            rtnTrainingData = np.concatenate([rtnTrainingData, trainingData])
            rtnValidationData = np.concatenate([rtnValidationData, validationData])
            rtnTestData = np.concatenate([rtnTestData, testData])

            rtnTrainingLabels = np.concatenate([rtnTrainingLabels, trainingLabels])
            rtnValidationLabels = np.concatenate([rtnValidationLabels, validationLabels])
            rtnTestLabels = np.concatenate([rtnTestLabels, testLabels])
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

# Export model
export_path = "./model/1" 
print 'Exporting trained model to', export_path
builder = saved_model_builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
values, indices = tf.nn.top_k(y, 3)
prediction_classes = tf.contrib.lookup.index_to_string(
      tf.to_int64(indices), mapping=["RED", "PURPLE", "GREEN"])
classification_inputs = utils.build_tensor_info(serializedModel)
classification_outputs_classes = utils.build_tensor_info(prediction_classes)
classification_outputs_scores = utils.build_tensor_info(values)

classification_signature = signature_def_utils.build_signature_def(
    inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
    outputs={
        signature_constants.CLASSIFY_OUTPUT_CLASSES:
            classification_outputs_classes,
        signature_constants.CLASSIFY_OUTPUT_SCORES:
            classification_outputs_scores
      },
    method_name=signature_constants.CLASSIFY_METHOD_NAME)

tensor_info_x = utils.build_tensor_info(x)
tensor_info_y = utils.build_tensor_info(y)

prediction_signature = signature_def_utils.build_signature_def(
    inputs={'images': tensor_info_x},
    outputs={'scores': tensor_info_y},
    method_name=signature_constants.PREDICT_METHOD_NAME)

legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    session, [tag_constants.SERVING],
    signature_def_map={
        'predict_images':
            prediction_signature,
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            classification_signature,
    },
    legacy_init_op=legacy_init_op)

builder.save()

print 'Done exporting!'

"""
graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:, :])
  tf_train_labels = tf.constant(train_labels[:])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([np.prod(REDUCED_IMAGE_SHAPE), 3]))
  biases = tf.Variable(tf.zeros([3]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 500

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

print "training model..."
with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))

  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

# Export model
export_path_base = sys.argv[-1]
export_path = os.path.join(
    compat.as_bytes(export_path_base),
    compat.as_bytes(str(FLAGS.model_version)))
print 'Exporting trained model to', export_path
builder = saved_model_builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
classification_inputs = utils.build_tensor_info(serialized_tf_example)
classification_outputs_classes = utils.build_tensor_info(prediction_classes)
classification_outputs_scores = utils.build_tensor_info(values)
"""
