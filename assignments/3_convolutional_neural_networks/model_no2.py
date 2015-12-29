###############################################################################
# @author           Alan Ramponi (179850)
# @course           Machine Learning
# @date             December 24th, 2015
# @description      Performance comparison of the deep convolutional network
#                   when removing layers (one at the time, keeping all the
#                   others) in classifying the MNIST dataset. This script
#                   regards the removing of the second layer.
#
# USAGE: python model_no2.py
###############################################################################

###############################################################################
# BASH COMMAND to activate the environment with TensorFlow installed inside it:
# =========================================================================== #
# source ~/tensorflow/bin/activate                                            #
###############################################################################


import tensorflow as tf
import input_data, plotter


def weight_variable(shape):
    """A function that initializes weigths with a small amount of noise for symmetry breaking and to prevent null gradients, that are usually yielded when the activation of a layer is made of ReLUs. An epsilon value defined around 0 makes the ReLU differentiable.

    Args:
        shape: the shape of the input vector

    Return:
        The vector of weigths initialized with a small amount of noise
    """
    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)

def bias_variable(shape):
    """A function that initializes biases with slightly positive values in order to avoid dead neurons that imply no learning.

    Args:
        shape: the shape of the input vector

    Return:
        The vector of biases initialized with a slightly positive values
    """
    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)

def conv2d(x, W):
    """A function that performs the convolution operation using a sliding window stride (for each dimension of the input) of 1. The convolution operation is setted to be 0-padded (i.e. the input is padded with 0s) so that the output has the same size as the input.

    Args:
        x: an input tensor that must be float32 or float64
        W: a filter tensor that must have the same type as the input tensor x

    Return:
        A tensor with the same type as the input tensor x
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    """A function that performs the max pooling operation over 2x2 blocks using a window (for each dimension of the input) of size [1,2,2,1] and a sliding window stride (for each dimension of the input) of [1,2,2,1]. The max pooling operation uses a padding algorithm that shrinks the dimension according to 'ksize' and 'strides'.

    Args:
        x: an input tensor with shape [batch, height, width, channels] and type float32, float64, qint8, quint8 or qint32

    Return:
        A max pooled tensor with the same type as the input tensor x
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Download MNIST data (55k train, 10k test, 5k validation into a lightw. class)
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


###############################################################################
# 1) GRAPH OF INTERACTING OPERATIONS THAT RUN ENTIRELY OUTSIDE PYTHON #########
###############################################################################

# Definition of placeholders, i.e. inputs of the computation
x = tf.placeholder("float", shape=[None, 784])  # images into 784-dim vectors
y = tf.placeholder("float", shape=[None, 10])   # images into 10 classes

# Reshaping of 784-dim vectors into 28x28 squares (for convol. and max pooling)
x_image = tf.reshape(x, [-1,28,28,1])


# FIRST CONVOLUTIONAL LAYER WITH MAX POOLING ##################################

# Definition of the weigth tensor: 5x5 patch, 1 input ch. and 32 output ch.
W_conv1 = weight_variable([5,5,1,32])

# Definition of the bias vector: 32 components, one for each output channel
b_conv1 = bias_variable([32])

# Definition of the convolution operation using the ReLU function too
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Definition of the max pooling operation on the 1st convolutional layer
h_pool1 = max_pool_2x2(h_conv1)


# SECOND CONVOLUTIONAL LAYER WITH MAX POOLING #################################
# THIS LAYER WAS REMOVED! #####################################################

# Definition of the weigth tensor: 5x5 patch, 32 input ch. and 64 output ch.
# W_conv2 = weight_variable([5,5,32,64])

# Definition of the bias vector: 64 components, one for each output channel
# b_conv2 = bias_variable([64])

# Definition of the convolution operation using the ReLU function too
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Definition of the max pooling operation on the 2nd convolutional layer
# h_pool2 = max_pool_2x2(h_conv2)


# THIRD FULLY-CONNECTED LAYER WITH DROPOUT ####################################

# Definition of the weigth tensor: 7*7 is the image size, 1024 are the neurons
W_fc1 = weight_variable([14*14*32,1024])

# Definition of the bias vector: 1024 components, one for each output channel
b_fc1 = bias_variable([1024])

# Reshaping of the tensor from the pooling layer into a batch of vectors
h_pool2_flat = tf.reshape(h_pool1, [-1,14*14*32])

# Reshaping of the whole input as a flat vector
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Definition of a placeholder for the prob. a neuron's output is kept during DO
keep_prob = tf.placeholder("float")

# Definition of the dropout operation that handles masking of neurons' outputs
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# FOURTH READOUT SOFTMAX LAYER ################################################

# Definition of the weigth tensor: the final weigths of pixel intensities
W_fc2 = weight_variable([1024,10])

# Definition of the bias vector: 10 components, the final extra evidences
b_fc2 = bias_variable([10])

# Definition of the softmax regression model, i.e. the output layer
y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # evidence to prob.


# OPTIMIZATION WITH ADAPTIVE GRADIENT #########################################

# Definition of the cross-entropy cost function
cross_entropy = -tf.reduce_sum(y*tf.log(y_hat))

# Definition of the training algorithm (AG optimizer with learning rate 1e-4)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


###############################################################################
# 2) LAUNCH THE GRAPH OF INTERACTING OPERATIONS IN A SESSION ##################
###############################################################################

# Start a new interactive session
sess = tf.InteractiveSession()

# Run the operation that initializes the variables
sess.run(tf.initialize_all_variables())

# Check if the prediction matches the truth, i.e. build a list of booleans
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))

# Determine what fraction is correct by taking the mean of the previous list
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train the model (by running the training step 20000 times)
for n in range(20000):
    # Get a batch of 50 random data points from the training set
    batch = mnist.train.next_batch(50)

    # Print the training accuracy every 100 steps
    if (n%100)==0:
        train_accuracy = accuracy.eval(
            feed_dict = {x:batch[0], y:batch[1], keep_prob:1.0}
        )
        print "step %d, TRAINING ACCURACY: %g" % (n, train_accuracy)

    # Run train_step feeding in the batches data to replace the placeholders
    sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})

# Print the accuracy of the prediction on test data
print "Evaluation of the deep convolutional network without the 2nd layer"
print "=================================================================="
print "Resulting TEST ACCURACY: %g" % accuracy.eval(
    feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0}
)
