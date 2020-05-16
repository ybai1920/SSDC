
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import config


IMAGE_SIZE = config.patch_size
NUM_CLASSES = config.num_classes
CHANNELS = config.channels
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS

KERNEL_SIZE_CONV = 3    # kernel size
TOTAL_BLOCKS = 1    # number of dense blocks
LAYERS_PER_BLOCK = 3    # number of layers in one dense block
GROWTH_RATE = 4    # growth rate, also known as k
BATCH_NORM = True    # whether to use batch normalization
KEEP_PROB = 0.5     # factor of dropout


global IS_TRAINING


def inference(images, is_training):
    """Build the model up to where it may be used for inference.
    Args:
    images: Images placeholder.
    is_training: True or False.

    Returns:
    logits: Output tensor with the computed logits.
    """     
    
    global IS_TRAINING
    IS_TRAINING = is_training

    # Conv
    with tf.variable_scope('conv') as scope:
        # Flattening the 3D image into a 1D array
        images = tf.reshape(images, [-1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS])
        images = tf.transpose(images, perm=[0,3,1,2])
        x_image = tf.reshape(images, [-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, 1])
        output = conv3d(x_image, out_features=24, kernel_size=1, kernel_depth=7, 
                         strides=[1, 2, 1, 1, 1], padding='VALID')
    print (output)
        
    with tf.variable_scope('dense_1') as scope:
        output = add_block_3d(output,
                          growth_rate=12,
                          kernel_size=1,
                          kernel_depth=7,
                          layers_per_block=3)
    print(output)
    
    output = reduce_dimension_layer(output, inter_features=200, out_features=24)
    print(output)

    with tf.variable_scope('dense_2') as scope:
        output = add_block_3d(output,
                          growth_rate=12,
                          kernel_size=3,
                          kernel_depth=1,
                          layers_per_block=3)
    print(output)
    with tf.variable_scope('flatten') as scope:
        output = bn_prelu(output)
        output = avg_pool3d(output)
        print(output)

        features_total = int(output.get_shape()[1])*int(output.get_shape()[2])*int(output.get_shape()[3])*int(output.get_shape()[4])
        output = tf.reshape(output, [-1, features_total])
        print(output)
        output = dropout(output)
     
    with tf.variable_scope('fc'):
        weights = tf.get_variable('weights', shape=[features_total, NUM_CLASSES],
                         initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.add(tf.matmul(output, weights), biases, name='output')
    print(logits)
    return logits

    
def avg_pool3d(_input):
    kernel_size = int(_input.get_shape()[2])
    depth = int(_input.get_shape()[1])
    with tf.variable_scope('avg_pool')as scope:
        output = tf.nn.avg_pool3d(_input, ksize=[1, depth, kernel_size, kernel_size, 1], 
                               strides=[1, depth, kernel_size, kernel_size, 1], 
                               padding='VALID', name='avg_pool')
    return output



def reduce_dimension_layer(_input, inter_features, out_features):
    kernel_size = int(_input.get_shape()[2])
    depth = int(_input.get_shape()[1])
    
    with tf.variable_scope('reduce_dim_1') as scope:
        output = bn_prelu_conv3d(_input, inter_features, kernel_size=1,kernel_depth=depth, padding='VALID')
        output = tf.transpose(output, perm=[0,4,2,3,1])
    with tf.variable_scope('reduce_dim_2'):
        output = bn_prelu_conv3d(output, out_features, kernel_size=3, kernel_depth=inter_features, padding='VALID')
    return output
    

def add_block_3d(_input, growth_rate, kernel_size, kernel_depth, layers_per_block):
    output = _input
    for layer in range(layers_per_block):
        with tf.variable_scope('layer_%d' % layer):
            output = add_internal_layer_3d(output, growth_rate, kernel_size, kernel_depth)
    return output


def add_internal_layer_3d(_input, out_features, kernel_size, kernel_depth):
    output = _input
        
    output = bn_prelu_conv3d(output, out_features, kernel_size, kernel_depth)
    print (output)
    
    # concatenate
    output = tf.concat(axis=-1, values=(_input, output))
    return output    


    
    
def bn_prelu(_input):
    with tf.variable_scope('bn_PReLU') as scope:
        output = batch_normalization(_input)
        output = parametric_relu(output)
    return output


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.25),
                        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg        
        
        

def bn_prelu_conv3d(_input, out_features, kernel_size, kernel_depth, strides=[1,1,1,1,1], padding='SAME'):
    with tf.variable_scope('bn_PReLU_conv3d'):
        output = batch_normalization(_input)
        output = parametric_relu(output)
        output = conv3d(output, out_features, kernel_size, kernel_depth, strides, padding)
    return output
    



def batch_normalization(_input):
    if BATCH_NORM:
        output = tf.contrib.layers.batch_norm(_input, scale=True, updates_collections=None, is_training=IS_TRAINING)
    else:
        output = _input
    return output

def conv3d(_input, out_features, kernel_size, kernel_depth, 
          strides=[1, 1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = weight_variable_msra(
        [kernel_depth, kernel_size, kernel_size, in_features, out_features],
        name='kernel')
    output = tf.nn.conv3d(_input, kernel, strides, padding)
    return output



def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer())


def dropout(_input):
    if KEEP_PROB < 1:
        output = tf.cond(
            IS_TRAINING,
            lambda: tf.nn.dropout(_input, KEEP_PROB),
            lambda: _input
        )
    else:
        output = _input
    return output



# Define the loss function
def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


# Define the Training OP
def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Return the Predicting result
def predicting(logits):
    """Return the predicting result of logits.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    Returns:
    prediction: Prediction tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    """
    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(softmax, axis=1)
    return prediction
