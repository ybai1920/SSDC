
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
KERNEL_NUM = 32    # kernel numbers used in each conv layer
TOTAL_LAYERS = 4    # total number of layers used in the network 
               # TOTAL_LAYERS = ( 1 conv + 2 * blocks + 1 fc )
               # Be sure it is an even number


BATCH_NORM = True


def inference(images, is_training):
    """Build the model up to where it may be used for inference.
    Args:
    images: Images placeholder.
    is_training: True or False.

    Returns:
    logits: Output tensor with the computed logits.
    """  
    
    print ('CNN TOTAL LAYERS: %d' % TOTAL_LAYERS)
    
    # Conv 1
    with tf.variable_scope('conv') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE_CONV, KERNEL_SIZE_CONV, CHANNELS, KERNEL_NUM], 
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
        # Flattening the 3D image into a 1D array
        x_image = tf.reshape(images, [-1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS])
        z = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='SAME')
        h_conv = tf.nn.relu(z, name=scope.name)
        print (h_conv)


    # Block layers
    num_blocks = ( TOTAL_LAYERS - 2 ) // 2
    for block in range(num_blocks):
        h_conv = add_one_block(block, h_conv, is_training)
    
    
    features_total = int(h_conv.get_shape()[1]) * int(h_conv.get_shape()[2]) * int(h_conv.get_shape()[3])
    h_conv = tf.reshape(h_conv, [-1, features_total])

    with tf.variable_scope('fc'):
        weights = tf.get_variable('weights', shape=[features_total, NUM_CLASSES],
                         initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(h_conv, weights) + biases
    
    return logits




def add_one_block(block, h_conv, is_training):
    """Add one block.
       There are two layers in one block.
     """
    # layer 1
    with tf.variable_scope('block_%d_1' % block) as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE_CONV, KERNEL_SIZE_CONV, KERNEL_NUM, KERNEL_NUM], 
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
        biases = tf.get_variable('biases', shape=[KERNEL_NUM], initializer=tf.constant_initializer(0.05))

        if BATCH_NORM:
            h_conv_1 = tf.contrib.layers.batch_norm(h_conv, scale=True, updates_collections=None, is_training=is_training)
            h_conv_1 = tf.nn.conv2d(h_conv_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            h_conv_1 = h_conv_1 + biases
            h_conv_1 = tf.nn.relu(h_conv_1, name=scope.name)
        else:
            h_conv_1 = tf.nn.conv2d(h_conv, weights, strides=[1, 1, 1, 1], padding='SAME')
            h_conv_1 = h_conv_1 + biases
            h_conv_1 = tf.nn.relu(h_conv_1, name=scope.name)
        print (h_conv_1)
    
    # layer 2
    with tf.variable_scope('block_%d_2' % block) as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE_CONV, KERNEL_SIZE_CONV, KERNEL_NUM, KERNEL_NUM], 
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
        biases = tf.get_variable('biases', shape=[KERNEL_NUM], initializer=tf.constant_initializer(0.05))

        if BATCH_NORM:
            h_conv_2 = tf.contrib.layers.batch_norm(h_conv_1, scale=True, updates_collections=None, is_training=is_training)
            h_conv_2 = tf.nn.conv2d(h_conv_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            h_conv_2 = h_conv_2 + biases
            h_conv_2 = tf.nn.relu(h_conv_2, name=scope.name)
        else:
            h_conv_2 = tf.nn.conv2d(h_conv_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            h_conv_2 = h_conv_2 + biases
            h_conv_2 = tf.nn.relu(h_conv_2, name=scope.name)
    
    output = h_conv_2
    print (output)
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
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
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
