
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



KERNEL_NUM = config.contextual_kernel_num
WEIGHT_DECAY = 0.00001
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

    
    images = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
    
    logits = network(images)
    
    return logits


def network(_input):
    output = multi_scale(_input)
    output = add_res_blocks(output)
    output = last_three_conv(output)
    output = softmax(output)
    print (output)
    return output
    
def multi_scale(_input):
    _input = tf.pad(_input, paddings=[[0,0],[2,2],[2,2],[0,0]])
    with tf.variable_scope('multi_conv1'):
        output1 = conv2d(_input, KERNEL_NUM, 5, weight_stddev=0.1, biases_init=1.0)
    with tf.variable_scope('multi_conv2'):
        output2 = conv2d(_input, KERNEL_NUM, 3, weight_stddev=0.1, biases_init=1.0)
        output2 = max_pooling(output2, ksize=[1, 3, 3, 1])
    with tf.variable_scope('multi_conv3'):
        output3 = conv2d(_input, KERNEL_NUM, 1, weight_stddev=0.1, biases_init=1.0)
        output3 = max_pooling(output3, ksize=[1, 5, 5, 1])
    output = tf.concat(axis=3, values=(output1, output2, output3))
    output = tf.nn.relu(output)
    output = local_response_norm(output)
    print (output)
    return output
    
    
def add_res_blocks(_input):
    with tf.variable_scope('res_conv1') as scope:
        output = conv2d(_input, KERNEL_NUM, 1, weight_stddev=0.1, biases_init=1.0)
        output = tf.nn.relu(output)
        output = local_response_norm(output)
    
    with tf.variable_scope('res_block1') as scope:
        output = add_one_block(output)
    with tf.variable_scope('res_block2') as scope:
        output = add_one_block(output)
    return output
    
    
def add_one_block(_input):
    """Add one residual block.
       There are two layers in one block.
     """
    # layer 1
    with tf.variable_scope('conv_1') as scope:
        output = conv2d(_input, KERNEL_NUM, 1, weight_stddev=0.05, biases_init=1.0)
        output = tf.nn.relu(output)
    print (output)
    
    # layer 2
    with tf.variable_scope('conv_2') as scope:
        output = conv2d(output, KERNEL_NUM, 1, weight_stddev=0.05, biases_init=1.0)
        output = output + _input
        output = tf.nn.relu(output)
    print (output)
    return output
    
def last_three_conv(_input):
    with tf.variable_scope('conv_1') as scope:
        output = conv2d(_input, KERNEL_NUM, 1, weight_stddev=0.05, biases_init=1.0)
        output = tf.nn.relu(output)
        output = dropout(output)

    with tf.variable_scope('conv_2') as scope:
        output = conv2d(output, KERNEL_NUM, 1, weight_stddev=0.05, biases_init=1.0)
        output = tf.nn.relu(output)
        output = dropout(output)

    with tf.variable_scope('conv_3') as scope:
        output = conv2d(output, KERNEL_NUM, 1, weight_stddev=0.1, biases_init=0.0)
    print (output)
    return output


def softmax(_input):
    features_total = int(_input.get_shape()[1]) * int(_input.get_shape()[2]) * int(_input.get_shape()[3])
    output = tf.reshape(_input, [-1, features_total])
    with tf.variable_scope('softmax') as scope:
        weights = weight_variable_normal(
                    [features_total, NUM_CLASSES],
                    name='weights', stddev=0.01)
        weight_decay = tf.multiply(tf.nn.l2_loss(weights), WEIGHT_DECAY, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        output = tf.matmul(output, weights) + biases
    return output

def conv2d(_input, out_features, kernel_size,
        strides=[1, 1, 1, 1], padding='VALID', weight_stddev=0.01, biases_init=1.0):
    in_features = int(_input.get_shape()[-1])
    kernel = weight_variable_normal(
        [kernel_size, kernel_size, in_features, out_features],
        name='kernel', stddev=weight_stddev)
    
    weight_decay = tf.multiply(tf.nn.l2_loss(kernel), WEIGHT_DECAY, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    
    biases = tf.get_variable('biases', shape=[out_features], 
               initializer=tf.constant_initializer(biases_init))
    output = tf.nn.conv2d(_input, kernel, strides, padding) + biases
    return output


def weight_variable_normal(shape, name, stddev):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev))


def max_pooling(_input, ksize, strides=[1, 1, 1, 1], padding='VALID'):
    return tf.nn.max_pool(_input, ksize=ksize, strides=strides, 
                padding=padding, name='max_pool')
    
    
def local_response_norm(_input):
    return tf.nn.lrn(_input, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn')
    



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
    optimizer = tf.train.RMSPropOptimizer(learning_rate, 
                            decay=0.9, momentum=0.0, epsilon=1e-10, 
                            use_locking=False, name='RMSProp')
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
