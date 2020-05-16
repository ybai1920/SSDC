# Replace the 2D dual channel in SSDC-DenseNet with 3D modules

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


BATCH_NORM = True    # whether to use batch normalization
KEEP_PROB = 0.5     # factor of dropout
KERNEL_NUM = config.ssdc_kernel_num
LAYER_NUM = 3


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
    

    
    with tf.variable_scope('Block1') as scope:
        output = tf.reshape(images, [-1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS])
        output = add_spectral_feature_extraction_block(output)

    with tf.variable_scope('Block2') as scope:
        output = tf.reshape(output, [-1,IMAGE_SIZE,IMAGE_SIZE, 128, 1])
        output = tf.transpose(output, perm=[0,3,1,2,4])
        output = add_spectral_spatial_features_learning_block_3d(output, KERNEL_NUM, LAYER_NUM)
        
    with tf.variable_scope('Block3') as scope:
        output = add_feature_fusion_block(output)
    
    return output


def add_spectral_feature_extraction_block(_input):
    output = conv2d(_input, kernel_size = 1, out_features = 128)
    output = relu(output)
    return output


def add_spectral_spatial_features_learning_block(_input, kernel_num, layer_num):
    with tf.variable_scope('spectral') as scope:
        spec = add_dense_block(_input, kernel_size = 1, kernel_num = kernel_num, layer_num = layer_num)
        spec = avg_pool(spec)

    with tf.variable_scope('spatial') as scope:
        spa = add_dense_block(_input, kernel_size = 3, kernel_num = kernel_num, layer_num = layer_num)
        spa = avg_pool(spa)
    
    output = tf.concat(axis=3, values=(spec, spa), name='concat')
    print(output)
    return output
    

def add_spectral_spatial_features_learning_block_3d(_input, kernel_num, layer_num):
    with tf.variable_scope('spectral_3d') as scope:
        spec = add_dense_block_3d(_input, kernel_depth = 7, kernel_size = 1, kernel_num = kernel_num, layer_num = layer_num)
        spec = avg_pool_3d(spec)
        spec = batch_norm(spec)
        spec = relu(spec)
        spec = conv3d(spec, kernel_depth=128, kernel_size=1, out_features=224, padding='VALID')
        spec = tf.reshape(spec, [-1, 3, 3, 224])
        
    with tf.variable_scope('spatial_3d') as scope:
        spa = add_dense_block_3d(_input, kernel_depth = 1, kernel_size = 3, kernel_num = kernel_num, layer_num = layer_num)
        spa = avg_pool_3d(spa)
        spa = batch_norm(spa)
        spa = relu(spa)
        spa = conv3d(spa, kernel_depth=128, kernel_size=1, out_features=224, padding='VALID')
        spa = tf.reshape(spa, [-1, 3, 3, 224])
    
    output = tf.concat(axis=3, values=(spec, spa), name='concat')
#     output = spec
    print(output)
    return output
    
    
    
def add_feature_fusion_block(_input):
    with tf.variable_scope('reduce_dim') as scope:
        output = batch_norm(_input)
        output = relu(output)
        output = conv2d(output, kernel_size = 1, out_features = 64)
        
    with tf.variable_scope('fusion') as scope:
        output = add_dense_block(output, kernel_size = 1, kernel_num = 32, layer_num = 3)
        output = global_avg_pool(output)
    
    with tf.variable_scope('fc') as scope:
        output = batch_norm(output)
        output = relu(output)
        output = fc(output)
    return output
    
    
def add_dense_block(_input, kernel_size, kernel_num, layer_num):
    output = _input
    with tf.variable_scope('dense_block') as scope:
        for layer in range(layer_num):
            with tf.variable_scope('dense_layer_%d' % layer) as scope:
                output = add_internal_layer(output, kernel_size, kernel_num)
    
    return output
    

def add_internal_layer(_input, kernel_size, kernel_num):
    output = batch_norm(_input)
    output = relu(output)
    output = conv2d(output, kernel_size, kernel_num)
    output = dropout(output)
    output = tf.concat(axis=3, values=(_input, output), name='concat')
    print(output)
    return output
    

    
def add_dense_block_3d(_input, kernel_depth, kernel_size, kernel_num, layer_num):
    output = _input
    with tf.variable_scope('dense_block_3d') as scope:
        for layer in range(layer_num):
            with tf.variable_scope('dense_layer_3d_%d' % layer) as scope:
                output = add_internal_layer_3d(output, kernel_depth, kernel_size, kernel_num)
    
    return output
    

def add_internal_layer_3d(_input, kernel_depth, kernel_size, kernel_num):
    output = batch_norm(_input)
    output = relu(output)
    output = conv3d(output, kernel_depth, kernel_size, kernel_num)
    output = dropout(output)
    output = tf.concat(axis=4, values=(_input, output), name='concat')
    print(output)
    return output
    
    
    
def batch_norm(_input):
    if BATCH_NORM:
        output = tf.contrib.layers.batch_norm(_input, scale=True, updates_collections=None, is_training=IS_TRAINING)
    else:
        output = _input
    print(output)
    return output 
    
    
def relu(_input):
    output = tf.nn.relu(_input)
    print(output)
    return output
    
    
def dropout(_input):
    if KEEP_PROB < 1:
        output = tf.cond(
            IS_TRAINING,
            lambda: tf.nn.dropout(_input, KEEP_PROB),
            lambda: _input,
            name='dropout'
        )
    else:
        output = _input
    print(output)
    return output

    
def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer())
        
        
def conv2d(_input, kernel_size, out_features, strides=[1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = weight_variable_msra(
        [kernel_size, kernel_size, in_features, out_features],
        name='kernel')
    output = tf.nn.conv2d(_input, kernel, strides, padding)
    print(output)
    return output
        
        
def conv3d(_input, kernel_depth, kernel_size, out_features,  
          strides=[1, 1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    kernel = weight_variable_msra(
        [kernel_depth, kernel_size, kernel_size, in_features, out_features],
        name='kernel')
    output = tf.nn.conv3d(_input, kernel, strides, padding)
    print(output)
    return output
    
    
def avg_pool(_input):
    # Avg pool
    output = tf.nn.avg_pool(_input, ksize=[1, 3, 3, 1], 
                         strides=[1, 2, 2, 1], 
                         padding='VALID', name='avg_pool')
    print(output)
    return output

    
def avg_pool_3d(_input):
    # Avg pool
    output = tf.nn.avg_pool3d(_input, ksize=[1, 1, 3, 3, 1], 
                         strides=[1, 1, 2, 2, 1], 
                         padding='VALID', name='avg_pool_3d')
    print(output)
    return output
    

def global_avg_pool(_input):
    size = int(_input.get_shape()[1])
    # global avg pool
    output = tf.nn.avg_pool(_input, ksize=[1, size, size, 1], 
                            strides=[1, size, size, 1], 
                            padding='VALID', name='global_avg_pool')
    print(output)
    return output
    
    
def fc(_input):
    with tf.variable_scope('fc'):
        features_total = int(_input.get_shape()[-1])
        output = tf.reshape(_input, [-1, features_total])
        weights = tf.get_variable('weights', shape=[features_total, NUM_CLASSES],
                         initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        output = tf.add(tf.matmul(output, weights), biases, name='output')
    print(output)
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
    optimizer = tf.train.AdamOptimizer(learning_rate)
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
