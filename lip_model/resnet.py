from __future__ import division

from util.tf_util import batch_normalization_wrapper
from tensorflow.contrib.keras import backend as K
import tensorflow as tf

kernel_initializer = tf.contrib.keras.initializers.he_normal()
kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-4)

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

def resnet_18(input_tensor, repetitions=(2, 2, 2, 2)):
  """Builds a custom ResNet like architecture.

  Args:
  input_shape: The input tensor with shape (nb_rows, nb_cols, nb_channels)
  num_outputs: The number of outputs at final softmax layer
  block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
  The original paper used basic_block for layers < 50
  repetitions: Number of repetitions of various block units.
  At each block unit, the number of filters are doubled and the input size is halved

  Returns:
  The keras `lip_model.Model`.
  """

  block = input_tensor
  filters = 64
  for i, r in enumerate(repetitions):
    block = _residual_block(filters=filters, repetitions=r,
                            is_first_layer=(i == 0))(block)
    filters *= 2

  # Last activation
  block = _bn_relu(block)

  # Classifier block
  block_shape = K.int_shape(block)
  pool2 = tf.layers.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                           strides=(1, 1))(block)

  flatten1 = tf.layers.Flatten()(pool2)

  return flatten1

def _residual_block(filters, repetitions, is_first_layer=False):
  """Builds a residual block with repeating bottleneck blocks.
  """
  def f(input):

    init_strides = (1, 1)
    if not is_first_layer:
      init_strides = (2, 2)
    input = _basic_block(filters=filters, init_strides=init_strides, is_first_block=True,
                         is_first_block_of_first_layer=(is_first_layer))(input)

    init_strides = (1, 1)
    for i in range(repetitions-1):
      input = _basic_block(filters=filters, init_strides=init_strides, is_first_block=False,
                           is_first_block_of_first_layer=False)(input)
    return input

  return f


def _basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False, is_first_block=False):
  """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
  Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
  """
  def f(input):

    if is_first_block and not is_first_block_of_first_layer:
      activation = _bn_relu(input)
      conv1 = _only_conv(filters=filters, kernel_size=(3, 3),
                            strides=init_strides)(activation)
      residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
      return _shortcut(activation, residual)

    if is_first_block_of_first_layer:
      # don't repeat bn->relu since we just did bn->relu->maxpool
      conv1 = tf.layers.conv2d(input, filters=filters, kernel_size=(3, 3), use_bias=False,
                     strides=init_strides,
                     padding="same",
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer )
    else:
      conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                            strides=init_strides)(input)

    residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
    return _shortcut(input, residual)

  return f

def _shortcut(input, residual):
  """Adds a shortcut between input and residual block and merges them with "sum"
  """
  # Expand channels of shortcut to match residual.
  # Stride appropriately to match residual (width, height)
  # Should be int if network architecture is correctly configured.
  input_shape = K.int_shape(input)
  residual_shape = K.int_shape(residual)
  stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
  stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
  equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

  shortcut = input
  # 1 X 1 conv if shape is different. Else identity.
  if stride_width > 1 or stride_height > 1 or not equal_channels:
    shortcut = tf.layers.conv2d(input, filters=residual_shape[CHANNEL_AXIS], use_bias=False,
                      kernel_size=(1, 1),
                      strides=(stride_width, stride_height),
                      padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)

  return tf.add(shortcut, residual)


def _conv_bn_relu(filters, kernel_size, strides=(1,1), padding='same', kernel_regularizer=kernel_regularizer):
  """Helper to build a conv -> BN -> relu block
  """
  def f(input):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, use_bias=False,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)
    return _bn_relu(conv)

  return f


def _bn_relu_conv(filters, kernel_size, strides=(1,1), padding='same', kernel_regularizer=kernel_regularizer):
  """Helper to build a BN -> relu -> conv block.
  This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
  """

  def f(input):
    activation = _bn_relu(input)
    return tf.layers.conv2d(activation, filters=filters, kernel_size=kernel_size, use_bias=False,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)

  return f


def _bn_relu(input):
  """Helper to build a BN -> relu block
  """
  norm = batch_normalization_wrapper(input, axis=CHANNEL_AXIS, scale=True)
  return tf.nn.relu(norm)

def _only_conv(filters, kernel_size, strides=(1,1), padding='same', kernel_regularizer=kernel_regularizer):
  """Helper to build a BN -> relu -> conv block.
  This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
  """

  def f(input):

    assert strides[0] == 2
    padded = tf.contrib.keras.layers.ZeroPadding2D(padding=(1, 1))(input)
    return tf.layers.conv2d(padded, filters=filters, kernel_size=kernel_size, use_bias=False,
                  strides=strides, padding='valid',
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer )

  return f

