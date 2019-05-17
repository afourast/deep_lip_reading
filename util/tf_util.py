import tensorflow as tf

# From https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def batch_normalization_wrapper(input_tensor, **kwargs):
  """
  Wrap batch normalization to switch between training and evaluation mode
  """
  training=False # This has been hardcoded for evaluation code - change for training
  return tf.layers.batch_normalization(input_tensor, training=training, **kwargs)