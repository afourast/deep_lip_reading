import tensorflow as tf

def cer(y_true, y_pred, return_all=False):
  labels_pred_sparse = one_hot_labels_to_sparse(y_pred)
  labels_true_sparse = one_hot_labels_to_sparse(y_true)
  ed = tf.edit_distance(tf.cast(labels_pred_sparse, tf.int32), labels_true_sparse)
  cer = tf.reduce_mean(ed)
  if return_all:
    return cer, ed
  else:
    return cer

def one_hot_labels_to_sparse(y_true):
  labels = tf.argmax(y_true,2)
  # sparse_labels = dense_to_sparse_padded(labels)
  sparse_labels = dense_to_sparse(labels)
  return tf.cast(sparse_labels,tf.int32)

def dense_to_sparse(dense_input):
  with tf.control_dependencies([tf.assert_integer(dense_input)]):
    idx = tf.where(tf.not_equal(dense_input, 0))
    vals = tf.gather_nd(dense_input, idx)
    shape = tf.cast(tf.shape(dense_input), tf.int64)
    sparse = tf.SparseTensor(idx, vals, shape)
    return sparse

