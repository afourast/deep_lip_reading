from util.tf_util import shape_list
from tensorflow.python.ops import array_ops, random_ops, math_ops, control_flow_ops
import tensorflow as tf

def wrap_in_training_phase(train_out, test_out, name=None):
  # This has been hardcoded for evaluation code where we only use the function for test
  # time augmentation. For train-time augmentation, change this accordingly
  training_phase = True

  if training_phase:
    return train_out
  else:
    return test_out

# --- resize ---

def resize_no_crop(vids, new_h, new_w, name=None):
  resized = tf.map_fn( lambda x: tf.image.resize_images(x, [new_h, new_w]), vids, name=name)
  return resized

def resize_vids(vids, scale, name=None):
  h,w = vids.shape.as_list()[2:4]
  h_n, w_n = int(scale*h), int(scale*w)
  resized = tf.map_fn( lambda x: tf.image.resize_images(x, [w_n, h_n]), vids )
  resized_cropped = tf.map_fn( lambda x: tf.image.resize_image_with_crop_or_pad(x, w, h), resized , name=name)
  return resized_cropped


# --- mean, std normalization ---

def normalize_mean_std(frames, mean, std, name=None):
  return tf.identity((frames - mean)/std, name=name)

def replicate_to_batch(frame, times):
  replicated = tf.tile( frame, [times*shape_list(frame)[0]] + [1]* (len(frame.shape)-1) )
  return replicated

# --- random horizontal flip ---

def random_reverse_vid(frames, prob = 0.5, seed = None, name=None):
  flip_fun = lambda image: array_ops.reverse(image, [2])

  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror_cond = math_ops.less(uniform_random, prob)
  result = control_flow_ops.cond(mirror_cond,
                                 lambda: flip_fun(frames),
                                 lambda: frames)
  return result

def random_reverse_vid_with_prob(prob, seed = None, name=None):
  return lambda frames: random_reverse_vid(frames, prob, seed, name)

def random_hor_flip_frames(frames, prob=0.5, seed=None, name=None):
  # vid = tf.image.random_flip_left_right(vid)
  return wrap_in_training_phase(tf.map_fn(random_reverse_vid_with_prob(prob), frames) , frames )


# --- random crop ---

def random_crop_frames(vid, crop_pixels):
  train_t = _random_crop_frames_train(vid, crop_pixels)
  infer_t = _random_crop_frames_infer(vid, crop_pixels)
  return wrap_in_training_phase( train_t , infer_t )

def _random_crop_frames_train(vid_batches, crop_pixels, name=None):
  new_shape = vid_batches.shape.as_list()
  new_shape[2] = new_shape[2] - 2*crop_pixels
  new_shape[3] = new_shape[3] - 2*crop_pixels
  crop_lambda = lambda vid: tf.random_crop(vid, new_shape[1:])
  return tf.map_fn(crop_lambda, vid_batches, name=name)

def _random_crop_frames_infer(vid_batches, crop_pixels, name=None):
  orig_shape = vid_batches.shape.as_list()
  assert  orig_shape[2] == orig_shape[3]
  pixels_orig = orig_shape[2]
  new_len = pixels_orig - 2*crop_pixels
  mid_idx = (pixels_orig - new_len)/2
  return tf.identity(vid_batches[:,:,mid_idx:mid_idx+new_len,mid_idx:mid_idx+new_len], name=name)
