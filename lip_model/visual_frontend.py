import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras import layers

from config import load_args
from lip_model.preproc_and_aug import resize_no_crop, \
  resize_vids, random_crop_frames, \
  random_hor_flip_frames, normalize_mean_std, replicate_to_batch
from lip_model.resnet import resnet_18
from util.tf_util import shape_list, batch_normalization_wrapper

config = load_args()

class VisualFrontend:

  def __init__(self, input):

    self.input = model = input

    aug_opts = {}
    if config.test_aug_times:
      # ------------------- With test augmentation, keep first sample  the same --------
      assert model.shape[0] == 1, 'Test augmentation only with bs=1'
      no_aug_input = model
      model = replicate_to_batch(model, config.test_aug_times-1)

      aug_opts = { 'horizontal_flip': config.horizontal_flip ,
                   'crop_pixels': config.crop_pixels,
                   }

      no_aug_out = self.preprocess_and_augment(no_aug_input, aug_opts={})

    flip_prob = 0.5 if not config.test_aug_times == 2 else 1
    self.aug_out = model = self.preprocess_and_augment(model,
                                                       aug_opts = aug_opts,
                                                       flip_prob=flip_prob)

    if config.test_aug_times:
      self.aug_out = model = tf.concat( [ no_aug_out,  self.aug_out], 0 )

    # spatio-temporal frontend
    model = tf.contrib.keras.layers.ZeroPadding3D(padding=(2, 3, 3))(model)
    model = tf.layers.Conv3D(filters = 64,
                             kernel_size = (5, 7, 7),
                             strides = [1, 2, 2],
                             padding = 'valid',
                             use_bias = False)(model)

    model = batch_normalization_wrapper(model)
    model = tf.nn.relu(model)
    model = tf.contrib.keras.layers.ZeroPadding3D(padding=(0, 1, 1))(model)
    model = tf.layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1,2,2))(model)

    # We want to apply the resnet on every timestep, so reshape into a batch of size b*t
    packed_model = temporal_batch_pack(model, input_shape=K.int_shape(model)[1:])
    resnet = resnet_18(packed_model)
    self.output = temporal_batch_unpack(resnet,
                                        shape_list(model)[1],
                                        input_shape=K.int_shape(resnet)[1:])

  def preprocess_and_augment(self, input_tens, aug_opts, flip_prob=0.5):
    output = input_tens

    # convert to grayscale if RGB
    if not config.img_channels == 1:
      assert config.img_channels == 3, 'Input video channels should be either 3 or 1'
      output = tf.image.rgb_to_grayscale(output)

    if config.resize_input:
      new_h = new_w = config.resize_input
      output = self.aug_resize = resize_no_crop(output,new_h, new_w)

    img_width = output.shape.as_list()[2]
    crp = img_width - config.net_input_size
    if 'crop_pixels' in aug_opts and aug_opts['crop_pixels']:
      crp -= 2 * aug_opts['crop_pixels']
    crp //= 2

    crp_l = crp_r = crp_t = crp_b = crp

    if config.scale:
      output = self.aug_scale = resize_vids(output,scale=config.scale)

    output = layers.Cropping3D(cropping=((0, 0), (crp_t, crp_b), (crp_l, crp_r)))(output)

    if 'crop_pixels' in aug_opts and aug_opts['crop_pixels']:
      output = self.aug_crop = random_crop_frames(output, aug_opts['crop_pixels'])

    if 'horizontal_flip' in aug_opts and aug_opts['horizontal_flip']:
      output = self.aug_flip = random_hor_flip_frames(output, prob=flip_prob)

    if config.mean and config.std:
      output = normalize_mean_std(output, mean=config.mean, std=config.std)

    return output

def temporal_batch_pack(input, input_shape):
  # newshape = tf.concat( ([-1], input_shape[1:]), axis=0 )
  newshape = (-1,) + input_shape[1:]
  return tf.reshape(input, newshape )

def temporal_batch_unpack(input, time_dim_size, input_shape):
  # newshape = tf.concat( ([-1], [time_dim_size] , input_shape), axis=0 )
  newshape = (-1, time_dim_size)  + input_shape
  return tf.reshape(input, newshape)


