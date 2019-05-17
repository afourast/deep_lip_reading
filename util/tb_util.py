import os
import tempfile

import moviepy.editor as mpy
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_op_util

# Code based on https://github.com/tensorflow/tensorboard/issues/39
def add_gif_summary(name,
                    im_thwc,
                    fps=25,
                    collections=None,
                    family=None,
                    timeline=False,
                    attention=[],
                    pred=[]):
  """
  IM_THWC: 4D tensor (TxHxWxC) for which GIF is to be generated.
  COLLECTION: collections to which the summary op is to be added.
  """
  # if summary_op_util.skip_summary(): return constant_op.constant('')
  with summary_op_util.summary_scope(name, family, values=[im_thwc]) as (tag, scope):
    pyfunc_args = [im_thwc, tag, fps, timeline, attention, pred]
    gif_summ = tf.py_func(py_encode_gif, pyfunc_args, tf.string, stateful=False)
    summary_op_util.collect(gif_summ, collections, [tf.GraphKeys.SUMMARIES])
  return gif_summ

def py_encode_gif(im_thwc, tag, fps=4, timeline=False, attention=[], preds=[]):
  """
  Given a 4D numpy tensor of images, encodes as a gif.
  """
  if not im_thwc.dtype == np.uint8:
    im_thwc = im_thwc - im_thwc.min()
    im_thwc = im_thwc / im_thwc.max()
    im_thwc = (im_thwc*255).astype(np.uint8)

  # maybe convert grayscale --> RGB
  if im_thwc.shape[-1] == 1:
    import cv2
    im_thwc = np.array([cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
                        for gray_img in im_thwc])

  # maybe add subtitles
  if len(attention) > 0 and len(preds) > 0:
    subs = align_subs_from_attention_matrix(attention, preds)
    im_thwc = add_subs_to_vid_tensor_cv(im_thwc, subs, scale=0.8)

  if timeline:
    add_time_line(im_thwc, width = 4)

  with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
  clip = mpy.ImageSequenceClip(list(im_thwc), fps=fps)
  clip.write_gif(fname, verbose=False, progress_bar=False)
  with open(fname, 'rb') as f: enc_gif = f.read()
  os.remove(fname)
  # create a tensorflow image summary protobuf:
  thwc = im_thwc.shape
  im_summ = tf.Summary.Image()
  im_summ.height = thwc[1]
  im_summ.width = thwc[2]
  im_summ.colorspace = 3 # fix to 3 == RGB
  im_summ.encoded_image_string = enc_gif
  # create a summary obj:
  summ = tf.Summary()
  summ.value.add(tag=tag, image=im_summ)
  summ_str = summ.SerializeToString()

  return summ_str

def add_time_line(vid, width = 4):
  """
  Adds a white bar on top of video to indicate time
  """
  time_dim, h, w, c = vid.shape
  # import ipdb; ipdb.set_trace(context=20)
  pixels_per_t_step = w/float(time_dim)
  for t in xrange(time_dim):
    img = vid[t]
    fill_width = int(pixels_per_t_step*t)
    if fill_width > 0:
      img[:width,:fill_width] = np.expand_dims(np.expand_dims( [255,255,255], 0),0)

# ---------------------------- Subs ----------------------------------------------------

def add_subs_to_vid_tensor_cv(vid, subs, scale=1):
  """
  :param vid_tensor:  t x h x w x c
  """

  import cv2
  font                   = cv2.FONT_HERSHEY_SIMPLEX
  font_scale              = scale
  font_color              = (255,255,255)
  bg_color              = (0, 0, 0)
  lineType               = 1

  time_dim, h, w, c = vid.shape

  pos = (10, h-10)

  for (from_t, to_t), txt in subs:
    for t in xrange(from_t, to_t):
      txt = txt.encode('utf-8')
      img = vid[t]
      (text_width, text_height) = cv2.getTextSize(txt,
                                                  font,
                                                  fontScale=font_scale,
                                                  thickness=1)[0]
      posx = (w-text_width) // 2 # place subtitle in the middle
      posy = h-10
      box_coords = ( (posx, posy + 1), (posx + text_width-2, posy - text_height - 2))
      cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
      cv2.putText(img, txt, (posx, posy), font, font_scale, font_color, lineType)

  return vid

def align_subs_from_attention_matrix(attention, preds):

  def find_best_path(grid_orig):
    """
    Simple dynamic progarm to find the highest score path from upper left to bottom right
    """
    m, n = grid_orig.shape
    grid = grid_orig.copy()

    for i in xrange(m - 2, -1, -1):
      grid[i,n - 1] += grid[i+1, n - 1]
    for j in xrange(n - 2, -1, -1):
      grid[m - 1, j] += grid[m - 1, j+1]

    for i in xrange(m - 2, -1, -1):
      for j in xrange(n - 2, -1, -1):
        grid[i, j] += max(grid[i + 1, j], grid[i, j + 1])

    path = [(0,0)]
    i = j = 0
    while i < m and j < n:
      if i == m-1:
        j+=1
      elif j == n-1:
        i+=1
      elif grid[i+1][j] > grid[i][j+1]:
        i += 1
      else:
        j+=1

      if  i < m and j < n:
        path.append((i,j))

    return path

  preds = preds.strip()
  space_indices = np.array([i for i, x in enumerate(preds) if x == '-'])
  frs = [0] + (space_indices + 1).tolist()
  tos = space_indices.tolist() + [len(preds)]
  words = preds.split('-')
  best_path = find_best_path(attention)
  path_mask = np.zeros_like(attention, dtype=int)
  path_mask[zip(*best_path)] = 1
  subs = []
  for fr_dec, to_dec, txt in zip(frs, tos, words):
    tups = [t_enc for t_enc, t_dec in best_path if t_dec >= fr_dec and t_dec < to_dec]
    fr_enc = min(tups)
    to_enc = max(tups)

    subs.append(((fr_enc, to_enc), txt))
  return subs
# --------------------------------------------------------------------------------

# https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
def colorize_image(value, vmin=None, vmax=None, cmap='viridis'):
  """
  A utility function for TensorFlow that maps a grayscale image to a matplotlib
  colormap for use with TensorBoard image summaries.

  By default it will normalize the input value to the range 0..1 before mapping
  to a grayscale colormap.

  Arguments:
    - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
      [height, width, 1].
    - vmin: the minimum value of the range used for normalization.
      (Default: value minimum)
    - vmax: the maximum value of the range used for normalization.
      (Default: value maximum)
    - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
      (Default: 'gray')

  Example usage:

  ```
  output = tf.random_uniform(shape=[256, 256, 1])
  output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
  tf.summary.image('output', output_color)
  ```

  Returns a 3D tensor of shape [height, width, 3].
  """

  # normalize
  vmin = tf.reduce_min(value) if vmin is None else vmin
  vmax = tf.reduce_max(value) if vmax is None else vmax
  value = (value - vmin) / (vmax - vmin) # vmin..vmax

  # squeeze last dim if it exists
  value = tf.squeeze(value)

  # quantize
  indices = tf.to_int32(tf.round(value * 255))

  # gather
  import matplotlib.cm
  cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
  colors = tf.constant(cm.colors, dtype=tf.float32)
  value = tf.gather(colors, indices)

  return value







