import av
import numpy as np


def load_video_frames(path, maxlen, pad_mode, grayscale):

  if not path.endswith('.mp4') and not path.endswith('.avi'):
    path+='.mp4'
  mat = load_mp4(path, grayscale=grayscale)

  mat = mat.astype('float')/255.

  if pad_mode:
    if len(mat) > maxlen:
      return None
    mat = pad_seq(mat, mode = pad_mode, maxlen = maxlen)

  return mat

def load_mp4(vid_path, grayscale=False):

  container = av.open(vid_path)

  ims = [frame.to_image() for frame in container.decode(video=0)]

  if grayscale:
    ims = [im.convert('L') for im in ims ]
  ims_c = np.array([np.array(im) for im in ims])

  if grayscale:
    ims_c = np.expand_dims(ims_c,axis=3)

  return ims_c

def pad_seq(mat, mode, maxlen):
  dat = np.zeros( ( maxlen ,)  + mat.shape[1:], dtype=mat.dtype )
  if mode == 'end':
    dat[ :mat.shape[0] ] = mat
    mat = dat
  elif mode == 'mid':
    assert maxlen >= mat.shape[0]
    padlen = (maxlen - mat.shape[0])//2
    dat[ padlen:padlen + mat.shape[0] ] = mat
    mat = dat
  return mat
