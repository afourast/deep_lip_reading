import os
import threading

import numpy as np

from config import load_args
from data.label_vectorization import SentenceVectorizer
from data.load_video import load_video_frames

config = load_args()

class ListGenerator:

  def __iter__(self):
    return self

  def __init__(self, data_list):

    self.lock = threading.Lock()

    self.batch_size = config.batch_size

    self.data_path = config.data_path

    self.all_samples = np.loadtxt(data_list, str, delimiter=', ')
    if self.all_samples.ndim == 1:
      self.all_samples = self.all_samples[None,:] # only one sample, expand batch dim
    assert self.all_samples.size > 0, "No samples found, please check the paths"
    self._tot_samps = len(self.all_samples)
    print("Found {} samples".format(self._tot_samps))

    self.v_idx = 0

    self.label_vectorizer = SentenceVectorizer(label_maxlen=config.maxlen,
                                               transcribe_digits=config.transcribe_digits)

  def calc_nbatches_per_epoch(self):
    return self.__len__()//self.batch_size

  def __len__(self):
    return self._tot_samps

  def next(self):

    frames_batch = []
    labels_batch = []

    video_frames = []
    cnt = 0
    while cnt< self.batch_size:
      with self.lock:
        vid, label = self.all_samples[self.v_idx]
        self.v_idx += 1
      frames = load_video_frames( os.path.join(self.data_path, vid),
                                  maxlen=config.maxlen,
                                  pad_mode=config.pad_mode,
                                  grayscale=config.img_channels == 1
                                  )
      video_frames.append(frames)
      labels_batch.append(label)
      cnt+=1

    assert len(video_frames) == self.batch_size
    video_frames = np.stack(video_frames, axis = 0)
    labels_batch = [self.label_vectorizer.vectorize(labels_batch)]
    frames_batch = [video_frames]

    return [frames_batch, labels_batch]

  def strip_extension(self,path):
    _, file_extension = os.path.splitext(path)
    path = path.replace(file_extension,'')
    return path
