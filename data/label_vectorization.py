import numpy as np

class SentenceVectorizer(object):
  """
  Input: String of characters
  Output: Padded vector of indices
  """
  def __init__(self, label_maxlen, transcribe_digits = None):

    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.!?:'")

    self.space_token = '-'
    self.pad_token = ' '
    self.go_token = '^'

    chars += [self.space_token,self.pad_token]
    self.chars = list(sorted(set(chars)))
    self.chars.append(self.go_token) # go token goes last
    self.char_indices = dict((c, i) for i, c in enumerate(self.chars))

    self.label_maxlen = label_maxlen
    self.transcribe_digits = transcribe_digits

  def vectorize(self,labels):
    targets = []
    for label in labels:
      if self.transcribe_digits:
        label = self.transcribe_digits_fun(label)
      targ = self.encode_label(label)
      targets.append(targ)
    return np.array(targets)

  def encode_label(self,label):
    label = label.replace(' ', self.space_token)
    label = [c for c in label]
    # Add pads
    npad = self.label_maxlen - len(label)
    label += [self.pad_token] * npad

    x = np.zeros(self.label_maxlen, dtype='int32')
    for i, c in enumerate(label):
      x[i] = self.char_indices[c]
    return x

  def transcribe_digits_fun(self,words):

    from num2words import num2words
    def is_number(s):
      try:
        int(s)
        return 'int'
      except ValueError:
        pass
      try:
        float(s)
        return 'float'
      except ValueError:
        return False

    words = words.split()
    new_words = []
    mutliple_word_number_found = False
    for word in words:
      if is_number(word):
        transcr = num2words(int(word)).upper().replace(',', '').replace('-', ' ').split()
        if len(transcr)>1:
          mutliple_word_number_found = True
        new_words += transcr
      else:
        new_words.append(word)

    new_words = ' '.join(new_words)

    if not new_words == words:
      print (words + ' ---> ' + new_words)

    return new_words

