"""
The MIT License (MIT)

Copyright (c) 2015 Sherjil Ozair
Copyright (c) 2019 Triantafyllos Afouras

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os

import tensorflow as tf
from six.moves import cPickle as pickle
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn


class CharRnnLm():
  def __init__(self,
               args,
               training=False,
               input_data=None,
               targets=None,
               initial_state=None):
    self.args = args
    if not training:
      # args.batch_size = 10
      args.seq_length = 1

    cells = []
    for _ in range(args.num_layers):

      from distutils.version import StrictVersion
      if StrictVersion(tf.__version__) >= StrictVersion('1.5'):
        cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, name='basic_lstm_cell')
      else:
        cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size)
      if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
        cell = rnn.DropoutWrapper(cell,
                                  input_keep_prob=args.input_keep_prob,
                                  output_keep_prob=args.output_keep_prob)
      cells.append(cell)

    self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

    if input_data is None:
      self.input_data = tf.placeholder(
        tf.int32, [args.batch_size, args.seq_length])
      self.targets = tf.placeholder(
        tf.int32, [args.batch_size, args.seq_length])
      self.initial_state = cell.zero_state(args.batch_size, tf.float32)
    else:
      self.input_data = input_data
      self.targets = None
      self.initial_state = initial_state

    with tf.variable_scope('rnnlm'):
      softmax_w = tf.get_variable("softmax_w",
                                  [args.rnn_size, args.vocab_size])
      softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

    embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
    inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    # dropout beta testing: double check which one should affect next line
    if training and args.output_keep_prob:
      inputs = tf.nn.dropout(inputs, args.output_keep_prob)

    inputs = tf.split(inputs, args.seq_length, 1)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    def loop(prev, _):
      prev = tf.matmul(prev, softmax_w) + softmax_b
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    loop_fn = loop if not training else None
    outputs, last_state = legacy_seq2seq.rnn_decoder(inputs,
                                                     self.initial_state,
                                                     cell,
                                                     loop_function=loop_fn,
                                                     scope='rnnlm')
    output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

    self.final_state = last_state

    self.logits = tf.matmul(output, softmax_w) + softmax_b
    self.probs = tf.nn.softmax(self.logits)

    # ======== Training ops etc
    if training:
      loss = legacy_seq2seq.sequence_loss_by_example(
        [self.logits],
        [tf.reshape(self.targets, [-1])],
        [tf.ones([args.batch_size * args.seq_length])])
      with tf.name_scope('cost'):
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

      self.lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                        args.grad_clip)
      with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(self.lr)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars))

      # instrument tensorboard
      tf.summary.histogram('logits', self.logits)
      tf.summary.histogram('loss', loss)
      tf.summary.scalar('train_loss', self.cost)


class CharRnnLmWrapperSingleton():
  # class singleton
  rnn_clm = None

  def __init__(self, batch_size=None, sess=None, checkpoint_path=None):
    if CharRnnLmWrapperSingleton.rnn_clm is None:
      CharRnnLmWrapperSingleton.rnn_clm = self.load_rnn_model(batch_size,
                                                              sess,
                                                              checkpoint_path)

  def load_rnn_model(self, batch_size, sess, checkpoint_path, load_weights=True):
    with open(os.path.join(checkpoint_path, 'config.pkl'), 'rb') as f:
      saved_args = pickle.load(f)
    with open(os.path.join(checkpoint_path, 'chars_vocab.pkl'), 'rb') as f:
      chars_lm, char_inds_lm = pickle.load(f)
    saved_args.batch_size = batch_size

    model = CharRnnLm(saved_args, training=False)
    if load_weights:
      if sess is None:
        sess = tf.Session()
      with sess.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)

    char_inds_lm['-'] = char_inds_lm[' ']
    del char_inds_lm[' ']
    rnn_clm = {'model': model,
               'chars': chars_lm,
               'char_inds': char_inds_lm,
               'saved_args': saved_args}
    return rnn_clm
