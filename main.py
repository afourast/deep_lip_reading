#!/usr/bin/env python
from __future__ import print_function

import os

import editdistance
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from config import load_args
from data.list_generator import ListGenerator
from language_model.char_rnn_lm import CharRnnLmWrapperSingleton
from lip_model.training_graph import TransformerTrainGraph
from lip_model.inference_graph import TransformerInferenceGraph

config = load_args()

graph_dict = {
              'train': TransformerTrainGraph,
              'infer': TransformerInferenceGraph,
              }

def evaluate_model():

    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)

    val_g, val_epoch_size, chars, sess, val_gen = init_models_and_data(istrain=0)

    tb_writer = None
    if config.tb_eval:
      import shutil
      try: shutil.rmtree('eval_tb_logs')
      except: pass
      tb_logdir = os.path.join(os.getcwd(), 'eval_tb_logs' , 'val')
      tb_writer = tf.summary.FileWriter(tb_logdir, sess.graph)

    with sess.as_default():
      for _ in range(config.n_eval_times):
        val_loss, val_cer, val_wer = validation_loop(sess, val_g,
                                                        val_epoch_size,
                                                        chars = chars,
                                                        val_gen = val_gen,
                                                        tb_writer = tb_writer,
                                                        )

        out_str = "lm={}, beam={}, bs={:d}, test_aug:{:d}, horflip {}:" \
                  " CER {:.4f}, WER {:4f}\n".format(config.lm_path,
                                                          config.beam_size,
                                                          config.batch_size,
                                                          config.test_aug_times,
                                                          config.horizontal_flip,
                                                          val_cer, val_wer)
        print(out_str)
        with open('output.txt', 'a') as fw:
          fw.write(out_str)

    print("Done")

def validation_loop(sess, g, n_batches, chars=None, val_gen = None, tb_writer=None):

  Loss = []
  Cer = []
  Wer = []

  progbar = Progbar(target=n_batches, verbose=1, stateful_metrics=['t'])
  print ('Strating validation Loop')

  for i in range(n_batches):

    x, y =  val_gen.next()
    if len(x) == 1: x = x[0]
    if len(y) == 1: y = y[0]

    # -- Autoregressive inference
    preds = np.zeros((config.batch_size, config.maxlen), np.int32)

    tile_preds = config.test_aug_times
    # -- For train graph feed in the previous step's predictions manually for the next
    if not 'infer' in config.graph_type:
      prev_inp = np.tile(preds, [config.test_aug_times,1]) if tile_preds else preds
      feed_dict = {g.x: x, g.prev: prev_inp, g.y: y}

      enc = sess.run( g.enc, feed_dict)
      if type(enc) is list:
        for enc_tens, enc_val in zip(g.enc, enc): feed_dict[enc_tens] = enc_val
      else:
        feed_dict[g.enc] = enc
      for j in range(config.maxlen):
        _preds, loss, cer = sess.run( [g.preds, g.mean_loss, g.cer], feed_dict)
        preds[:, j] = _preds[:, j]
        prev_inp = np.tile(preds, [config.test_aug_times,1]) if tile_preds else preds
        feed_dict[g.prev]=prev_inp
        # if all samples in batch predict the pad symbol (char_id==0)
        if np.sign(preds[:,j]).sum() == 0:
          if g.tb_sum is not None:
            tb_sum = sess.run( g.tb_sum, {g.x: x, g.prev: prev_inp, g.y: y})
          break

    # -- Autoregression loop is built into the beam search graph
    else:
      feed_dict = {g.x: x, g.y: y}
      enc = sess.run( g.enc, feed_dict)
      if type(enc) is list:
        for enc_tens, enc_val in zip(g.enc, enc): feed_dict[enc_tens] = enc_val
      else:
        feed_dict[g.enc] = enc
      _preds, loss, cer = sess.run( [g.preds, g.mean_loss, g.cer], feed_dict)
      preds = _preds

    # use last loss
    gt_sents = [ ''.join([ chars[cid] for cid in prr]).strip() for prr in y]
    gt_words = [ sent.split('-') for sent in gt_sents]

    def decode_preds_to_chars(decoding):
      return ''.join([ chars[cid] for cid in decoding]).strip()

    pred_sentences = [ decode_preds_to_chars(prr) for prr in preds]

    pred_words = [sent.split('-') for sent in  pred_sentences]

    edists = [rel_edist(gt, dec_str) for gt, dec_str in zip(gt_words, pred_words)]
    wer = np.mean(edists)

    # -- Write tb_summaries if any
    if g.tb_sum is not None:
      if wer == 0:
        tb_writer.add_summary(tb_sum, i)

    if config.print_predictions:
      print()
      for gts, prs, wr in zip(gt_sents, pred_sentences, edists):
        print ('(wer={:.1f}) {} --> {}'.format(wr*100, gts, prs))

    progbar.update(i+1, [ ('cer',cer), ('wer', wer) ] )
    Wer.append(wer)

    Cer.append(cer)
    Loss.append(loss)

  return np.average(Loss), np.average(Cer), np.average(Wer)

def init_models_and_data(istrain):

  print ('Loading data generators')
  val_gen, val_epoch_size = setup_generators()
  print ('Done')

  os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  sess = tf.Session(config=sess_config)

  if config.lm_path:
    # initialize singleton rnn so that RNN tf graph is created first
    beam_batch_size = 1
    lm_handle = CharRnnLmWrapperSingleton(batch_size=beam_batch_size,
                                          sess=sess,
                                          checkpoint_path=config.lm_path)

  TransformerGraphClass = graph_dict[config.graph_type]

  (shapes_in, dtypes_in), (shapes_out, dtypes_out) = \
    TransformerGraphClass.get_model_input_target_shapes_and_types()

  go_idx = val_gen.label_vectorizer.char_indices[val_gen.label_vectorizer.go_token]
  x_val = tf.placeholder(dtypes_in[0], shape=shapes_in[0])
  prev_shape = list(shapes_out[0])
  if config.test_aug_times : prev_shape[0] *= config.test_aug_times
  prev_ph = tf.placeholder(dtypes_out[0], shape=prev_shape)
  y_ph = tf.placeholder(dtypes_out[0], shape=shapes_out[0])
  y_val = [prev_ph, y_ph]

  chars = val_gen.label_vectorizer.chars
  val_g = TransformerGraphClass(x_val,
                                y_val,
                                is_training=False,
                                reuse=tf.AUTO_REUSE,
                                go_token_index=go_idx,
                                chars=chars)
  print("Validation Graph loaded")

  sess.run(tf.tables_initializer())

  load_checkpoints(sess)

  return val_g, val_epoch_size, chars, sess, val_gen

def load_checkpoints(sess, var_scopes = ('encoder', 'decoder', 'dense')):

  checkpoint_path =  config.lip_model_path
  if checkpoint_path:
    if os.path.isdir(checkpoint_path):
      checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    else:
      checkpoint = checkpoint_path

  if config.featurizer:

    if checkpoint_path:
      from tensorflow.contrib.framework.python.framework import checkpoint_utils
      var_list = checkpoint_utils.list_variables(checkpoint)
      for var in var_list:
        if 'visual_frontend' in var[0]:
          var_scopes = var_scopes + ('visual_frontend',)
          break

    if not 'visual_frontend' in var_scopes:
      featurizer_vars = tf.global_variables(scope='visual_frontend')
      featurizer_ckpt = tf.train.get_checkpoint_state(config.featurizer_model_path)
      featurizer_vars = [var for var in featurizer_vars if not 'Adam' in var.name]
      tf.train.Saver(featurizer_vars).restore(sess, featurizer_ckpt.model_checkpoint_path)

  all_variables = []
  for scope in var_scopes:
    all_variables += [var for var in tf.global_variables(scope=scope)
                      if not 'Adam' in var.name ]
  if checkpoint_path:
    tf.train.Saver(all_variables).restore(sess, checkpoint)

    print("Restored saved model {}!".format(checkpoint))

def setup_generators(verbose=False):
  val_gen = ListGenerator(data_list=config.data_list)
  val_epoch_size = val_gen.calc_nbatches_per_epoch()
  return val_gen, val_epoch_size

def rel_edist(tr, pred):
  return editdistance.eval(tr,pred) / float(len(tr))


def main():
  evaluate_model()

if __name__ == '__main__':
  main()

