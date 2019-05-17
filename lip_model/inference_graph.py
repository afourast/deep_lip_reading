from distutils.version import StrictVersion

import tensorflow as tf
from tensorflow.python.util import nest

from config import load_args
from language_model.char_rnn_lm import CharRnnLm, CharRnnLmWrapperSingleton
from lip_model import beam_search
from lip_model.beam_search import log_prob_from_logits
from lip_model.training_graph import TransformerTrainGraph
from lip_model.modules import embedding, multihead_attention, feedforward
from util.tf_util import shape_list

config = load_args()

class TransformerInferenceGraph(TransformerTrainGraph):
  """
  This subclass only changes the decoder implementation to allow for a beam search.
  The encoding graph is the same as the one used for training.
  """

  def project_output(self):
    return False

  def decoder_body(self, enc, dec, is_training, top_scope=None):

    assert not is_training, 'Inference graph not to be used for training'

    inputs = enc
    batch_size = shape_list(dec)[0]
    decode_length = shape_list(dec)[1]

    # Create the positional encodings in advance,
    # so that we can add them on every time step within the loop
    timing_signal = self.positional_encoding(dec, scope='dec_pe')
    timing_signal = tf.expand_dims(timing_signal[0],0)

    symbols_to_logits_fn = self.get_symbols_to_logits_fun(enc,
                                                          timing_signal,
                                                          is_training,
                                                          top_scope)

    # Determine the batch_size of the logits. This will be diffent to the batch size if
    # we're doing test-time augmentation, as the different augmentations will have been
    # merged into 1 when we get to the logits
    logits_bs = 1 if config.test_aug_times else batch_size

    # Initialize cache and the decoding outputs to be filled in
    cache = self.initialize_cache(batch_size, enc)
    decoded_ids = tf.zeros([logits_bs, 0], dtype=tf.int64)
    decoded_logits = tf.zeros([logits_bs, 0, config.n_labels], dtype=tf.float32)
    next_id = self.go_token_idx*tf.ones([logits_bs, 1], dtype=tf.int64)

    # If we are using language model, get the symbols -> logprobs function for it
    lm_symbols_to_logprobs_fn = None
    if config.lm_path:
      lm_symbols_to_logprobs_fn = self.get_lm_symbols_to_logprobs_handle(cache,
                                                                         logits_bs,
                                                                         top_scope)

    scores = None

    if config.beam_size > 1:  # Beam Search
      vocab_size = config.n_labels
      initial_ids = next_id[:,0]
      decoded_ids, scores = beam_search.beam_search(
        symbols_to_logits_fn,
        lm_symbols_to_logprobs_fn,
        initial_ids,
        config.beam_size,
        decode_length,
        vocab_size,
        batch_size=logits_bs,
        batch_size_states=batch_size,
        alpha=config.len_alpha,
        lm_alpha=config.lm_alpha,
        states=cache,
        # stop_early=(config.top_beams == 1),
        stop_early=False)

      if config.top_beams == 1:
        decoded_ids = decoded_ids[:, 0, 1:]
        if StrictVersion(tf.__version__) >= StrictVersion('1.5'):
          decoded_ids.__dict__['_shape_val'] = tf.TensorShape([logits_bs, None])
        else:
          decoded_ids._shape = tf.TensorShape([logits_bs, None])
      else:
        decoded_ids = decoded_ids[:, :config.top_beams, 1:]
        if StrictVersion(tf.__version__) >= StrictVersion('1.5'):
          decoded_ids.decoded_ids.__dict__['_shape_val'] = \
                              tf.TensorShape([logits_bs, config.top_beams, None])
        else:
          decoded_ids._shape = tf.TensorShape([logits_bs, config.top_beams, None])

      decoded_logits = tf.zeros( [logits_bs, decode_length, config.n_labels] )

    else: # Greedy decoding
      def inner_loop(i, next_id, decoded_ids, decoded_logits, cache):
        logits, cache = symbols_to_logits_fn(next_id, i, cache)

        lip_logprobs = log_prob_from_logits(logits, axis=-1)
        if config.lm_path:
          lm_logprobs, cache = lm_symbols_to_logprobs_fn(next_id, i, cache)
        combined_scores = lip_logprobs
        if config.lm_path:
          combined_scores += config.lm_alpha * lm_logprobs
        next_id = tf.to_int64(tf.argmax(combined_scores, axis=-1))
        next_id = tf.expand_dims(next_id, axis=1)

        decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
        decoded_logits = tf.concat([decoded_logits,
                                    tf.expand_dims(logits, axis=1)], axis=1)
        return i + 1, next_id, decoded_ids, decoded_logits, cache

      _, _, decoded_ids, decoded_logits, _ = tf.while_loop(
        lambda i, *_: tf.less(i, decode_length),
        inner_loop,
        [tf.constant(0), next_id, decoded_ids, decoded_logits, cache],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([None, None]),
          tf.TensorShape([None, None]),
          tf.TensorShape([None, None, config.n_labels]),
          nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
        ])

    return decoded_ids, scores, decoded_logits

  def decoder_step(self, dec, cache, is_training, reuse_loop):

    # Initialize the masks for the pads from here,
    # because after positional embeddings are added, nothing will be 0
    key_masks_enc = key_masks_dec = query_masks_dec = None

    # TODO: for now only self-attention layers are cached in each block.
    #  Add caching of enc-dec att as well
    enc_output = cache["encoder_output"]

    for i in range(config.num_blocks):
      layer_name = "layer_{}".format(i)
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope("num_blocks_{}".format(i)):
        ## self-attention
        dec, alignmets = multihead_attention(queries=dec,
                                             query_masks=query_masks_dec,
                                             keys=dec,
                                             key_masks=key_masks_dec,
                                             num_units=config.hidden_units,
                                             num_heads=config.num_heads,
                                             dropout_rate=config.dropout_rate,
                                             is_training=is_training,
                                             causality=True,
                                             scope="self_attention",
                                             cache=layer_cache,
                                             reuse=reuse_loop)
        # self.alignment_history["dec_self_att_{}".format(i)] = alignmets # save for tb

        ## vanilla attention
        dec, alignmets = multihead_attention(queries=dec,
                                             query_masks=query_masks_dec,
                                             keys=enc_output,
                                             key_masks=key_masks_enc,
                                             num_units=config.hidden_units,
                                             num_heads=config.num_heads,
                                             dropout_rate=config.dropout_rate,
                                             is_training=is_training,
                                             causality=False,
                                             scope="vanilla_attention",
                                             reuse=reuse_loop
                                             )

        # self.alignment_history["dec_attention_{}".format(i)] = alignmets # save for tb

        ## Feed Forward
        dec = feedforward(dec,
                          num_units=[4 * config.hidden_units, config.hidden_units],
                          reuse=reuse_loop)

    return dec, cache

  def initialize_cache(self, batch_size, enc):
    key_channels = config.hidden_units
    value_channels = config.hidden_units
    num_layers = config.num_blocks
    cache = {
      "layer_{}".format(layer): {
        "k": tf.zeros([batch_size, 0, key_channels]),
        "v": tf.zeros([batch_size, 0, value_channels]),
      }
      for layer in range(num_layers)
    }
    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    # Note: Tensor.set_shape() does not work here since it merges shape info.
    # TODO: Setting __dict__['_shape_val'] is a hack here for newer tf versions that
    #       don't allow setting _shape It is done properly in newer T2T versions
    for layer in cache:
      if StrictVersion(tf.__version__) >= StrictVersion('1.5'):
        cache[layer]["k"].__dict__['_shape_val'] =\
          tf.TensorShape( [None, None, key_channels])
        cache[layer]["v"].__dict__['_shape_val'] =\
          tf.TensorShape( [None, None, value_channels])
      else:
        cache[layer]["k"]._shape = tf.TensorShape( [None, None, key_channels])
        cache[layer]["v"]._shape = tf.TensorShape( [None, None, value_channels])

    # The beam search handles beam-expanding the dims of the cache, so put this here
    cache[ "encoder_output"] = enc
    return cache

  def get_symbols_to_logits_fun(self, enc, timing_signal, is_training, top_scope):

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]

      # if i is a tf tensor then its in the while loop so only called once
      reuse_loop = i > 0 if isinstance(i, int) else None

      # targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = ids
      targets = preprocess_targets(targets, i, reuse_loop=reuse_loop)

      # -------- For test augmentation, we need to tile the previous input ----------
      if config.test_aug_times:
        targets = tf.tile(tf.expand_dims(targets, axis=0),
                          [config.test_aug_times] + [1] * (len(targets.shape)))
        bs, beam, t, c = shape_list(targets)
        targets = tf.reshape(targets, [bs * beam, t, c])

      pre_logits, cache = self.decoder_step(targets, cache, is_training,
                                            reuse_loop=reuse_loop)
      with tf.variable_scope(top_scope, reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(pre_logits, config.n_labels, reuse=reuse_loop)

      if config.test_aug_times:
        logits = tf.reshape(logits, [bs, beam, t, logits.shape[-1]])
        logits = tf.reduce_mean(logits, 0)

      return tf.squeeze(logits, axis=[1]), cache

    def preprocess_targets(targets, i, reuse_loop):
      """Performs preprocessing steps on the targets to prepare for the decoder.
      """
      targets = embedding(targets,
                          vocab_size=config.n_labels,
                          num_units=config.hidden_units,
                          scale=True,
                          scope="dec_embed",
                          reuse=reuse_loop)

      ## Positional Encoding
      targets += timing_signal[:, i:i + 1]

      ## Dropout
      targets = tf.layers.dropout(targets,
                                  rate=config.dropout_rate,
                                  training=tf.convert_to_tensor(is_training))
      return targets

    return symbols_to_logits_fn

  def get_lm_symbols_to_logprobs_handle(self, cache, logits_bs, top_scope):
    rnn_dict = CharRnnLmWrapperSingleton().rnn_clm  # it should alreadhy have been inited
    self.clm_opts = rnn_dict['saved_args']
    self.char_inds = dict((v, k) for k, v in enumerate(self.chars))
    # ==============  Table and keys that map from lm -> lip ids ================
    """ The lip model has an extra 5 characters:
    0: ' ' (pad) , 1: '!', 3: ',' 5: '.' and 45: '^' (go token)
    So we need to map from lip to lm indices to get the input to the LM and  
    from lm to lip on the lm output ( probs )
    """
    # Table mapping lip --> lm
    lip_to_lm_chars = dict( [(self.char_inds[chr], lm_idx)
                                 for chr, lm_idx in rnn_dict['char_inds'].items()
                                     if chr in self.char_inds])

    # We don't have a go token for the LM so replace this with '-' (space)
    lip_to_lm_chars[self.go_token_idx] = rnn_dict['char_inds']['-']
    # Map the lip padding to <space> for LM.
    # We do this because we would like the end of the
    # sentence to correspond to an end of a word
    lip_to_lm_chars[0] = rnn_dict['char_inds']['-']
    # Hashtable init crashes for int64 -> int32
    keys = tf.constant(list(lip_to_lm_chars.keys()), dtype=tf.int64)
    values = tf.constant(list(lip_to_lm_chars.values()), dtype=tf.int64)
    table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    # Keys mapping lm -> lip (used with gather to map 40 LM output probs to 45 characters)
    import copy
    lip_to_lm_chars_copy = copy.deepcopy(lip_to_lm_chars)  # just to make sure
    # we only want the go token to be mapped to  lm '-' for lip -> lm and not the inverse
    del lip_to_lm_chars_copy[self.go_token_idx]
    chars2lip_keys = []
    for lip_idx in range(len(self.char_inds)):
      chars2lip_keys.append(lip_to_lm_chars_copy[lip_idx]
                               if lip_idx in lip_to_lm_chars_copy else -1)
    import numpy as np
    chars2lip_keys = np.array(chars2lip_keys)
    zero_mask = 1 - (chars2lip_keys == -1).astype(np.float32)
    zero_mask_tf = tf.constant(zero_mask[None,:])
    # Replace the -1, to avoid error on CPU
    chars2lip_keys[chars2lip_keys == -1] = 0  # 0 index doesn't matter as we will mask it

    cache["lm_state"] = rnn_dict['model'].cell.zero_state(logits_bs, 'float32')

    # =============================================
    def lm_symbols_to_logprobs_fn(ids, unused_i, cache, inf_mask_tf=zero_mask_tf):
      """Go from ids to logits for next symbol."""

      ids = ids[:, -1:]
      # Map from lip to lm indices
      lm_ids = table.lookup(ids)

      with tf.control_dependencies([tf.assert_non_negative(lm_ids, [lm_ids])]):
        with tf.variable_scope(top_scope, reuse=True): # we have initialized LM outside
          model_pl = CharRnnLm(input_data=lm_ids,
                               initial_state=cache["lm_state"],
                               args=self.clm_opts,
                               training=False)
      probs_lm, cache["lm_state"] = model_pl.probs, model_pl.final_state

      lm_probs_mapped_to_lip = tf.gather(probs_lm, chars2lip_keys, axis=-1)
      lm_probs_mapped_to_lip *= zero_mask_tf # mask out indices with no lip correspondence
      lm_logprobs = tf.log(lm_probs_mapped_to_lip)  # Entries with value 0 will get -Inf
      return lm_logprobs, cache

    return lm_symbols_to_logprobs_fn



