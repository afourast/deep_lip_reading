import tensorflow as tf
import numpy as np

from config import load_args
from lip_model.losses import cer
from lip_model.modules import embedding, sinusoid_encoding, multihead_attention, \
  feedforward, label_smoothing
from lip_model.visual_frontend import VisualFrontend
from util.tf_util import shape_list

config = load_args()

class TransformerTrainGraph():
  def __init__(self,
               x,
               y,
               is_training=True,
               reuse=None,
               embed_input=False,
               go_token_index=2,
               chars=None):

    self.is_training = is_training
    self.x = x

    if config.featurizer:
      vid_inp = x[0] if type(x) is tuple or type(x) is list else x
      istarget = tf.not_equal(vid_inp, 0)
      self.padding_mask = tf.to_float(tf.reduce_any(istarget, axis=[2,3,4]))

      with tf.variable_scope('visual_frontend', reuse=reuse):
        self.visual_frontend = VisualFrontend(vid_inp)
        vid_inp = self.visual_frontend.output
      vid_inp = vid_inp * tf.expand_dims(self.padding_mask,-1)
      # pad = 30
      # x = tf.keras.layers.ZeroPadding1D(padding=(pad, pad))(x)
      if type(x) is tuple or type(x) is list:
        x = [vid_inp] + list(x[1:])
      else:
        x = vid_inp

    if is_training:
      self.prev = y
      self.y = y
    else:
      # This is the partial prediction used for the autoregression -
      # augmented by one more element on every step when autoregression is on
      self.prev = y[0]
      self.y = y[1] # This is the whole ground truth transcription

    self.alignment_history = {} # to be filled in by decoder

    self.go_token_idx = go_token_index

    # define decoder inputs
    self.decoder_inputs = tf.concat(
      (tf.ones_like(self.prev[:, :1]) * go_token_index, self.prev[:, :-1]), -1)  # 2:<S>

    # Encoder
    self.enc = x
    with tf.variable_scope("encoder", reuse=reuse) as scope:
      self.enc = self.encoder_body(self.enc, is_training)

    # import ipdb; ipdb.set_trace()
    # Decoder
    self.dec = self.decoder_inputs

    top_scope = tf.get_variable_scope() # this is a hack to be able to use same model
    self.chars = chars # needed for decoding with external LM

    # --------------- index to char dict for summaries --------------------------------
    if chars is not None:
      keys = tf.constant( np.arange(len(chars)) , dtype=tf.int64)
      values = tf.constant(chars , dtype=tf.string)
      self.char_table = tf.contrib.lookup.HashTable(
                      tf.contrib.lookup.KeyValueTensorInitializer(keys, values), '')


    with tf.variable_scope("decoder", reuse=reuse) as scope:
      self.dec = self.decoder_body(self.enc, self.dec, is_training, top_scope=top_scope)
      if type(self.dec) == tuple:
        self.preds, self.scores, self.dec = self.dec # Inference graph output

    self.add_loss_and_metrics(reuse, is_training)

    if config.tb_eval:
      self.add_tb_summaries()
    self.tb_sum = tf.summary.merge_all()

  def project_output(self):
    return True

  def decoder_body(self, enc, dec, is_training, top_scope=None):
    # Initialize the masks for the pads from here,
    #  because after positional embeddings are added, nothing will be 0
    if config.mask_pads: # Guard this for backwards compatibility
      key_masks_enc = tf.sign(tf.abs(tf.reduce_sum(enc, axis=-1))) # (N, T_k)
      key_masks_dec = tf.cast( tf.sign(tf.abs(dec)), 'float32' ) # (N, T_k)
      query_masks_dec = tf.cast( tf.sign(tf.abs(dec)), 'float32' ) # (N, T_k)
    else:
      key_masks_enc = key_masks_dec = query_masks_dec = None

    ## Embedding
    dec = self.decoder_embeddings(dec, is_training)

    for i in range(config.num_blocks):
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
                                       scope="self_attention")
        # self.alignment_history["dec_self_att_{}".format(i)] = alignmets # save for tb

        ## vanilla attention
        dec, alignmets = multihead_attention(queries=dec,
                                       query_masks=query_masks_dec,
                                       keys=enc,
                                       key_masks=key_masks_enc,
                                       num_units=config.hidden_units,
                                       num_heads=config.num_heads,
                                       dropout_rate=config.dropout_rate,
                                       is_training=is_training,
                                       causality=False,
                                       scope="vanilla_attention")

        self.alignment_history["enc_dec_attention_{}".format(i)] = alignmets # save for tb

        ## Feed Forward
        dec = feedforward(dec, num_units=[4 * config.hidden_units, config.hidden_units])

    return dec

  def decoder_embeddings(self, decoder_inputs, is_training):
    dec = embedding(decoder_inputs,
                         vocab_size=config.n_labels,
                         num_units=config.hidden_units,
                         scale=True,
                         scope="dec_embed")

    # if self.is_training:
    #   dec = dec[:,:self.out_last_non_pad_idx]

    ## Positional Encoding
    pos = self.positional_encoding(decoder_inputs, scope='dec_pe')
    # if self.is_training:
    #   pos = pos[:,:self.out_last_non_pad_idx]
    dec += pos

    ## Dropout
    dec = tf.layers.dropout(dec,
                                 rate=config.dropout_rate,
                                 training=tf.convert_to_tensor(is_training))
    return dec

  def positional_encoding(self, inp, scope):
    if config.sinusoid:
      return sinusoid_encoding(inp,
                               num_units=config.hidden_units,
                               zero_pad=False,
                               scale=False,
                               scope=scope,
                               T  = config.maxlen
                               )
    else:
      return embedding(
        tf.tile(tf.expand_dims(tf.range(tf.shape(inp)[1]), 0),
                [tf.shape(inp)[0], 1]),
        vocab_size=config.maxlen,
        num_units=config.hidden_units,
        zero_pad=False,
        scale=False,
        scope="dec_pe")

  def encoder_body(self, enc, is_training):

    num_blocks = config.num_blocks

    if config.mask_pads: # Guard this for backwards compatibility
      # Initialize the masks for the pads from here,
      # because after positional embeddings are added, nothing will be 0
      key_masks = tf.sign(tf.abs(tf.reduce_sum(enc, axis=-1))) # (N, T_k)
      query_masks = tf.sign(tf.abs(tf.reduce_sum(enc, axis=-1))) # (N, T_k)
    else:
      key_masks = query_masks = None


    enc = self.encoder_embeddings(enc, is_training)

    for i in range(num_blocks):
      with tf.variable_scope("num_blocks_{}".format(i)):
        ### Multihead Attention
        enc, alignmets = multihead_attention(queries=enc,
                                       query_masks=query_masks,
                                       keys=enc,
                                       key_masks=key_masks,
                                       num_units=config.hidden_units,
                                       num_heads=config.num_heads,
                                       dropout_rate=config.dropout_rate,
                                       is_training=is_training,
                                       causality=False)
        # key_masks = query_masks = None #

        ### Feed Forward
        enc = feedforward(enc, num_units=[4 * config.hidden_units, config.hidden_units])

        # self.alignment_history["enc_self_att_{}".format(i)] = alignmets # save for tb

    return enc

  def encoder_embeddings(self, x, is_training, embed_input=0):
    # Embedding
    if embed_input:
      enc = embedding(x,
                      vocab_size=config.n_input_vocab,
                      num_units=config.hidden_units,
                      scale=True,
                      scope="enc_embed")
    else:
      enc = x

    ## Positional Encoding
    feat_dim = shape_list(enc)[-1]
    # if input features are not same size as transformer units, make a linear projection
    if not feat_dim == config.hidden_units:
      enc = tf.layers.dense(enc, config.hidden_units)

    enc += self.positional_encoding(enc, scope='enc_pe')

    if embed_input:
      ## Dropout
      enc = tf.layers.dropout(enc,
                                   rate=config.dropout_rate,
                                   training=tf.convert_to_tensor(is_training))
    return enc

  def add_loss_and_metrics(self, reuse, is_training):
    # Final linear projection
    if self.project_output():
      # with tf.variable_scope("ctc_conv1d_net/ctc_probs", reuse=reuse) as scope:
        self.logits = tf.layers.dense(self.dec, config.n_labels, reuse=reuse)
    else:
      assert(self.dec.get_shape()[-1].value == config.n_labels)
      self.logits = self.dec


    if config.test_aug_times:
      self.logits_aug = self.logits
      self.logits = tf.reduce_mean(self.logits, 0, keep_dims=True)

    self.istarget = tf.to_float(tf.not_equal(self.y, 0))

    # Loss
    self.y_one_hot = tf.one_hot(self.y, depth=config.n_labels)
    self.y_smoothed = label_smoothing(self.y_one_hot)

    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                        labels=self.y_smoothed)

    # we want to know when to stop so learn padding as well
    self.mean_loss = tf.reduce_sum(self.loss) / (tf.reduce_sum(self.istarget))
    self.logprobs = tf.log(tf.nn.softmax(self.logits))

    if not 'infer' in config.graph_type:
      self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
      self.cer, self.cer_per_sample = cer(self.y_one_hot, self.logits, return_all=True)
    else:
      self.preds = tf.to_int32(self.preds)
      one_hot_from_preds = tf.one_hot(self.preds, depth=config.n_labels)
      self.cer, self.cer_per_sample = cer(self.y_one_hot,
                                          one_hot_from_preds,
                                          return_all=True)

  def add_tb_summaries(self):
    from util.tb_util import add_gif_summary, colorize_image
    
    fps = 10
    timeline = False

    # ----------------    Add video summaries -------------------------------
    bs = int(self.visual_frontend.output.shape[0])
    b_id = 0
    non_pad_inds = tf.cast(tf.where(self.padding_mask[b_id] > 0)[:, 0], tf.int64)
    fr_in, to_in = non_pad_inds[0], non_pad_inds[-1] + 1 # For masking out input paddings
    add_gif_summary('1-video_input',
                self.visual_frontend.input[b_id][fr_in:to_in], fps=fps, timeline=timeline)
    if not config.test_aug_times:
      add_gif_summary('2-input_to_resnet',
              self.visual_frontend.aug_out[b_id][fr_in:to_in], fps=fps, timeline=timeline)
    else:
      # Viz the different test augmentations
      add_gif_summary('2-input_to_resnet',
                      tf.concat([ self.visual_frontend.aug_out[b_id][fr_in:to_in]
                            for b_id in xrange(bs) ], axis=2), fps=fps, timeline=timeline)

    # ----------------   Add text summaries -------------------------------
    pred_strings_tf = self.char_table.lookup(tf.cast(self.preds, tf.int64))
    joined_pred = tf.string_join(
      tf.split(pred_strings_tf, pred_strings_tf.shape[1], 1))[ :, 0]
    gt_strings_tf = self.char_table.lookup(tf.cast(self.y, tf.int64))
    joined_gt = tf.string_join(
      tf.split(gt_strings_tf, pred_strings_tf.shape[1], 1))[:, 0]
    joined_all = tf.string_join([joined_gt, joined_pred], ' --> ')
    tf.summary.text('Predictions', joined_all)

    # ----------------   Add image summaries -------------------------------
    all_atts = []
    for layer_name, alignment_history in self.alignment_history.items():
      for att_head_idx, attention_images in enumerate(alignment_history):
        all_atts.append(attention_images)
    avg_att = tf.exp(tf.reduce_mean(tf.log(all_atts), axis=0))

    # Permute and reshape (batch, t_dec, t_enc) --> (batch, t_enc, t_dec, 1)
    attention_img = tf.expand_dims(tf.transpose(avg_att, [0, 2, 1]), -1)
    attention_img *= 255 # Scale to range [0, 255]

    b_id = 0 # visualize only the first sample of the batch
    to_out =  tf.where( self.preds[b_id]> 0 )[-1][0] + 1 # To mask output paddings                                                                                                                               |~
    color_img = tf.map_fn( colorize_image, (attention_img[:, fr_in:to_in, :to_out]) )
    tf.summary.image("3-enc_dec_attention", color_img)

    # ----------------   Add image with subs summaries -------------------------------
    # import ipdb; ipdb.set_trace()
    add_gif_summary('4-subs',
          self.visual_frontend.input[b_id][fr_in:to_in], fps=fps, timeline=timeline,
          attention=attention_img[b_id][fr_in:to_in, :to_out,0], pred=joined_pred[b_id])

  @classmethod
  def get_input_shapes_and_types(cls, batch=0):

    input_types = []

    input_shape = []
    if config.featurizer:
      input_shape += [ (config.time_dim, config.img_width, config.img_height, config.img_channels) ]
      input_types +=['float32']
    else:
      input_shape += [ (config.time_dim, config.feat_dim) ]
      input_types +=['float32']

    if batch:
      input_shape = [ (config.batch_size,) + shape for shape in input_shape ]

    return input_shape, input_types


  @classmethod
  def get_target_shapes_and_types(cls, batch=0):
    target_shape = [ ( config.time_dim, ) ]
    if batch:
      target_shape = [ (config.batch_size,) + shape for shape in target_shape ]
    target_types = ['int64']
    return target_shape, target_types

  @classmethod
  def get_model_input_target_shapes_and_types(cls, batch_dims=1):
    return cls.get_input_shapes_and_types(batch=batch_dims),\
           cls.get_target_shapes_and_types(batch=batch_dims)


















