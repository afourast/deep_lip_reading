import configargparse

def load_args(default_config=None):

  parser = configargparse.ArgumentParser(description = "main")

  parser.add_argument('--gpu_id', type=str, default='', help='Argument to set to CUDA_VISIBLE_DEVICES env variable. If nothing passed, CPU is used as default')

  parser.add_argument('--print_predictions', type=bool, default=True, help='Print the sentence predictions during evaluation')
  parser.add_argument('--seed', type=int, default=0, help='np random seed')
  parser.add_argument('--tb_eval', type=bool, default=False, help='Write tb summaries during evaluation')

  # Data
  parser.add_argument('--data_path', type=str, default='media/example', help='Path to the videos')
  parser.add_argument('--data_list', type=str, default='media/example/demo_list.txt', help='List of (video_path, transcription) entries. Video paths should be relative to --data_path')

  parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

  parser.add_argument('--img_width', type=int, default=160, help='Input video width')
  parser.add_argument('--img_height', type=int, default=160, help='Input video height')
  parser.add_argument('--img_channels', type=int, default=1, help='Number of input channels (3 for RGB, 1 for grayscale)')
  parser.add_argument('--n_labels', type=int, default=45, help='Number of output characters')
  parser.add_argument('--pad_mode', type=str, default='mid', help='Padding for video sequences, Can be one of [mid,end]')
  # Merge time_dim and maxlen
  parser.add_argument('--time_dim', type=int, default=145, help='Max input sequence length in video frames')
  parser.add_argument('--maxlen', type=int, default=145, help='Max output sequence length')
  parser.add_argument('--transcribe_digits', type=bool, default=False, help='Set to replace digits with their transcription')

  # Model
  parser.add_argument('--lip_model_path', type=str, default=None, help='Path to the trained model to evaluate')

  parser.add_argument('--graph_type', type=str, default='train', help='Train or infer graph')
  parser.add_argument('--net_input_size', type=int, default=112, help='Resolution of input to visual frontend')
  parser.add_argument('--featurizer', type=bool, default=True, help='')
  parser.add_argument('--feat_dim', type=int, default=512, help='Video features dimension - used if loading features directly (featurizer=False)')

  # Transformer config
  parser.add_argument('--num_blocks', type=int, default=6, help='# of transformer blocks')
  parser.add_argument('--hidden_units', type=int, default=512, help='Transformer model size')
  parser.add_argument('--num_heads', type=int, default=8, help='# of attention heads')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout probability')
  parser.add_argument('--sinusoid', type=bool, default=True, help='Use sinusoid positional encodings')
  parser.add_argument('--mask_pads', type=bool, default=False, help='Whether to mask the padded parts of sequence')

  # beam search
  parser.add_argument('--lm_path', type=str, default=None, help='The path to the language model. If not specified no lm is used')
  parser.add_argument('--beam_size', type=int, default=0, help='The beam width')
  parser.add_argument('--len_alpha', type=float, default=0.7, help='Length penalty hyperparameter')
  parser.add_argument('--lm_alpha', type=float, default=0.1, help='LM weight hyperparameter')
  parser.add_argument('--top_beams', type=int, default=1, help='How many hypotheses to return per sample')

  # Preprocessing
  parser.add_argument('--resize_input', type=int, default=160, help='Resize the input video frames to this resolution')
  parser.add_argument('--scale', type=float, default=1.4, help='Scale the input video with that factor')
  parser.add_argument('--mean', type=float, default=0.4161, help='Mean normalization (over gray channel)')
  parser.add_argument('--std', type=float, default=0.1688, help='Std normalization (over gray channel)')

  # test-time augmentation
  parser.add_argument('--horizontal_flip', type=bool, default=True, help='Horizontal flipping')
  parser.add_argument('--crop_pixels', type=int, default=0, help='Random croping of +- pixels')
  parser.add_argument('--test_aug_times', type=int, default=0, help='How many different test augmentations to perform (including original)')
  parser.add_argument('--n_eval_times', type=int, default=1, help='How many times to perform the evaluation - used for averaging over test augmentation runs')

  args = parser.parse_args()

  return args


