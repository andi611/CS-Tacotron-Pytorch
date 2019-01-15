import tensorflow as tf


# # Default hyperparameters:
# hparams = tf.contrib.training.HParams(
# 	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
# 	# text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
# 	cleaners=None,
# 	use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

# 	# Audio:
# 	num_mels=80,
# 	num_freq=1025,
# 	sample_rate=20000,
# 	frame_length_ms=50,
# 	frame_shift_ms=12.5,
# 	preemphasis=0.97,
# 	min_level_db=-100,
# 	ref_level_db=20,

# 	# Model:
# 	# TODO: add more configurable hparams
# 	outputs_per_step=5,
# 	padding_idx=None,
# 	use_memory_mask=False,

# 	# Data loader
# 	pin_memory=True,
# 	num_workers=2,

# 	# Training:
# 	batch_size=16,
# 	adam_beta1=0.9,
# 	adam_beta2=0.999,
# 	initial_learning_rate=0.002,
# 	decay_learning_rate=True,
# 	nepochs=1000,
# 	weight_decay=0.0,
# 	clip_thresh=1.0,

# 	# Save
# 	checkpoint_interval=2000,

# 	# Eval:
# 	max_iters=200,
# 	griffin_lim_iters=60,
# 	power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
# )

##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='arguments')

	parser.add_argument('--checkpoint_dir', type=str, default='../ckpt', help='Directory where to save model checkpoints')
	parser.add_argument('--checkpoint_path', type=str, default=None, help='Restore model from checkpoint path if given')
	parser.add_argument('--data_root', type=str, default='../data/meta', help='Directory contains preprocessed features')

	audio_parser = parser.add_argument_group('audio')
	audio_parser.add_argument('--num_mels', type=int, default=80)
	audio_parser.add_argument('--num_freq', type=int, default=1025)
	audio_parser.add_argument('--sample_rate', type=int, default=20000)
	audio_parser.add_argument('--frame_length_ms', type=int, default=50)
	audio_parser.add_argument('--frame_shift_ms', type=float, default=12.5)
	audio_parser.add_argument('--preemphasis', type=float, default=0.97)
	audio_parser.add_argument('--min_level_db', type=int, default=-100)
	audio_parser.add_argument('--ref_level_db', type=int, default=20)
	
	model_parser = parser.add_argument_group('model')
	model_parser.add_argument('--outputs_per_step', type=int, default=5)
	model_parser.add_argument('--padding_idx', type=int, default=None)
	model_parser.add_argument('--use_memory_mask', type=bool, default=False)

	dataloader_parser = parser.add_argument_group('dataloader')
	dataloader_parser.add_argument('--pin_memory', type=bool, default=True)
	dataloader_parser.add_argument('--num_workers', type=int, default=2)

	training_parser = parser.add_argument_group('training')
	training_parser.add_argument('--batch_size', type=int, default=16)
	training_parser.add_argument('--adam_beta1', type=float, default=0.9)
	training_parser.add_argument('--adam_beta2', type=float, default=0.999)
	training_parser.add_argument('--initial_learning_rate', type=float, default=0.002)
	training_parser.add_argument('--decay_learning_rate', type=bool, default=True)
	training_parser.add_argument('--nepochs', type=int, default=1000)
	training_parser.add_argument('--weight_decay', type=float, default=0.0)
	training_parser.add_argument('--clip_thresh', type=float, default=1.0)
	training_parser.add_argument('--checkpoint_interval', type=int, default=2000)

	testing_parser = parser.add_argument_group('testing')
	max_iters_parser.add_argument('--max_iters', type=int, default=200)
	max_iters_parser.add_argument('--griffin_lim_iters', type=int, default=60)
	max_iters_parser.add_argument('--power', type=float, default=1.5)

	args = parser.parse_args()
	return args

args = get_config()

# def hparams_debug_string():
# 	values = hparams.values()
# 	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
# 	return 'Hyperparameters:\n' + '\n'.join(hp)
