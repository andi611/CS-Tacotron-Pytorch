# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ config.py ]
#   Synopsis     [ configurations ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import argparse


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='arguments')

	parser.add_argument('--checkpoint_dir', type=str, default='../ckpt', help='Directory where to save model checkpoints')
	parser.add_argument('--checkpoint_path', type=str, default=None, help='Restore model from checkpoint path if given')
	parser.add_argument('--data_root', type=str, default='../data/meta', help='Directory contains preprocessed features')
	parser.add_argument('--meta_text', type=str, default='meta_text.txt', help='model-ready training text data')

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
	model_parser.add_argument('--embedding_dim', type=int, default=256)
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
	testing_parser.add_argument('--max_iters', type=int, default=200)
	testing_parser.add_argument('--griffin_lim_iters', type=int, default=60)
	testing_parser.add_argument('--power', type=float, default=1.5, help='Power to raise magnitudes to prior to Griffin-Lim')

	args = parser.parse_args()
	return args

args = get_config()

