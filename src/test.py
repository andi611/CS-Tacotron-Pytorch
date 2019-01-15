# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess.py ]
#   Synopsis     [ preprocess text transcripts and audio speech of the LectureDSP dataset for the Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import torch
import numpy as np
import librosa
import librosa.display
from pypinyin import Style, pinyin
#--------------------------------#
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.utils import test_visualize
#--------------------------------#
from synthesis import tts
from model.tacotron import Tacotron
from config import args


#############
# CONSTANTS #
#############
hop_length = 250


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='preprocess')

	parser.add_argument('--plot', action='store_true', help='whether to plot')
	parser.add_argument('--long_input', action='store_true', help='whether to set the model for long input')

	path_parser = parser.add_argument_group('path')
	path_parser.add_argument('--result_dir', type=str, default='../result/', help='path to output test results')
	path_parser.add_argument('--ckpt_dir', type=str, default='../ckpt/', help='path to the directory where model checkpoints are saved')
	path_parser.add_argument('--checkpoint_name', type=str, default='checkpoint_step', help='model name prefix for checkpoint files')
	path_parser.add_argument('--model', type=str, default='130000', help='model step name for checkpoint files')
	
	args = parser.parse_args()
	return args


####################
# SYNTHESIS SPEECH #
####################
def synthesis_speech(model, text, figures=True, path=None):
	waveform, alignment, spectrogram = tts(model, text)
	if figures:
		test_visualize(alignment, spectrogram, path)
	librosa.output.write_wav(path + '.wav', waveform, args.sample_rate)


#############
# CH2PINYIN #
#############
def ch2pinyin(txt_ch):
	ans = pinyin(txt_ch, style=Style.TONE2, errors=lambda x: x, strict=False)
	return ' '.join([x[0] for x in ans if x[0] != 'EMPH_A'])


########
# MAIN #
########
def main():

	#---initialize---#
	args = get_config()
	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=256,
					 mel_dim=args.num_mels,
					 linear_dim=args.num_freq,
					 r=args.outputs_per_step,
					 padding_idx=args.padding_idx,
					 use_memory_mask=args.use_memory_mask)

	#---handle path---#
	checkpoint_path = args.ckpt_dir + checkpoint_name + model + '.pth'
	output_name = args.result_dir + args.model
	os.makedirs(args.result_dir, exist_ok=True)
	
	#---load and set model---#
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint["state_dict"])
	if args.long_input:
		model.decoder.max_decoder_steps = 500 # Set large max_decoder steps to handle long sentence outputs
	
	#---testing loop---#
	while True:
		try:
			text = str(input('< Tacotron > Text to speech: '))
			synthesis_speech(model, text=ch2pinyin(text), figures=args.plot, path=output_name)
		except KeyboardInterrupt:
			break

