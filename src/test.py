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
import argparse
import librosa
import librosa.display
from pypinyin import Style, pinyin
#--------------------------------#
from config import args
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize
#--------------------------------#
from synthesis import tts
from model.tacotron import Tacotron
from config import get_test_config




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
	config = get_test_config()

	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=args.embedding_dim,
					 mel_dim=args.num_mels,
					 linear_dim=args.num_freq,
					 r=args.outputs_per_step,
					 padding_idx=args.padding_idx,
					 use_memory_mask=args.use_memory_mask)

	#---handle path---#
	checkpoint_path = config.ckpt_dir + config.checkpoint_name + config.model + '.pth'
	output_name = config.result_dir + config.model
	os.makedirs(config.result_dir, exist_ok=True)
	
	#---load and set model---#
	print('Loading model: ', checkpoint_path)
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint["state_dict"])
	if config.long_input:
		model.decoder.max_decoder_steps = 500 # Set large max_decoder steps to handle long sentence outputs
	
	#---testing loop---#
	while True:
		try:
			text = str(input('< Tacotron > Text to speech: '))
			text = ch2pinyin(text)
			print('Model input: ', text)
			synthesis_speech(model, text=text, figures=config.plot, path=output_name)
		except KeyboardInterrupt:
			print()
			print('Terminating!')
			break

if __name__ == "__main__":
	main()
