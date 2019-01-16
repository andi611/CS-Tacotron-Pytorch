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
import sys
import nltk
import argparse
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from docopt import docopt
from pypinyin import Style, pinyin
#--------------------------------#
import torch
from torch.autograd import Variable
#--------------------------------#
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize, plot_alignment
#--------------------------------#
from model.tacotron import Tacotron
from config import config, get_test_args


############
# CONSTANT #
############
USE_CUDA = torch.cuda.is_available()


##################
# TEXT TO SPEECH #
##################
def tts(model, text):
	"""Convert text to speech waveform given a Tacotron model.
	"""
	if USE_CUDA:
		model = model.cuda()
	# TODO: Turning off dropout of decoder's prenet causes serious performance
	# regression, not sure why.
	# model.decoder.eval()
	model.encoder.eval()
	model.postnet.eval()

	sequence = np.array(text_to_sequence(text))
	sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
	if USE_CUDA:
		sequence = sequence.cuda()

	# Greedy decoding
	mel_outputs, linear_outputs, alignments = model(sequence)

	linear_output = linear_outputs[0].cpu().data.numpy()
	spectrogram = audio._denormalize(linear_output)
	alignment = alignments[0].cpu().data.numpy()

	# Predicted audio signal
	waveform = audio.inv_spectrogram(linear_output.T)

	return waveform, alignment, spectrogram


####################
# SYNTHESIS SPEECH #
####################
def synthesis_speech(model, text, figures=True, path=None):
	waveform, alignment, spectrogram = tts(model, text)
	if figures:
		test_visualize(alignment, spectrogram, path)
	librosa.output.write_wav(path + '.wav', waveform, config.sample_rate)


#############
# CH2PINYIN #
#############
def ch2pinyin(txt_ch):
	ans = pinyin(txt_ch, style=Style.TONE2, errors=lambda x: x, strict=False)
	return ' '.join([x[0] for x in ans if x[0] != 'EMPH_A'])


def synthesis():
	config = docopt(__doc__)
	print("Command line config:\n", config)
	checkpoint_path = config["<checkpoint>"]
	text_list_file_path = config["<text_list_file>"]
	dst_dir = config["<dst_dir>"]
	max_decoder_steps = int(config["--max-decoder-steps"])
	file_name_suffix = config["--file-name-suffix"]

	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=config.embedding_dim,
					 mel_dim=config.num_mels,
					 linear_dim=config.num_freq,
					 r=config.outputs_per_step,
					 padding_idx=config.padding_idx,
					 use_memory_mask=config.use_memory_mask,
					 )
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint["state_dict"])
	model.decoder.max_decoder_steps = max_decoder_steps

	os.makedirs(dst_dir, exist_ok=True)

	with open(text_list_file_path, "rb") as f:
		lines = f.readlines()
		for idx, line in enumerate(lines):
			text = line.decode("utf-8")[:-1]
			words = nltk.word_tokenize(text)
			print("{}: {} ({} chars, {} words)".format(idx, text, len(text), len(words)))
			waveform, alignment, _ = tts(model, text)
			dst_wav_path = os.path.join(dst_dir, "{}{}.wav".format(idx, file_name_suffix))
			dst_alignment_path = os.path.join(dst_dir, "{}_alignment.png".format(idx))
			plot_alignment(alignment.T, dst_alignment_path,
						   info="tacotron, {}".format(checkpoint_path))
			audio.save_wav(waveform, dst_wav_path)

	print("Finished! Check out {} for generated audio samples.".format(dst_dir))
	sys.exit(0)


########
# MAIN #
########
def main():

	#---initialize---#
	args = get_test_args()

	if args.interactive:

		model = Tacotron(n_vocab=len(symbols),
						 embedding_dim=config.embedding_dim,
						 mel_dim=config.num_mels,
						 linear_dim=config.num_freq,
						 r=config.outputs_per_step,
						 padding_idx=config.padding_idx,
						 use_memory_mask=config.use_memory_mask)

		#---handle path---#
		checkpoint_path = args.ckpt_dir + args.checkpoint_name + args.model + '.pth'
		output_name = args.result_dir + args.model
		os.makedirs(args.result_dir, exist_ok=True)
		
		#---load and set model---#
		print('Loading model: ', checkpoint_path)
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint["state_dict"])
		if args.long_input:
			model.decoder.max_decoder_steps = 500 # Set large max_decoder steps to handle long sentence outputs
		
		#---testing loop---#
		while True:
			try:
				text = str(input('< Tacotron > Text to speech: '))
				text = ch2pinyin(text)
				print('Model input: ', text)
				synthesis_speech(model, text=text, figures=args.plot, path=output_name)
			except KeyboardInterrupt:
				print()
				print('Terminating!')
				break
	else:
		synthesis()

if __name__ == "__main__":
	main()
