# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils.py ]
#   Synopsis     [ utility functions for preprocess.py ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
import math
#-------------#
import librosa
import librosa.display
import numpy as np
from scipy import signal
from pydub import AudioSegment
#-------------#
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


############
# CONSTANT #
############
window_size = 256


##############
# GET MAPPER #
##############
def get_mapper(path):
	mapper = {}
	suffix = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
	with open(path, 'r', encoding='utf-8') as f:
		counter = 0
		lines = f.readlines()
		for line in lines:
			line = line.split()
			if len(line) != 0:
				head = line[0]
				if head == 'code': counter = 0
				elif head != 'code':
					counter += 1
					shift = 1 if counter == 5 else 0
					tail = line[1:]
					for i in range(len(tail)):
						cur_head = str(head[:3]) + suffix[i + shift]
						if cur_head not in mapper:
							mapper[cur_head] = tail[i]
						else: raise RuntimeError()
	return mapper

####################
# HIGH PASS FILTER #
####################
def highpass_filter(y, sr):
	filter_stop_freq = 100  # Hz
	filter_pass_freq = 300  # Hz
	filter_order = 1001

	# High-pass filter
	nyquist_rate = sr / 2.
	desired = (0, 0, 1, 1)
	bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
	filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

	# Apply high-pass filter
	filtered_audio = signal.filtfilt(filter_coefs, [1], y)
	yt = np.ascontiguousarray(filtered_audio)
	return yt



def match_target_amplitude(wav, suffix='wav', target_dBFS=-10.0):
	sound = AudioSegment.from_file(wav, suffix)
	change_in_dBFS = target_dBFS - sound.dBFS
	return sound.apply_gain(change_in_dBFS)


#################
# VISUALIZATION #
#################
"""
	visualize the preprocessed waveforms
"""
def visualization(name, y, yt, sr, output_dir, visualization_dir, multi_plot):
	#---visualization---#
	plt.figure(figsize=(16, 4))
	if multi_plot:
		plt.subplot(2, 1, 1)
		librosa.display.waveplot(yt, sr=sr, color='r')
		plt.title('Processed Waveform')
		plt.subplot(2, 1, 2)
	librosa.display.waveplot(y, sr=sr, color='tab:orange')
	plt.title('Original Waveform')
	plt.tight_layout()

	#---save---#
	plt.savefig(visualization_dir + name + '.jpeg')
	plt.close()



#########
# CHECK #
#########
"""
	Checks if all audios have been correctly processed,
	if not reprocess them.
"""
def check(input_dir, output_dir, file_suffix='*.wav'):
	redo_list = []
	
	#---get original file names---#
	wavs = sorted(glob.glob(os.path.join(input_dir, file_suffix)))
	for i in range(len(wavs)):
		wavs[i] = wavs[i].split('/')[-1] 
	
	#---get all preprocessed file names---#
	wavs_preprocess = sorted(glob.glob(os.path.join(output_dir, file_suffix)))
	for i in range(len(wavs_preprocess)):
		wavs_preprocess[i] = wavs_preprocess[i].split('/')[-1] 

	#---check for match and collect a redo list---#
	if len(wavs) != len(wavs_preprocess):
		for wav in wavs:
			if wav not in wavs_preprocess:
				redo_list.append(os.path.join(input_dir ,wav))
	
	#---reprocess---#
	if len(redo_list) != 0:
		print(redo_list)
		print('Found %i audio files that needs to be re-processed!' % len(redo_list))
	else:
		print('Audio pre-processing complete!' % len(redo_list))