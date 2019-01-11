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
#-------------#
from utils import audio
from functools import partial
from concurrent.futures import ProcessPoolExecutor


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
		print('Audio pre-processing complete!')


def write_metadata(metadata, out_dir, frame_shift_ms):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
		frames = sum([m[2] for m in metadata])
		hours = frames * frame_shift_ms / (3600 * 1000)
		print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
		print('Max input length:  %d' % max(len(m[3]) for m in metadata))
		print('Max output length: %d' % max(m[2] for m in metadata))


def build_from_path(meta_path, wav_dir, out_dir, num_workers=1, tqdm=lambda x: x):
	'''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

	Args:
		in_dir: The directory where you have downloaded the LJ Speech dataset
		out_dir: The directory to write the output into
		num_workers: Optional number of worker processes to parallelize across
		tqdm: You can optionally pass tqdm to get a nice progress bar

	Returns:
		A list of tuples describing the training examples. This should be written to train.txt
	'''

	# We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
	# can omit it and just call _process_utterance on each input if you want.
	executor = ProcessPoolExecutor(max_workers=num_workers)
	futures = []
	index = 1
	with open(meta_path, encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(wav_dir, '%s.wav' % parts[0])
			text = parts[2]
			futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
			index += 1
	return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
	'''Preprocesses a single utterance audio/text pair.

	This writes the mel and linear scale spectrograms to disk and returns a tuple to write
	to the train.txt file.

	Args:
	out_dir: The directory to write the spectrograms into
	index: The numeric index to use in the spectrogram filenames.
	wav_path: Path to the audio file containing the speech input
	text: The text spoken in the input audio file

	Returns:
	A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
	'''

	# Load the audio to a numpy array:
	wav = audio.load_wav(wav_path)

	# Compute the linear-scale spectrogram from the wav:
	spectrogram = audio.spectrogram(wav).astype(np.float32)
	n_frames = spectrogram.shape[1]

	# Compute a mel-scale spectrogram from the wav:
	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

	# Write the spectrograms to disk:
	spectrogram_filename = 'spec-%05d.npy' % index
	mel_filename = 'mel-%05d.npy' % index
	np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
	np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example:
	return (spectrogram_filename, mel_filename, n_frames, text)

