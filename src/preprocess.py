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
import glob
import nltk
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from utils import utils
from config import args
from pypinyin import Style, pinyin
from multiprocessing import cpu_count
from config import get_preprocess_config


################
# PROCESS TEXT #
################
def process_text(mapper, input_path, output_path):
	input_file = []
	with open(input_path, 'r', encoding='utf-8') as r:
		lines = r.readlines()
		input_file = [ line.split() for line in lines ]

	with open(output_path, 'w', encoding='utf-8') as w:
		for line in input_file:
			w.write(line[0] + ' ')
			write = ''
			for word in line[1:]:
				for char in word.split(']['):
					try:	
						write += mapper[char.strip('[').strip(']')]
					except:
						write += word
				write += ' '
			w.write(write.strip() + '\n')


#################
# PROCESS AUDIO #
#################
"""
	Trim out the noisy parts in the audios,
	add begining and ending silence, and finally realign them.
	Reprocess .wav files into clean and aligned .wav audios files.
"""
def process_audio(input_dir, output_dir, visualization_dir, file_suffix='*.wav', start_from=0, multi_plot=False, vis_origin=False):

	if not os.path.isdir(input_dir):
		raise ValueError('Please make sure there are .wav files in the directory: ', input_dir)
	if os.path.isdir(output_dir) or os.path.isdir(visualization_dir):
		print('Output directories already exist, please remove these existing directories to proceed.')
		while True:
			proceed = str(input('Proceed? [y/n]: '))
			if proceed == 'y': break
			elif proceed == 'n': return
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if not os.path.isdir(visualization_dir):
		os.makedirs(visualization_dir)	

	wavs = sorted(glob.glob(os.path.join(input_dir, file_suffix)))
	
	if vis_origin:
		for wav in wavs:
			y, sr = librosa.load(wav)
			utils.visualization(wav.split('/')[-1].split('.')[0], y, None, sr, output_dir, visualization_dir, multi_plot=False)
	
	else:
		for i, wav in enumerate(tqdm(wavs)):
			if i + 1 >= start_from:
				
				y, sr = librosa.load(wav)
				yt = utils.highpass_filter(y, sr)

				name = wav.split('/')[-1].split('.')[0]
				new_wav = output_dir + name + '.wav'
				librosa.output.write_wav(path=new_wav, y=yt.astype(np.float32), sr=sr)

				sound = utils.match_target_amplitude(new_wav, suffix='wav', target_dBFS=-10.0)
				sound.export(new_wav, format="wav")

				yt, sr = librosa.load(new_wav)
				utils.visualization(name, y, yt, sr, output_dir, visualization_dir, multi_plot)

		print('Progress: %i/%i: Complete!' % (len(wavs), len(wavs)))


##################
# PROCESS PINYIN #
##################
def process_pinyin(meta_path, text_dir, all_text_output_path, text_input_file_list):
	
	all_text = []
	with open(os.path.join(text_dir, all_text_output_path), 'w', encoding='utf-8') as w:
		for input_path in text_input_file_list:
			input_path = os.path.join(text_dir, input_path)
			with open(input_path, 'r', encoding='utf-8') as r:
				lines = r.readlines()
				for line in lines: 
					w.write(line)

	def _ch2pinyin(txt_ch):
		ans = pinyin(txt_ch, style=Style.TONE2, errors=lambda x: x, strict=False)
		return [x[0] for x in ans if x[0] != 'EMPH_A']
	
	with open(meta_path, 'w') as w:
		with open(os.path.join(text_dir, all_text_output_path), 'r') as r:
			lines = r.readlines()
			for line in lines:
				tokens = line[:-1].split(' ')
				wid, txt_ch = tokens[0], ' '.join(_ch2pinyin(tokens[1:]))
				w.write(wid + '|' + txt_ch + '\n')


#############
# MAKE META #
#############
def make_meta(train_all_meta_path, input_wav_dir, meta_audio_dir, num_workers, frame_shift_ms):
	os.makedirs(meta_audio_dir, exist_ok=True)
	metadata = utils.build_from_path(train_all_meta_path, input_wav_dir, meta_audio_dir, num_workers, tqdm=tqdm)
	utils.write_meta_data(metadata, meta_audio_dir, frame_shift_ms)


####################
# DATASET ANALYSIS #
####################
def dataset_analysis(wav_dir, text_dir, text_input_file_list):
	nltk.download('wordnet')
	
	all_text = []
	all_audio = []
	for input_path in text_input_file_list:
		input_path = os.path.join(text_dir, input_path)
		with open(input_path, 'r', encoding='utf-8') as r:
			lines = r.readlines()
			for line in lines: 
				all_text.append(line.split()[1:])
				all_audio.append(line.split()[0])
	print('Training data count: ', len(all_text))
	
	line_switch_count = 0
	total_switch_count = 0
	
	for line in all_text:
		for text in line:
			if nltk.corpus.wordnet.synsets(text): total_switch_count += 1
		for text in line:
			if nltk.corpus.wordnet.synsets(text): 
				line_switch_count += 1
				break
	print('Total number of switches: ', total_switch_count)
	print('Total number of sentences containing a switch: ', line_switch_count)

	duration = 0.0
	max_d = 0
	min_d = 60
	for audio in tqdm(all_audio):
		y, sr = librosa.load(os.path.join(wav_dir, audio + '.wav'))
		d = librosa.get_duration(y=y, sr=sr)
		if d > max_d: max_d = d
		if d < min_d: min_d = d
		duration += d
	print('Switch frequency - total number of switch / hour: ', total_switch_count / (duration / 60**2))
	print('Speech total length (hr): ', duration / 60**2)
	print('Max duration (seconds): ', max_d)
	print('Min duration (seconds): ', min_d)
	print('Average duration (seconds): ', duration / len(all_audio))


########
# MAIN #
########
def main():

	config = get_preprocess_config()
	
	#---preprocess text---#
	if config.mode == 'all' or config.mode == 'text':
		mapper = utils.get_mapper(os.path.join(config.text_dir, config.mapper_path))
		process_text(mapper, input_path=os.path.join(config.text_dir, config.text_input_train_path), output_path=os.path.join(config.text_dir, config.text_output_train_path))
		process_text(mapper, input_path=os.path.join(config.text_dir, config.text_input_dev_path), output_path=os.path.join(config.text_dir, config.text_output_dev_path))
		process_text(mapper, input_path=os.path.join(config.text_dir, config.text_input_test_path), output_path=os.path.join(config.text_dir, config.text_output_test_path))
		process_pinyin(config.train_all_meta_path, config.text_dir, config.all_text_output_path, [config.text_output_train_path, config.text_output_dev_path, config.text_output_test_path])		

	#---preprocess audio---#
	elif config.mode == 'all' or config.mode == 'audio':
		process_audio(config.audio_input_dir, 
					  config.audio_output_dir, 
					  config.visualization_dir, 
					  file_suffix='*.wav', 
					  start_from=0, 
					  multi_plot=True, 
					  vis_origin=False)
		utils.check(config.audio_input_dir, config.audio_output_dir, file_suffix='*.wav')

	#---preprocess text and data to be model ready---#
	elif config.mode == 'all' or config.mode == 'model_ready':
		make_meta(config.train_all_meta_path, config.audio_output_dir, config.meta_audio_dir, config.num_workers, args.frame_shift_ms)

	#---dataset analysis---#
	elif config.mode == 'all' or config.mode == 'analysis':
		dataset_analysis(config.audio_input_dir, config.text_dir, [config.text_output_train_path, config.text_output_dev_path, config.text_output_test_path])
	
	else:
		raise RuntimeError('Invalid mode!')



if __name__ == '__main__':
	main()