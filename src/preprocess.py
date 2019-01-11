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
from hparams import hparams
from pypinyin import Style, pinyin
from multiprocessing import cpu_count


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='preprocess')

	parser.add_argument('--mode', choices=['text', 'audio', 'model_ready', 'analysis', 'all'], default='all', help='what to preprocess')
	parser.add_argument('--num_workers', type=int, default=cpu_count(), help='multi-thread processing')

	meta_path = parser.add_argument_group('meta_path')
	meta_path.add_argument('--meta_audio_dir', type=str, default='../data/meta/', help='path to the model ready training acoustic features')
	
	audio_path = parser.add_argument_group('audio_path')
	audio_path.add_argument('--audio_input_dir', type=str, default='../data/audio/original/', help='directory path to the original audio data')
	audio_path.add_argument('--audio_output_dir', type=str, default='../data/audio/processed/', help='directory path to output the processed audio data')
	audio_path.add_argument('--visualization_dir', type=str, default='../data/audio/visualization/', help='directory path to output the audio visualization images')
	
	text_path = parser.add_argument_group('text_path')
	text_path.add_argument('--text_dir', type=str, default='../data/text/', help='directory to the text transcripts')
	text_path.add_argument('--mapper_path', type=str, default='mapper.txt', help='path to the encoding mapper')

	input_path = parser.add_argument_group('text_input_path')
	input_path.add_argument('--text_input_train_path', type=str, default='train_ori.txt', help='path to the original training text data')
	input_path.add_argument('--text_input_dev_path', type=str, default='dev_ori.txt', help='path to the original development text data')
	input_path.add_argument('--text_input_test_path', type=str, default='test_ori.txt', help='path to the original testing text data')
	input_path.add_argument('--train_all_meta_path', type=str, default='../data/text/train_all_pinyin.txt', help='path to the model ready training text transcripts')
	
	output_path = parser.add_argument_group('text_output_path')
	output_path.add_argument('--text_output_train_path', type=str, default='train.txt', help='path to the processed training text data')
	output_path.add_argument('--text_output_dev_path', type=str, default='dev.txt', help='path to the processed development text data')
	output_path.add_argument('--text_output_test_path', type=str, default='test.txt', help='path to the processed testing text data')
	output_path.add_argument('--all_text_output_path', type=str, default='train_all.txt', help='path to the joint processed text data')

	args = parser.parse_args()
	return args


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

	args = get_config()
	
	#---preprocess text---#
	if args.mode == 'all' or args.mode == 'text':
		mapper = utils.get_mapper(os.path.join(args.text_dir, args.mapper_path))
		process_text(mapper, input_path=os.path.join(args.text_dir, args.text_input_train_path), output_path=os.path.join(args.text_dir, args.text_output_train_path))
		process_text(mapper, input_path=os.path.join(args.text_dir, args.text_input_dev_path), output_path=os.path.join(args.text_dir, args.text_output_dev_path))
		process_text(mapper, input_path=os.path.join(args.text_dir, args.text_input_test_path), output_path=os.path.join(args.text_dir, args.text_output_test_path))
		process_pinyin(args.train_all_meta_path, args.text_dir, args.all_text_output_path, [args.text_output_train_path, args.text_output_dev_path, args.text_output_test_path])		

	#---preprocess audio---#
	elif args.mode == 'all' or args.mode == 'audio':
		process_audio(args.audio_input_dir, 
					  args.audio_output_dir, 
					  args.visualization_dir, 
					  file_suffix='*.wav', 
					  start_from=0, 
					  multi_plot=True, 
					  vis_origin=False)
		utils.check(args.audio_input_dir, args.audio_output_dir, file_suffix='*.wav')

	#---preprocess text and data to be model ready---#
	elif args.mode == 'all' or args.mode == 'model_ready':
		make_meta(args.train_all_meta_path, args.audio_output_dir, args.meta_audio_dir, args.num_workers, hparams.frame_shift_ms)

	#---dataset analysis---#
	elif args.mode == 'all' or args.mode == 'analysis':
		dataset_analysis(args.audio_input_dir, args.text_dir, [args.text_output_train_path, args.text_output_dev_path, args.text_output_test_path])
	
	else:
		raise RuntimeError('Invalid mode!')



if __name__ == '__main__':
	main()