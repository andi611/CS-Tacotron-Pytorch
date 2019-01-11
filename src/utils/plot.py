# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ plot.py ]
#   Synopsis     [ plot utility functions ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
from utils import audio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


##################
# PLOT ALIGNMENT #
##################
def plot_alignment(alignment, path, info=None):
	fig, ax = plt.subplots()
	im = ax.imshow(
		alignment,
		aspect='auto',
		origin='lower',
		interpolation='none')
	fig.colorbar(im, ax=ax)
	xlabel = 'Decoder timestep'
	if info is not None:
		xlabel += '\n\n' + info
	plt.xlabel(xlabel)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()
	plt.savefig(path, format='png')


####################
# PLOT SPECTROGRAM #
####################
def plot_spectrogram(path, linear_output):
	spectrogram = audio._denormalize(linear_output)
	plt.figure(figsize=(16, 10))
	plt.imshow(spectrogram.T, aspect="auto", origin="lower")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(path, format="png")
	plt.close()	
