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
