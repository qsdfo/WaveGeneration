import numpy as np
import os
import shutil

from scipy.io import wavfile


def make_batch(chunk_list, chunk_counter, batch_size=32, quantization=256):
	inputs_matrix = None
	targets_matrix = None
	chunk_list_len = len(chunk_list)
	for batch_index in range(batch_size):
		# Load chunk
		data = np.load(chunk_list[chunk_counter])
		# Update counter
		chunk_counter = (chunk_counter + 1) % chunk_list_len
		
		# Quantize inputs
		bins = np.linspace(-1, 1, quantization)
		inputs = np.digitize(data[0:-1], bins, right=False) - 1
		inputs = bins[inputs]
		if inputs_matrix is None:
			inputs_matrix = np.zeros((batch_size, inputs.shape[0], inputs.shape[1]))
		inputs_matrix[batch_index] = inputs

		# Encode targets as ints.
		targets = (np.digitize(data[1::], bins, right=False) - 1)
		if targets_matrix is None:
			targets_matrix = np.zeros((batch_size, targets.shape[0]))
		targets_matrix[batch_index] = targets[:,0]

	return inputs_matrix, targets_matrix, chunk_counter

def init_directory(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.makedirs(path)
	return

def up_criterion(val_tab, epoch, number_strips=3, validation_order=2):
	#######################################
	# Article
	# Early stopping, but when ?
	# Lutz Prechelt
	# UP criterion
	UP = True
	OVERFITTING = True
	s = 0
	epsilon = 0.001
	while(UP and s < number_strips):
		t = epoch - s
		tmk = epoch - s - validation_order
		UP = val_tab[t] > val_tab[tmk] - epsilon * abs(val_tab[tmk])   # Avoid extremely small evolutions
		s = s + 1
		if not UP:
			OVERFITTING = False
	return OVERFITTING