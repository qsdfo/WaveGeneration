#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def load_mat(chunk_list, batch_size, chunk_counter):
	"""Thread for loading matrices during training
	""" 
	
	matrix_return = None

	chunk_list_len = len(chunk_list)
	for batch_index in range(batch_size):
		# Load chunk
		chunk = np.load(chunk_list[chunk_counter])
		# Update counter
		chunk_counter = (chunk_counter + 1) % chunk_list_len
		# Instanciate matrix_return
		if matrix_return is None:
			matrix_return = np.zeros((batch_size, chunk.shape[0], chunk.shape[1]))
		# Write in matrix
		matrix_return[batch_index] = chunk
	return matrix_return, chunk_counter
