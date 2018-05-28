#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import csv


def load_mat(chunk_list, csv_list, batch_size, chunk_counter):
	"""Thread for loading matrices during training
	""" 
	
	matrix_return = None
	chunk_list_len = len(chunk_list)
	for batch_index in range(batch_size):
		# Load chunk
		chunk = np.load(chunk_list[chunk_counter])
		# Load cond
		cond = np.loadtxt(csv_list[chunk_counter], delimiter=";", dtype=np.int)
		# Update counter
		chunk_counter = (chunk_counter + 1) % chunk_list_len
		# Instanciate matrix_return
		if matrix_return is None:
			matrix_return = np.zeros((batch_size, chunk.shape[0], chunk.shape[1]))
			cond_return = np.zeros((batch_size, len(cond)))
		# Write in matrix
		matrix_return[batch_index] = chunk
		cond_return[batch_index] = cond
	return matrix_return, cond_return, chunk_counter
