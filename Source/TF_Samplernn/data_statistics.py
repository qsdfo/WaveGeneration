import random
import numpy as np
import build_db
from samplernn.ops import mu_law_encode, linear_encode
import tensorflow as tf
import progressbar

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def bar_activations(data_directory, plot_path, seq_len):
	chunk_list = build_db.find_files(data_directory + '/chunk', pattern="*.npy")
	random.shuffle(chunk_list)
	activation_counter_lin = np.zeros((256))
	activation_counter_mu = np.zeros((256))

	chunk_PH = tf.placeholder(tf.float32, shape=(seq_len, 1))
	lin_encoded=tf.cast(linear_encode(chunk_PH, 256), tf.float32)
	mu_law_encoded=tf.cast(mu_law_encode(chunk_PH, 256), tf.float32)

	with tf.Session() as sess:
		for chunk_path in progressbar.progressbar(chunk_list):
			# Load chunk
			chunk = np.load(chunk_path)
			# Encode
			mu_encoded_ = sess.run(mu_law_encoded, feed_dict={chunk_PH: chunk})
			lin_encoded_ = sess.run(lin_encoded, feed_dict={chunk_PH: chunk})
			# Stats	
			for val in mu_encoded_:
				activation_counter_mu[int(val)]+= 1
			for val in lin_encoded_:
				activation_counter_lin[int(val)]+= 1
			
	fig, ax = plt.subplots(tight_layout=True)
	ax.bar(range(256), activation_counter_lin)
	fig.savefig(plot_path + '/activation_counter_lin.pdf')
	plt.clf()
	fig, ax = plt.subplots(tight_layout=True)
	ax.bar(range(256), activation_counter_mu)
	fig.savefig(plot_path + '/activation_counter_mu.pdf')
	return

if __name__ == '__main__':
	data_directory='/fast-1/leo/WaveGeneration/Data/ordinario_xs/8000_16392_0.01'
	main(data_directory='/fast-1/leo/WaveGeneration/Data/ordinario_xs/8000_16392_0.01')