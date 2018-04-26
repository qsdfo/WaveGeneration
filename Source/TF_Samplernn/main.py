import argparse
from datetime import datetime
import json
import os
import random
import sys
import shutil
import time
import pickle as pkl

import librosa
import numpy as np
import tensorflow as tf
import build_db
from tensorflow.python.client import timeline
from samplernn import SampleRnnModel, AudioReader, mu_law_decode, optimizer_factory
from keras import backend as K

from asynchronous_load_mat import load_mat
import build_db
import early_stopping
import samplernn.ops as ops


PREFIX="/fast-1/leo"
DATA_DIRECTORY=PREFIX+'/WaveGeneration/Data/contrabass_no_cond/ordinario'
LOGDIR_ROOT=PREFIX+'/WaveGeneration/logdir/1'

CHECKPOINT_EVERY=20
NUM_STEPS=int(1e5)
LEARNING_RATE=1e-3
L2_REGULARIZATION_STRENGTH=0
DROPOUT=0
MOMENTUM=0.9
MAX_TO_KEEP=5

TIERS="8,2,2"
RNNS="100,101"
MLPS="200,201,202"
Q_LEVELS=256        # Quantification for the amplitude of the audio samples
SEQ_LEN=512 		# Size for one BPTT pass
EMB_SIZE=256
OPTIMIZER='adam'

N_SECS=3
NUM_EXEMPLE_GENERATED=10

BATCH_SIZE=64
NUM_GPU=1

SAMPLE_RATE=8000
SAMPLE_SIZE=2**14+8
SLIDING_RATIO=0.75
SILENCE_THRESHOLD=0.01

def get_arguments():
	parser = argparse.ArgumentParser(description='SampleRnn example network')
	# Framework
	parser.add_argument('--num_gpus',         type=int,   default=NUM_GPU)
	parser.add_argument('--data_dir',         type=str,   default=DATA_DIRECTORY)
	parser.add_argument('--logdir_root',      type=str,   default=LOGDIR_ROOT)
	# Data
	parser.add_argument('--sample_rate',      type=int,   default=SAMPLE_RATE)
	parser.add_argument('--sample_size',      type=int,   default=SAMPLE_SIZE)
	parser.add_argument('--sliding_ratio',    type=float, default=SLIDING_RATIO)
	parser.add_argument('--silence_threshold',type=int,   default=SILENCE_THRESHOLD)
	# Architecture
	parser.add_argument('--tiers', 			  type=str,   default=TIERS)
	parser.add_argument('--rnns', 			  type=str,   default=RNNS)
	parser.add_argument('--mlps', 			  type=str,   default=MLPS)
	parser.add_argument('--q_levels',         type=int,   default=Q_LEVELS)
	parser.add_argument('--emb_size',         type=int,   default=EMB_SIZE)
	# Regularization
	parser.add_argument('--dropout', type=float, default=DROPOUT)	
	parser.add_argument('--l2_regularization_strength', type=float, default=L2_REGULARIZATION_STRENGTH)	
	# Optim
	parser.add_argument('--optimizer',        type=str,   default=OPTIMIZER, choices=list(optimizer_factory.keys()))
	parser.add_argument('--learning_rate',    type=float, default=LEARNING_RATE)
	parser.add_argument('--momentum',         type=float, default=MOMENTUM)
	# Training
	parser.add_argument('--seq_len',          type=int, default=SEQ_LEN)	# Use for BPTT
	parser.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
	parser.add_argument('--num_steps',        type=int,   default=NUM_STEPS)
	parser.add_argument('--checkpoint_every', type=int,   default=CHECKPOINT_EVERY)
	parser.add_argument('--max_checkpoints',  type=int, default=MAX_TO_KEEP)
	parser.add_argument('--load_existing_model',  type=int, default=False)
	# Generation
	parser.add_argument('--num_example_generated',  type=int, default=NUM_EXEMPLE_GENERATED)
	# Debug
	parser.add_argument('--summary',  type=bool, default=False)
	return parser.parse_args()

def save(saver, sess, logdir, step):
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(logdir, model_name)
		print('Storing checkpoint to {} ...'.format(logdir), end="")
		sys.stdout.flush()

		if not os.path.exists(logdir):
			os.makedirs(logdir)

		saver.save(sess, checkpoint_path, global_step=step)
		print(' Done.')


def load(saver, sess, logdir):
	print("Trying to restore saved checkpoints from {} ...".format(logdir),
				end="")

	ckpt = tf.train.get_checkpoint_state(logdir)
	if ckpt:
		print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
		global_step = int(ckpt.model_checkpoint_path
											.split('/')[-1]
											.split('-')[-1])
		print("  Global step was: {}".format(global_step))
		print("  Restoring...", end="")
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" Done.")
		return global_step
	else:
		print(" No checkpoint found.")
		return None

def create_model(args):
	tiers = [int(item) for item in args.tiers.split(',')]
	rnns = [int(item) for item in args.rnns.split(',')]
	mlps = [int(item) for item in args.mlps.split(',')]
	# Create network.
	net = SampleRnnModel(
		tiers=tiers,
		rnns=rnns,
		mlps=mlps,
		q_levels=args.q_levels,
		emb_size=args.emb_size,
		dropout=args.dropout,
		summary=args.summary)
	return net

# GENERATE 
def create_gen_wav_para(net):
	with tf.name_scope('generation'):
		gen_input = {}
		gen_output = {}

		gen_input['big_frame'] = tf.placeholder(tf.float32, shape=(None, net.big_frame_size, 1), name="gen_big_frame")
		gen_input['big_frame_state'] = []
		for layer, rnn_dim in enumerate(net.rnns):
			gen_input['big_frame_state'].append(tf.placeholder(tf.float32, shape=(None, rnn_dim), name="gen_big_frame_state_" + str(layer)))

		gen_input['frame_from_big'] = tf.placeholder(tf.float32, shape=(None, 1, net.rnns[0]), name='gen_frame_from_big')
		gen_input['frame'] = tf.placeholder(tf.float32, (None, net.frame_size, 1), name="gen_frame")
		gen_input['frame_state'] = []
		for layer, rnn_dim in enumerate(net.rnns):
			gen_input['frame_state'].append(tf.placeholder(tf.float32, (None, rnn_dim), name="gen_frame_state"))

		gen_input['sample_from_frame'] = tf.placeholder(tf.float32, shape=(None, 1, net.mlps[0]), name='gen_frame_from_big')
		gen_input['sample'] = tf.placeholder(tf.int32, (None, net.autoregressive_order, 1), name="gen_frame")

		gen_output['big_frame'], gen_output['big_frame_state'] = net._create_network_BigFrame(big_frame_states=gen_input['big_frame_state'],
			big_input_sequences=gen_input['big_frame'],
			seq_len=None)

		gen_output['frame'], gen_output['frame_state'] = net._create_network_Frame(
			big_frame_outputs=gen_input['frame_from_big'],
			frame_states=gen_input['frame_state'],
			input_sequences=gen_input['frame'],
			seq_len=None)

		sample_out = net._create_network_Sample(frame_outputs=gen_input['sample_from_frame'],
			sample_input_sequences=gen_input['sample'],
			seq_len=None)

		sample_out = tf.reshape(sample_out, [-1, net.q_levels])
		gen_output['sample'] = tf.cast(tf.nn.softmax(tf.cast(sample_out, tf.float64)), tf.float32)

		gen_input['sample_to_decode'] = tf.placeholder(tf.int32)
		gen_output['sample_decoded'] = mu_law_decode(gen_input['sample_to_decode'], net.q_levels)

		return gen_input, gen_output

def write_wav(waveform, sample_rate, filename):
	y = np.array(waveform)
	librosa.output.write_wav(filename, y, sample_rate)
	print('Updated wav file at {}'.format(filename))

def generate_and_save_samples(step, length, sample_rate, num_example_generated, net, gen_input, gen_output, sess, write_dir):
	# Initialize sequence to generate
	samples = np.zeros((num_example_generated, length, 1), dtype='int32')
	samples[:, :net.big_frame_size,:] = np.int32(net.q_levels//2)

	# Initialize rnn_states
	final_big_s = []
	final_s = []
	for rnn_dim in net.rnns:
		final_big_s.append(np.zeros((num_example_generated, rnn_dim), dtype=np.float32))
		final_s.append(np.zeros((num_example_generated, rnn_dim), dtype=np.float32))
	
	# Output of different levels of RNN
	big_frame_out = None
	frame_out = None
	sample_out = None
	
	for t in range(net.big_frame_size, length):
		#big frame 
		if t % net.big_frame_size == 0:
			big_frame_out = None
			big_input_sequences = samples[:, t-net.big_frame_size:t,:].astype('float32')
			inp_dict={}
			inp_dict[gen_input['big_frame']] = big_input_sequences
			for state_PH, state_value in zip(gen_input['big_frame_state'], final_big_s):
				inp_dict[state_PH] = state_value
			inp_dict[K.learning_phase()] = 0
			big_frame_out, final_big_s = sess.run([gen_output['big_frame'] , gen_output['big_frame_state']], feed_dict=inp_dict)		

		#frame 
		if t % net.frame_size == 0:
			frame_input_sequences = samples[:, t-net.frame_size:t,:].astype('float32')
			big_frame_output_idx = (t//net.frame_size)%(net.big_frame_size//net.frame_size)
			inp_dict = {}
			inp_dict[gen_input['frame_from_big']] = big_frame_out[:,[big_frame_output_idx],:]
			inp_dict[gen_input['frame']] = frame_input_sequences
			for state_PH, state_value in zip(gen_input['frame_state'], final_s):
				inp_dict[state_PH] = state_value
			inp_dict[K.learning_phase()] = 0
			frame_out, final_s = sess.run([gen_output['frame'], gen_output['frame_state']], feed_dict=inp_dict)
		
		#sample
		sample_input_sequences = samples[:, t-net.autoregressive_order:t,:]
		frame_output_idx = t%net.frame_size
		sample_out= sess.run(gen_output['sample'],
			feed_dict={gen_input['sample_from_frame'] : frame_out[:,[frame_output_idx],:],
				gen_input['sample'] : sample_input_sequences,
				K.learning_phase() : 0})

		# Sample from the softmax distribution sample_out
		sample_next_list = []
		for row in sample_out:
			sample_next = np.random.choice(np.arange(net.q_levels), p=row)
			sample_next_list.append(sample_next)
		samples[:, t] = np.array(sample_next_list).reshape([-1,1])

	# Decode mu_law
	for i in range(0, num_example_generated):
		inp = samples[i].reshape([-1,1]).tolist()
		out = sess.run(gen_output['sample_decoded'], feed_dict={gen_input['sample_to_decode']: inp, K.learning_phase() : 0})
		write_wav(out, sample_rate, write_dir + '/' + str(step)+'_'+str(i)+'.wav')
	return
					
def main():

	##############################
	# Get args	
	args = get_arguments()
	if args.l2_regularization_strength == 0:
			args.l2_regularization_strength = None
	tiers = [int(item) for item in args.tiers.split(',')]
	rnns = [int(item) for item in args.rnns.split(',')]
	mlps = [int(item) for item in args.mlps.split(',')]
	seq_len_padded = args.seq_len + tiers[0]
	##############################

	##############################
	# Get data directory
	config_str = "_".join([str(args.sample_rate), str(args.sample_size), str(args.sliding_ratio), str(args.silence_threshold)])
	files_dir = args.data_dir
	npy_dir = files_dir + '/' + config_str
	# Check if exists
	if not os.path.isdir(npy_dir):
		# Build if not
		build_db.main(files_dir, npy_dir, args.sample_rate, args.sample_size, args.sliding_ratio, args.silence_threshold)
		# data_statistics.bar_activations(save_dir, save_dir, sample_size)
	##############################

	##############################
	# Init dirs
	ops.init_directory(args.logdir_root)
	if args.summary:
		logdir_summary = os.path.join(args.logdir_root, 'summary')
		ops.init_directory(logdir_summary)
	# Save	
	logdir_save = os.path.join(args.logdir_root, 'save')
	# Wave
	logdir_wav = os.path.join(args.logdir_root, 'wav')
	ops.init_directory(logdir_wav)
	##############################

	##############################
	# Get Data and Split them
	# Get list of data chunks
	chunk_list = build_db.find_files(npy_dir, pattern="*.npy")
	# To always have the same train/validate split, init the random seed
	random.seed(210691)
	random.shuffle(chunk_list)
	# Adapt batch_size if we have very few files
	num_chunk = len(chunk_list) 
	batch_size = min(args.batch_size, num_chunk)
	# Split 90 / 10
	training_chunks = chunk_list[:int(0.9*num_chunk)]
	valid_chunks = chunk_list[int(0.9*num_chunk):]
	##############################

	##############################	
	# Create network
	import time; ttt=time.time()
	net =  create_model(args)
	print("TTT: Instanciate network : {}".format(time.time()-ttt))
	##############################

	##############################	
	# Init optimizer and declare variable
	ttt = time.time()
	global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
	optim = optimizer_factory[args.optimizer](
		learning_rate=args.learning_rate,
		momentum=args.momentum)
	print("TTT: Initialize graph's variables : {}".format(time.time()-ttt))
	##############################

	##############################	
	# Placeholders
	ttt = time.time()
	train_input_batch_rnn_PH = tf.placeholder(tf.float32, shape=(None, seq_len_padded, 1), name="train_input_batch_rnn")
	generate_input_batch_rnn_PH = tf.placeholder(tf.float32, shape=(None, args.tiers[0], 1), name="generate_input_batch_rnn")
	big_frame_state_PH = []
	frame_state_PH = []
	for layer, rnn_dim in enumerate(rnns):
		big_frame_state_PH.append(tf.placeholder(tf.float32, shape=(None, rnn_dim), name="big_frame_state_" + str(layer)))
		frame_state_PH.append(tf.placeholder(tf.float32, shape=(None, rnn_dim), name="frame_state_" + str(layer)))
	##############################

	##############################
	# Compute losses
	loss_N, final_big_frame_state_N, final_frame_state_N, raw_output_N = net.loss_SampleRnn(
		train_input_batch_rnn_PH,
		big_frame_state_PH,
		frame_state_PH,
		seq_len_padded,
		l2_regularization_strength=args.l2_regularization_strength)

	grad_vars = optim.compute_gradients(loss_N, tf.trainable_variables())
	grads, vars = list(zip(*grad_vars))
	grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
	grad_vars = list(zip(grads_clipped, vars))

	for name in grad_vars:  
		print(name) 
	apply_gradient_op_N = optim.apply_gradients(grad_vars, global_step=global_step) 
	print("TTT: Create loss and grads nodes : {}".format(time.time()-ttt))
	##############################

	##############################
	# Generation network
	ttt = time.time()
	gen_input, gen_output = create_gen_wav_para(net)
	print("TTT: Instanciate generation net : {}".format(time.time()-ttt))
	##############################

	# Allocate only a fraction of GPU memory
	configSession = tf.ConfigProto()
	configSession.gpu_options.per_process_gpu_memory_fraction = 0.6

	with tf.Session(config=configSession) as sess:
		# Summary
		if args.summary:
			summary_weight_N = tf.summary.merge_all(key='weights')
			summary_preds_N = tf.summary.merge_all(key='pred_soft_max')
			summary_loss_N = tf.summary.merge_all(key='loss')
			writer = tf.summary.FileWriter(logdir_summary, sess.graph)

		# Keras
		K.set_session(sess)
		# Init weights
		init = tf.global_variables_initializer()
		sess.run(init)
		# Saver
		saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

		##############################
		# Load previously trained model
		ttt = time.time()
		if args.load_existing_model:
			try:
				saved_global_step = load(saver, sess, logdir_save)
				if saved_global_step is None:
					# The first training step will be saved_global_step + 1,
					# therefore we put -1 here for new or overwritten trainings.
					saved_global_step = -1
			except:
				print("Something went wrong while restoring checkpoint. "
						"We will terminate training to avoid accidentally overwriting "
						"the previous model.")
				raise
			print("TTT: Load previously trained model : {}".format(time.time()-ttt))
		else:
			# Remove existing models
			ops.init_directory(logdir_save)
			saved_global_step = -1
		##############################

		##############################
		# Infer some dimensions parameters
		step = None
		chunk_counter_train = 0
		length_generation = int(N_SECS*args.sample_rate)  # For generation
		audio_length = args.sample_size - int(args.tiers[0])
		bptt_length = args.seq_len
		stateful_rnn_length = audio_length//bptt_length 
		val_tab = np.zeros((args.num_steps))
		overfitting = False
		if args.summary:
			train_list=[summary_weight_N, summary_loss_N, summary_preds_N,\
				loss_N, apply_gradient_op_N, final_big_frame_state_N, final_frame_state_N, raw_output_N]
		else:
			train_list=[loss_N, apply_gradient_op_N, final_big_frame_state_N, final_frame_state_N, raw_output_N]
		valid_list=[loss_N, final_big_frame_state_N, final_frame_state_N]
		##############################
		try:
			for step in range(saved_global_step + 1, args.num_steps):
				if (step-1) % 20 == 0 and step>20:
					generate_and_save_samples(step, length_generation, args.sample_rate, args.num_example_generated, net, gen_input, gen_output, sess, logdir_wav)
				# Just to confirm that 0 is white noise. It is indeed
				# if step==0:
				# 	generate_and_save_samples(step, length_generation, args.sample_rate, args.num_example_generated, net, gen_input, gen_output, sess, logdir_wav)

				##############################
				# Initialize states, indices and losses
				final_big_s = []
				final_s = []
				for rnn_dim in rnns:
					final_big_s.append(np.zeros((batch_size, rnn_dim), dtype=np.float32))
					final_s.append(np.zeros((batch_size, rnn_dim), dtype=np.float32))
				loss_sum = 0
				idx_begin = 0
				last_saved_step = saved_global_step
				##############################
				
				start_time = time.time()
				
				##############################
				# Get train batch
				train_matrix, chunk_counter_train = load_mat(training_chunks, batch_size, chunk_counter_train)
				##############################

				for i in range(0, stateful_rnn_length):
					##############################
					# Write data in input ict
					inp_dict={}
					inp_dict[train_input_batch_rnn_PH] = train_matrix[:, idx_begin: idx_begin+seq_len_padded,:]
					for state_PH, state_value in zip(big_frame_state_PH, final_big_s):
						inp_dict[state_PH] = state_value
					for state_PH, state_value in zip(frame_state_PH, final_s):
						inp_dict[state_PH] = state_value
					inp_dict[K.learning_phase()] = 1
					idx_begin += args.seq_len
					##############################

					##############################
					# Run
					if args.summary:
						summary_weights, summary_loss, summary_preds, loss, _, final_big_s, final_s, raw_output = sess.run(train_list, feed_dict=inp_dict)
					else:
						loss, _, final_big_s, final_s, raw_output = sess.run(train_list, feed_dict=inp_dict)
					loss_sum += loss
					##############################

					##############################
					# Write summaries
					if args.summary:
						writer.add_summary(summary_loss, step)
						writer.add_summary(summary_preds, step)
					loss_norm = loss_sum / stateful_rnn_length
					##############################


				##############################
				# Get valid batch
				# For validation we can make one huge batch
				valid_matrix, _ = load_mat(valid_chunks, len(valid_chunks), 0)
				final_big_s_val = []
				final_s_val = []
				for rnn_dim in rnns:
					final_big_s_val.append(np.zeros((len(valid_chunks), rnn_dim), dtype=np.float32))
					final_s_val.append(np.zeros((len(valid_chunks), rnn_dim), dtype=np.float32))
				idx_begin = 0
				##############################

				loss_val_sum = 0
				for i in range(0, stateful_rnn_length):
					valid_dict={}
					valid_dict[train_input_batch_rnn_PH] = valid_matrix[:, idx_begin: idx_begin+seq_len_padded,:]
					for state_PH, state_value in zip(big_frame_state_PH, final_big_s_val):
						valid_dict[state_PH] = state_value
					for state_PH, state_value in zip(frame_state_PH, final_s_val):
						valid_dict[state_PH] = state_value
					valid_dict[K.learning_phase()] = 0
					idx_begin += args.seq_len

					loss_val, final_big_s_val, final_s_val = sess.run(valid_list, feed_dict=valid_dict)
					loss_val_sum += loss_val

				loss_val_mean = loss_val_sum / stateful_rnn_length

				val_tab[step] = loss_val_mean
				if step > 10:
					# Perform at least 10 epoch
					overfitting = early_stopping.up_criterion(val_tab, step, 3, 2)

				if args.summary:
					writer.add_summary(summary_weights, step)
				duration = time.time() - start_time
				print('step {:d} - loss = {:.5f} - val : {:.5f}'
							.format(step, loss_norm, loss_val_mean))

				if step % args.checkpoint_every == 0:
					save(saver, sess, logdir_save, step)
					last_saved_step = step

				##############################
				##############################
				##############################
				##############################
				##############################
				overfitting = (step > 1)
				##############################
				##############################
				##############################
				##############################
				##############################

				if overfitting:
					break

		except KeyboardInterrupt:
			# Introduce a line break after ^C is displayed so save message
			# is on its own line.
			print()
		finally:
			generate_and_save_samples(step, length_generation, args.sample_rate, args.num_example_generated, net, gen_input, gen_output, sess, logdir_wav)
			np.save(os.path.join(args.logdir_root, 'validation_loss.npy'), val_tab[:step])
			if step > last_saved_step:
				save(saver, sess, logdir_save, step)	

if __name__ == '__main__':
		main()
