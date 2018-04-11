from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import random
import sys
import time
import cPickle as pkl

import librosa
import numpy as np
import tensorflow as tf
import build_db
from tensorflow.python.client import timeline
from samplernn import SampleRnnModel, AudioReader, mu_law_decode, optimizer_factory

from asynchronous_load_mat import load_mat

DATA_DIRECTORY='/home/aciditeam-leo/Aciditeam/WaveGeneration/Data/contrabass_no_cond/ordinario/8000_4104_0.01'
LOGDIR_ROOT='./logdir'
CHECKPOINT_EVERY=10
NUM_STEPS=int(1e5)
LEARNING_RATE=1e-3
L2_REGULARIZATION_STRENGTH=0
MOMENTUM=0.9
MAX_TO_KEEP=5

BIG_FRAME_SIZE=8
FRAME_SIZE=2        
Q_LEVELS=256        # Quantification for the amplitude of the audio samples
RNN_TYPE='GRU'
DIM=1024            # Number of units in RNNs
N_RNN=1
SEQ_LEN=520         # Size for one BPTT pass
EMB_SIZE=256
AUTOREGRESSIVE_ORDER=2
OPTIMIZER='adam'

N_SECS=3

BATCH_SIZE=64
NUM_GPU=1

def get_arguments():
	parser = argparse.ArgumentParser(description='SampleRnn example network')
	parser.add_argument('--num_gpus',         type=int,   default=NUM_GPU)
	parser.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
	parser.add_argument('--data_dir',         type=str,   default=DATA_DIRECTORY)
	parser.add_argument('--logdir_root',      type=str,   default=LOGDIR_ROOT)
	parser.add_argument('--checkpoint_every', type=int,   default=CHECKPOINT_EVERY)
	parser.add_argument('--num_steps',        type=int,   default=NUM_STEPS)
	parser.add_argument('--learning_rate',    type=float, default=LEARNING_RATE)
	parser.add_argument('--l2_regularization_strength', type=float, default=L2_REGULARIZATION_STRENGTH)	
	parser.add_argument('--optimizer',        type=str,   default=OPTIMIZER, choices=optimizer_factory.keys())
	parser.add_argument('--momentum',         type=float, default=MOMENTUM)

	parser.add_argument('--seq_len',          type=int, default=SEQ_LEN)
	parser.add_argument('--big_frame_size',   type=int, default=BIG_FRAME_SIZE)
	parser.add_argument('--frame_size',       type=int, default=FRAME_SIZE)
	parser.add_argument('--q_levels',         type=int, default=Q_LEVELS)
	parser.add_argument('--dim',              type=int, default=DIM)
	parser.add_argument('--n_rnn',            type=int, choices=xrange(1,6), default=N_RNN)
	parser.add_argument('--emb_size',         type=int, default=EMB_SIZE)
	parser.add_argument('--autoregressive_order', type=int, default=AUTOREGRESSIVE_ORDER)
	parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], default=RNN_TYPE)
	parser.add_argument('--max_checkpoints',  type=int, default=MAX_TO_KEEP)
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
	# Create network.
	net = SampleRnnModel(
		batch_size=args.batch_size,
		big_frame_size=args.big_frame_size,
		frame_size=args.frame_size,
		q_levels=args.q_levels,
		rnn_type=args.rnn_type,
		dim=args.dim,
		n_rnn=args.n_rnn,
		seq_len=args.seq_len,
		emb_size=args.emb_size,
		autoregressive_order=args.autoregressive_order)
	return net 

#>>
# GENERATE 
# def create_gen_wav_para(net):
# 	with tf.name_scope('infe_para'):
# 		infe_para = dict()
# 		infe_para['infe_big_frame_inp'] = \
# 			tf.get_variable("infe_big_frame_inp", 
# 				[net.batch_size, net.big_frame_size,1], dtype=tf.float32)
# 		infe_para['infe_big_frame_outp'] = \
# 			tf.get_variable("infe_big_frame_outp", 
# 				[net.batch_size, net.big_frame_size/net.frame_size, net.dim], dtype=tf.float32)

# 		infe_para['infe_big_frame_outp_slices'] = \
# 			tf.get_variable("infe_big_frame_outp_slices", 
# 				[net.batch_size, 1, net.dim], dtype=tf.float32)
# 		infe_para['infe_frame_inp'] = \
# 			tf.get_variable("infe_frame_inp", 
# 				[net.batch_size, net.frame_size,1], dtype=tf.float32)
# 		infe_para['infe_frame_outp'] = \
# 			tf.get_variable("infe_frame_outp", 
# 				[net.batch_size, net.frame_size, net.dim], dtype=tf.float32)

# 		infe_para['infe_frame_outp_slices'] = \
# 			tf.get_variable("infe_frame_outp_slices", 
# 				[net.batch_size, 1, net.dim], dtype=tf.float32)
# 		infe_para['infe_sample_inp'] = \
# 			tf.get_variable("infe_sample_inp", 
# 				[net.batch_size, net.autoregressive_order,1], dtype=tf.int32)

# 		infe_para['infe_big_frame_state'] = net.big_cell.zero_state(net.batch_size, tf.float32)
# 		infe_para['infe_frame_state']     = net.cell.zero_state(net.batch_size, tf.float32)

# 		tf.get_variable_scope().reuse_variables()
# 		infe_para['infe_big_frame_outp'], \
# 		infe_para['infe_final_big_frame_state'] = \
# 				net._create_network_BigFrame(num_steps = 1,
# 					big_frame_state = infe_para['infe_big_frame_state'],
# 					big_input_sequences = infe_para['infe_big_frame_inp'])

# 		infe_para['infe_frame_outp'], \
# 		infe_para['infe_final_frame_state'] = \
# 				net._create_network_Frame(num_steps = 1,
# 					big_frame_outputs = infe_para['infe_big_frame_outp_slices'],
# 					frame_state = infe_para['infe_frame_state'],
# 					input_sequences = infe_para['infe_frame_inp'])

# 		sample_out = \
# 			net._create_network_Sample(frame_outputs=infe_para['infe_frame_outp_slices'],
# 				sample_input_sequences = infe_para['infe_sample_inp'])

# 		sample_out = \
# 			tf.reshape(sample_out, [-1, net.q_levels])
# 		infe_para['infe_sample_outp'] = tf.cast(
# 			tf.nn.softmax(tf.cast(sample_out, tf.float64)), tf.float32)

# 		infe_para['infe_sample_decode_inp'] = \
# 			tf.placeholder(tf.int32)
# 		infe_para['infe_decode'] = \
# 			mu_law_decode(infe_para['infe_sample_decode_inp'], net.q_levels)

# 		return infe_para
#>>

def write_wav(waveform, sample_rate, filename):
	y = np.array(waveform)
	librosa.output.write_wav(filename, y, sample_rate)
	print('Updated wav file at {}'.format(filename))

def generate_and_save_samples(step, length, net, infe_para, sess):
	samples = np.zeros((net.batch_size, length, 1), dtype='int32')
	samples[:, :net.big_frame_size,:] = np.int32(net.q_levels//2)

	final_big_s,final_s = sess.run([net.big_initial_state,net.initial_state])
	big_frame_out = None
	frame_out = None
	sample_out = None
	for t in xrange(net.big_frame_size, length):
		#big frame 
		if t % net.big_frame_size == 0:
			big_frame_out = None
			big_input_sequences = samples[:, t-net.big_frame_size:t,:].astype('float32')
			big_frame_out, final_big_s= \
			sess.run([infe_para['infe_big_frame_outp'] , 
		infe_para['infe_final_big_frame_state'] ],
							 feed_dict={
									infe_para['infe_big_frame_inp'] : big_input_sequences,
									infe_para['infe_big_frame_state'] : final_big_s})
		#frame 
		if t % net.frame_size == 0:
			frame_input_sequences = samples[:, t-net.frame_size:t,:].astype('float32')
			big_frame_output_idx = (t/net.frame_size)%(net.big_frame_size/net.frame_size)
			frame_out, final_s= \
			sess.run([infe_para['infe_frame_outp'], 
		infe_para['infe_final_frame_state']],
							feed_dict={
	infe_para['infe_big_frame_outp_slices'] : big_frame_out[:,[big_frame_output_idx],:],
	infe_para['infe_frame_inp'] : frame_input_sequences,
	infe_para['infe_frame_state'] : final_s})
		#sample
		sample_input_sequences = samples[:, t-net.frame_size:t,:]
		frame_output_idx = t%net.frame_size
		sample_out= \
		sess.run(infe_para['infe_sample_outp'],
						 feed_dict={
								infe_para['infe_frame_outp_slices'] : frame_out[:,[frame_output_idx],:],
								infe_para['infe_sample_inp'] : sample_input_sequences})
		sample_next_list = []
		for row in sample_out:
			sample_next = np.random.choice(
					np.arange(net.q_levels), p=row )
			sample_next_list.append(sample_next)
		samples[:, t] = np.array(sample_next_list).reshape([-1,1])
	for i in range(0, net.batch_size):
		inp = samples[i].reshape([-1,1]).tolist()
		out = sess.run(infe_para['infe_decode'], 
		feed_dict={infe_para['infe_sample_decode_inp']: inp})
		write_wav(out, 16000, './test_wav/'+ identifier + '/' + str(step)+'_'+str(i)+'.wav')
		if i >= 10:
			break
			
def main():
	args = get_arguments()
	if args.l2_regularization_strength == 0:
			args.l2_regularization_strength = None
	params_data=pkl.load(open(args.data_dir + '/params.pkl', 'rb'))

	# identifier = str(random.randint(1,100000))
	identifier = "0"
	print("Identifier: {}".format(identifier))
	logdir = os.path.join(args.logdir_root, identifier)

	##############################
	# Get list of data chunks
	chunk_list = build_db.find_files(DATA_DIRECTORY + '/chunk', pattern="*.npy")
	random.shuffle(chunk_list)
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
	# Instanciate input nodes and get intermediate nodes for stateful RNNs
	ttt = time.time()
	train_input_batch_rnn = tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]), 
										trainable=False ,name="input_batch_rnn", dtype=tf.float32)
	train_big_frame_state = net.big_cell.zero_state(net.batch_size, tf.float32)
	final_big_frame_state = net.big_cell.zero_state(net.batch_size, tf.float32)
	train_frame_state = net.cell.zero_state(net.batch_size, tf.float32)
	final_frame_state = net.cell.zero_state(net.batch_size, tf.float32)
	##############################

	##############################
	# Compute losses
	loss,\
	final_big_frame_state,\
	final_frame_state = net.loss_SampleRnn(
		train_input_batch_rnn,
		train_big_frame_state,
		train_frame_state,
		l2_regularization_strength=args.l2_regularization_strength)

	apply_gradients_op = optim.minimize(loss)
	# grad_vars = optim.compute_gradients(loss, tf.trainable_variables())
	# grads, vars = zip(*grad_vars)
	# grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
	# grad_vars = zip(grads_clipped, vars)

	# for name in grad_vars:  
	# 	print(name) 
	# apply_gradient_op = optim.apply_gradients(grad_vars, global_step=global_step) 
	print("TTT: Create loss and grads nodes : {}".format(time.time()-ttt))
	##############################

	##############################
	# Generation network
	ttt = time.time()
	#>>
	# infe_para = create_gen_wav_para(net)
	#>>
	print("TTT: Instanciate generation net : {}".format(time.time()-ttt))
	##############################

	##############################
	# Various shit
	ttt = time.time()
	writer = tf.summary.FileWriter(logdir)
	writer.add_graph(tf.get_default_graph())

	summaries = tf.summary.merge_all()
	import pdb; pdb.set_trace()

	with tf.Session() as sess:
		
		init = tf.global_variables_initializer()
		sess.run(init)

		saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

		##############################
		# Load previously trained model
		ttt = time.time()
		try:
			saved_global_step = load(saver, sess, logdir)
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
		##############################

		step = None
		chunk_counter = 0
		last_saved_step = saved_global_step
		length_generation = N_SECS*params_data["sample_rate"]  # For generation
		try:
			for step in range(saved_global_step + 1, args.num_steps):
				#>>
				# if (step-1) % 20 == 0 and step>20:
				# 	generate_and_save_samples(step, length_generation, net, infe_para, sess)
				#>>

				##############################
				# Initialize the stateful RNN
				import time; ttt=time.time()
				final_big_s = sess.run(net.big_initial_state)
				final_s = sess.run(net.initial_state)
				print("TTT: Initialize the stateful RNN : {}".format(time.time()-ttt))
				##############################
				
				start_time = time.time()
				
				##############################
				# Read data from GPU to CPU... WTF ???
				import time; ttt=time.time()
				#### Here asynchronously load (batch, dimen, 1) matrices
				train_matrix, chunk_counter = load_mat(chunk_list, net.batch_size, chunk_counter)
				print("TTT: Load training data : {}".format(time.time()-ttt))
				##############################

				##############################
				# Infer some dimensions parameters
				import time; ttt=time.time()
				loss_sum = 0
				idx_begin = 0
				audio_length = params_data["sample_size"] - args.big_frame_size
				bptt_length = args.seq_len - args.big_frame_size
				stateful_rnn_length = audio_length/bptt_length 
				outp_list=[summaries,\
					loss, \
				 	apply_gradient_op, \
				 	final_big_frame_state, \
				 	final_frame_state]
				print("TTT: Infer some dimensions parameters : {}".format(time.time()-ttt))
				##############################

				for i in range(0, stateful_rnn_length):
					##############################
					# Write data in input ict
					import time; ttt=time.time()
					inp_dict={}
					inp_dict[train_input_batch_rnn] = train_matrix[:, idx_begin: idx_begin+args.seq_len,:]
					inp_dict[train_big_frame_state] = final_big_s
					inp_dict[train_frame_state] = final_s
					idx_begin += args.seq_len-args.big_frame_size
					print("TTT: Write data in input dict : {}".format(time.time()-ttt))
					##############################

					##############################
					# Run
					import time; ttt=time.time()
					summary, loss_computed, _, final_big_s, final_s= \
						sess.run(outp_list, feed_dict=inp_dict)
					print("Run : {}".format(time.time() - ttt))
					##############################

					##############################
					# Write summaries
					import time; ttt=time.time()
					writer.add_summary(summary, step)
					loss_computed_norm = loss_computed/stateful_rnn_length
					import pdb; pdb.set_trace()
					print("TTT: Write summaries : {}".format(time.time()-ttt))
					##############################
				duration = time.time() - start_time
				print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
							.format(step, loss_computed_norm, duration))

				if step % args.checkpoint_every == 0:
					save(saver, sess, logdir, step)
					last_saved_step = step

		except KeyboardInterrupt:
			# Introduce a line break after ^C is displayed so save message
			# is on its own line.
			print()
		finally:
			if step > last_saved_step:
					save(saver, sess, logdir, step)

if __name__ == '__main__':
		main()
