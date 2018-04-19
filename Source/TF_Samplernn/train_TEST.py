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

from asynchronous_load_mat import load_mat

# DATA_DIRECTORY='/home/leo/WaveGeneration/Data/contrabass_no_cond/ordinario/8000_4104_0.01'
DATA_DIRECTORY="/Users/leo/Recherche/WaveGeneration/Data/contrabass_no_cond/ordinario_xs/8000_16392_0.01"
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
NUM_EXAMPLE_GENERATED=10

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
	parser.add_argument('--optimizer',        type=str,   default=OPTIMIZER, choices=list(optimizer_factory.keys()))
	parser.add_argument('--momentum',         type=float, default=MOMENTUM)

	parser.add_argument('--seq_len',          type=int, default=SEQ_LEN)
	parser.add_argument('--big_frame_size',   type=int, default=BIG_FRAME_SIZE)
	parser.add_argument('--frame_size',       type=int, default=FRAME_SIZE)
	parser.add_argument('--q_levels',         type=int, default=Q_LEVELS)
	parser.add_argument('--dim',              type=int, default=DIM)
	parser.add_argument('--n_rnn',            type=int, choices=list(range(1,6)), default=N_RNN)
	parser.add_argument('--emb_size',         type=int, default=EMB_SIZE)
	parser.add_argument('--autoregressive_order', type=int, default=AUTOREGRESSIVE_ORDER)
	parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], default=RNN_TYPE)
	parser.add_argument('--max_checkpoints',  type=int, default=MAX_TO_KEEP)
	parser.add_argument('--load_existing_model',  type=int, default=False)
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
		emb_size=args.emb_size,
		autoregressive_order=args.autoregressive_order)
	return net 

# GENERATE 
def create_gen_wav_para(net):
	with tf.name_scope('generation'):
		gen_input = {}
		gen_output = {}

		gen_input['big_frame'] = tf.placeholder(tf.float32, shape=(None, net.big_frame_size, 1), name="big_frame")
		gen_input['big_frame_state'] = tf.placeholder(tf.float32, shape=(None, net.dim), name="big_frame_state")

		gen_input['frame_from_big'] = tf.placeholder(tf.float32, shape=(None, 1, net.dim), name='frame_from_big')
		gen_input['frame'] = tf.placeholder(tf.float32, (None, net.frame_size, 1), name="frame")
		gen_input['frame_state'] = tf.placeholder(tf.float32, (None, net.dim), name="frame_state")

		gen_input['sample_from_frame'] = tf.placeholder(tf.float32, shape=(None, 1, net.dim), name='frame_from_big')
		gen_input['sample'] = tf.placeholder(tf.int32, (None, net.autoregressive_order, 1), name="frame")

		gen_output['big_frame'], gen_output['big_frame_state'] = net._create_network_BigFrame(big_frame_state=gen_input['big_frame_state'],
			big_input_sequences=gen_input['big_frame'],
			seq_len=None)

		gen_output['frame'], gen_output['frame_state'] = net._create_network_Frame(
			big_frame_outputs=gen_input['frame_from_big'],
			frame_state=gen_input['frame_state'],
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

def generate_and_save_samples(step, length, num_example_generated, net, gen_input, gen_output, sess):
	# Initialize sequence to generate
	samples = np.zeros((num_example_generated, length, 1), dtype='int32')
	samples[:, :net.big_frame_size,:] = np.int32(net.q_levels//2)

	# Initialize rnn_states
	final_big_s = np.zeros((num_example_generated, net.dim), dtype=np.float32)
	final_s = np.zeros((num_example_generated, net.dim), dtype=np.float32)
	
	# Output of different levels of RNN
	big_frame_out = None
	frame_out = None
	sample_out = None
	
	for t in range(net.big_frame_size, length):
		#big frame 
		if t % net.big_frame_size == 0:
			big_frame_out = None
			big_input_sequences = samples[:, t-net.big_frame_size:t,:].astype('float32')
			big_frame_out, final_big_s= \
			sess.run([gen_output['big_frame'] , gen_output['big_frame_state'] ],
				feed_dict={gen_input['big_frame'] : big_input_sequences,
					gen_input['big_frame_state'] : final_big_s})
		#frame 
		if t % net.frame_size == 0:
			frame_input_sequences = samples[:, t-net.frame_size:t,:].astype('float32')
			big_frame_output_idx = (t/net.frame_size)%(net.big_frame_size/net.frame_size)
			frame_out, final_s = sess.run([gen_output['frame'], gen_output['frame_state']],
				feed_dict={gen_input['frame_from_big'] : big_frame_out[:,[big_frame_output_idx],:],
					gen_input['frame'] : frame_input_sequences,
					gen_input['frame_state'] : final_s})
		
		#sample
		sample_input_sequences = samples[:, t-net.autoregressive_order:t,:]
		frame_output_idx = t%net.frame_size
		sample_out= sess.run(gen_output['sample'],
			feed_dict={gen_input['sample_from_frame'] : frame_out[:,[frame_output_idx],:],
				gen_input['sample'] : sample_input_sequences})

		# Sample from the softmax distribution sample_out
		sample_next_list = []
		for row in sample_out:
			sample_next = np.random.choice(np.arange(net.q_levels), p=row)
			sample_next_list.append(sample_next)
		samples[:, t] = np.array(sample_next_list).reshape([-1,1])

	# Decode mu_law
	for i in range(0, num_example_generated):
		inp = samples[i].reshape([-1,1]).tolist()
		out = sess.run(gen_output['sample_decoded'], feed_dict={gen_input['sample_to_decode']: inp})
		write_wav(out, 16000, 'test_wav/' + str(step)+'_'+str(i)+'.wav')
	return
					
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
	# Placeholders
	ttt = time.time()
	train_input_batch_rnn_PH = tf.placeholder(tf.float32, shape=(None, args.seq_len, 1), name="train_input_batch_rnn")
	generate_input_batch_rnn_PH = tf.placeholder(tf.float32, shape=(None, args.big_frame_size, 1), name="generate_input_batch_rnn")
	big_frame_state_PH = tf.placeholder(tf.float32, shape=(None, args.dim), name="big_frame_state")
	frame_state_PH = tf.placeholder(tf.float32, shape=(None, args.dim), name="frame_state")
	##############################

	##############################
	# Compute losses
	loss_N, final_big_frame_state_N, final_frame_state_N = net.loss_SampleRnn(
		train_input_batch_rnn_PH,
		big_frame_state_PH,
		frame_state_PH,
		args.seq_len,
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

	
	writer = tf.summary.FileWriter(logdir)
	writer.add_graph(tf.get_default_graph())

	summaries_N = tf.summary.merge_all()

	# Allocate only a fraction of GPU memory
	configSession = tf.ConfigProto()
	configSession.gpu_options.per_process_gpu_memory_fraction = 0.5

	with tf.Session(config=configSession) as sess:
		
		init = tf.global_variables_initializer()
		sess.run(init)

		saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

		##############################
		# Load previously trained model
		ttt = time.time()
		if args.load_existing_model:
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
		else:
			# Remove existing models
			shutil.rmtree(logdir)
			os.makedirs(logdir)
			saved_global_step = -1
		##############################

		step = None
		chunk_counter = 0
		last_saved_step = saved_global_step
		length_generation = int(N_SECS*params_data["sample_rate"])  # For generation
		try:
			for step in range(saved_global_step + 1, args.num_steps):
				if (step-1) % 20 == 0 and step>20:
					generate_and_save_samples(step, length_generation, NUM_EXAMPLE_GENERATED, net, gen_input, gen_output, sess)

				##############################
				# Initialize the stateful RNN
				final_big_s = np.zeros((args.batch_size, args.dim), dtype=np.float32)
				final_s = np.zeros((args.batch_size, args.dim), dtype=np.float32)
				##############################
				
				start_time = time.time()
				
				##############################
				# Get train batch
				train_matrix, chunk_counter = load_mat(chunk_list, args.batch_size, chunk_counter)
				# mean_abs = np.mean(np.absolute(train_matrix))
				# print(mean_abs)
				##############################

				##############################
				# Infer some dimensions parameters
				loss_sum = 0
				idx_begin = 0
				audio_length = params_data["sample_size"] - args.big_frame_size
				bptt_length = args.seq_len - args.big_frame_size
				stateful_rnn_length = audio_length//bptt_length 
				outp_list=[summaries_N,\
					loss_N, \
				 	apply_gradient_op_N, \
				 	final_big_frame_state_N, \
				 	final_frame_state_N]
				##############################

				for i in range(0, stateful_rnn_length):
					##############################
					# Write data in input ict
					inp_dict={}
					inp_dict[train_input_batch_rnn_PH] = train_matrix[:, idx_begin: idx_begin+args.seq_len,:]
					inp_dict[big_frame_state_PH] = final_big_s
					inp_dict[frame_state_PH] = final_s
					idx_begin += args.seq_len-args.big_frame_size
					##############################

					##############################
					# Run
					summary, loss, _, final_big_s, final_s= sess.run(outp_list, feed_dict=inp_dict)
					##############################

					##############################
					# Write summaries
					writer.add_summary(summary, step)
					loss_norm = loss/stateful_rnn_length
					##############################
				duration = time.time() - start_time
				print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
							.format(step, loss_norm, duration))

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
