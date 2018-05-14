import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from .ops import mu_law_encode

import keras
from keras.layers.recurrent import GRU
from keras.layers import Dense, Dropout

from samplernn.weight_summary import keras_layer_summary, variable_summary

class SampleRnnModel(object):
	def __init__(self, tiers, rnns, mlps, q_levels, emb_size, dropout, summary):
		self.big_frame_size=tiers[0]
		self.frame_size=tiers[1]
		self.autoregressive_order=tiers[2]

		self.rnns=rnns
		self.mlps=mlps

		self.q_levels=q_levels
		self.emb_size=emb_size

		self.dropout=dropout
		self.summarize=summary

		self._init_weigths()

		return

	def _init_weigths(self):
		self.weights={}

		# Big Frame level
		self.big_upsampling_ratio = self.big_frame_size//self.frame_size 
		self.big_project_dim = self.rnns[0] * self.big_upsampling_ratio
		self.weights["big_frame_rnn"] = []
		for num_unit in self.rnns:
			 self.weights["big_frame_rnn"].append(GRU(num_unit, return_sequences=True, return_state=True, activation='relu', dropout=self.dropout))
		self.weights["big_frame_proj_weights"] = tf.get_variable("big_frame_proj_weights", [self.rnns[-1], self.big_project_dim], dtype=tf.float32)

		# Frame level
		self.fr_upsampling_ratio = self.frame_size
		self.fr_project_dim = self.mlps[0] * self.fr_upsampling_ratio
		self.weights["frame_rnn"]=[]
		for num_unit in self.rnns:
			 self.weights["frame_rnn"].append(GRU(num_unit, return_sequences=True, return_state=True, activation='relu', dropout=self.dropout))
		self.weights["frame_proj_weights"] = tf.get_variable("frame_proj_weights", [self.rnns[-1], self.fr_project_dim], dtype=tf.float32)
		self.weights["frame_cell_proj_weights"] = tf.get_variable("frame_cell_proj_weights", [self.frame_size, self.rnns[0]], dtype=tf.float32)

		# Sample level
		self.weights["embedding"] = tf.get_variable("embedding", [self.q_levels, self.emb_size])
		'''Create a convolution filter variable with the specified name and shape,
		and initialize it using Xavier initialition.'''
		filter_initializer = tf.contrib.layers.xavier_initializer_conv2d()
		# conv_shape : (kernel_size, channel, output_dim)
		sample_filter_shape = [self.emb_size*self.autoregressive_order, 1, self.mlps[0]]  # self.emb_size*self.autoregressive_order : le noyau de convolution couvre un horizon de deux samples. 
		# Cf l'argument stride dans la fonction conv1D plus bas : on fait des pas de taille emb_size a chaque fois
		self.weights["sample_filter"] = tf.get_variable("sample_filter", sample_filter_shape, initializer=filter_initializer)
		self.weights["sample_mlp"]=[]
		for layer_ind in range(len(self.mlps)-1):
			 self.weights["sample_mlp"].append(tf.get_variable("sample_mlp_" + str(layer_ind), [self.mlps[layer_ind], self.mlps[layer_ind+1]], dtype=tf.float32))
		self.weights["sample_mlp"].append(tf.get_variable("sample_mlp_" + str(len(self.mlps)), [self.mlps[-1], self.q_levels], dtype=tf.float32))

		# Summarize weights (keras layers are initialized after input are fed)
		if self.summarize:
			for weight_name, weight_value in self.weights.items():
				if type(weight_value).__name__ == 'list':
					for weight_val in weight_value:
						if type(weight_val).__name__ == 'Variable':
							variable_summary(weight_val, plot_bool=False, collections=["weights"])
				else:
					if type(weight_value).__name__ == 'Variable':
							variable_summary(weight_value, plot_bool=False, collections=["weights"])
		return


	def _preprocess_audio_inputs(self, input_frames):
		# Rescale input from [0, q_level] to [-2, 2]
		input_frames = (input_frames / (self.q_levels/2.0)) - 1.0
		input_frames *= 2.0
		return input_frames

	def _upsampling_reshape(self, x, shape_1, shape_2):
		"""Take as input a list of upsampled tensor and reshape them to fit witht the frame_size of the next level
		"""
		x = tf.reshape(x, [-1, shape_1, shape_2])
		return x

	def _create_network_BigFrame(self,
			big_frame_states,
			big_input_sequences,
			seq_len):
		if seq_len is None:
			num_time_frames = 1
		else:
			num_time_frames = (seq_len-self.big_frame_size) // self.big_frame_size
		
		with tf.variable_scope('BigFrame_layer'):

			with tf.variable_scope('Preprocess_audio_inputs'):
				big_input_frames = tf.reshape(big_input_sequences, [-1, num_time_frames, self.big_frame_size])
				big_input_frames = self._preprocess_audio_inputs(big_input_frames)
				# Note : self.big_frame_size/self.frame_size est le ratio d'upsampling
			
			with tf.variable_scope("BIG_FRAME_RNN"):
				next_states = []
				rnn_in = big_input_frames
				for rnn_state, rnn in zip(big_frame_states, self.weights["big_frame_rnn"]):
					output, state = rnn(rnn_in, initial_state=rnn_state)
					rnn_in = output
					next_states.append(state)
					keras_layer_summary(rnn, plot_bool=False, collections=["weights"])

			with tf.variable_scope("Projection_to_frame_level"):
				output = tf.reshape(output, [-1, self.rnns[-1]])
				big_frame_outputs = math_ops.matmul(output, self.weights["big_frame_proj_weights"])
				big_frame_outputs = tf.reshape(big_frame_outputs, [-1, num_time_frames, self.big_project_dim])

				with tf.variable_scope("Reshape_upsampled"):
					big_frame_outputs = self._upsampling_reshape(big_frame_outputs, num_time_frames * self.big_upsampling_ratio, self.rnns[0])

		return big_frame_outputs, next_states

	def _create_network_Frame(self,
			big_frame_outputs,
			frame_states,
			input_sequences,
			seq_len):
		if seq_len is None:
			num_time_frames = 1
		else:
			num_time_frames = (seq_len-self.big_frame_size) // self.frame_size

		with tf.variable_scope('Frame_layer'):
			with tf.variable_scope('Preprocess_audio_inputs'):
				input_frames = tf.reshape(input_sequences,[-1, num_time_frames, self.frame_size])
				input_frames = self._preprocess_audio_inputs(input_frames)

			with tf.variable_scope("FRAME_RNN"):
				cell_input = tf.reshape(input_frames, [-1, self.frame_size])
				cell_input = math_ops.matmul(cell_input, self.weights["frame_cell_proj_weights"])
				cell_input = tf.reshape(cell_input, [-1, num_time_frames, self.rnns[0]])
				cell_input += big_frame_outputs
				# Rnn
				next_states = []
				rnn_in = cell_input
				for rnn_state, rnn in zip(frame_states, self.weights["frame_rnn"]):
					output, state = rnn(rnn_in, initial_state=rnn_state)
					rnn_in = output
					next_states.append(state)
					keras_layer_summary(rnn, plot_bool=False, collections=["weights"])

			with tf.variable_scope("Projection_to_sample_level"):
				output = tf.reshape(output, [-1, self.rnns[-1]])
				frame_outputs = math_ops.matmul(output, self.weights["frame_proj_weights"])
				frame_outputs = tf.reshape(frame_outputs, [-1, num_time_frames, self.fr_project_dim])
				with tf.variable_scope("Reshape_upsampled"):
					frame_outputs = self._upsampling_reshape(frame_outputs, num_time_frames * self.fr_upsampling_ratio, self.mlps[0])   # Actually self.frame_size / 1 as the upsampling ratio of the last level is 1
					# frame_outputs = self._upsampling_reshape(frame_outputs, num_time_frames * upsampling_ratio)   # Actually self.frame_size / 1 as the upsampling ratio of the last level is 1

		return frame_outputs, next_states

	def _create_network_Sample(self,
		frame_outputs,
		sample_input_sequences,
		seq_len):

		if seq_len is None:
			num_time_frames = self.autoregressive_order
		else:
			num_time_frames = (seq_len-(self.big_frame_size-self.autoregressive_order+1))

		with tf.variable_scope('Sample_layer'):

			sample_shap=[-1,
				num_time_frames*self.emb_size,
				1]

			# Embedding
			sample_input_sequences = embedding_ops.embedding_lookup(self.weights["embedding"], tf.reshape(sample_input_sequences,[-1]))
			# Ici les embedding de chaque frames sont mises les uns a cote des autres (a l'interieur d'un meme batch)
			sample_input_sequences = tf.reshape(sample_input_sequences, sample_shap)

			out = tf.nn.conv1d(sample_input_sequences,
				self.weights["sample_filter"],
				stride=self.emb_size,
				padding="VALID", 
				name="sample_conv")
			out = out + frame_outputs

			# MLP
			out = tf.reshape(out, [-1, self.mlps[0]])
			for mlp_weight in self.weights["sample_mlp"]:
				out = math_ops.matmul(out, mlp_weight)
				out = tf.nn.relu(out)
			out = tf.reshape(out, [-1, num_time_frames-self.autoregressive_order+1, self.q_levels])
			
		return out

	def _create_network_SampleRnn(self,
			train_big_frame_states,
			train_frame_states,
			seq_len):
		with tf.name_scope('SampleRnn_net'):
			#big frame 
			big_input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)[:,:-self.big_frame_size,:]
			big_frame_outputs, final_big_frame_state = self._create_network_BigFrame(big_frame_states=train_big_frame_states, big_input_sequences=big_input_sequences, seq_len=seq_len)

			#frame 
			input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)[:, self.big_frame_size-self.frame_size:-self.frame_size, :]
			frame_num_steps = (seq_len-self.big_frame_size)/self.frame_size
			frame_outputs, final_frame_state  = self._create_network_Frame(big_frame_outputs=big_frame_outputs, frame_states=train_frame_states, input_sequences=input_sequences, seq_len=seq_len)

			#sample
			sample_input_sequences = self.encoded_input_rnn[:, self.big_frame_size-self.autoregressive_order:-1, :]
			sample_output = self._create_network_Sample(frame_outputs, sample_input_sequences=sample_input_sequences, seq_len=seq_len)

		return sample_output, final_big_frame_state, final_frame_state

	def loss_SampleRnn(self,
		train_input_batch_rnn,
		train_big_frame_state,
		train_frame_state,
		seq_len,
		l2_regularization_strength=None,
		name='sample'):
		
		with tf.name_scope(name):
			# Process input
			self.encoded_input_rnn = mu_law_encode(train_input_batch_rnn, self.q_levels)
		
			# Train
			raw_output, final_big_frame_state, final_frame_state = self._create_network_SampleRnn(train_big_frame_state, train_frame_state, seq_len)

			if self.summarize:
				# Summarize prediction
				preds_summary = tf.reshape(raw_output, [-1, seq_len-self.big_frame_size, self.q_levels, 1])
				tf.summary.image("pred_soft_max", preds_summary, 10, collections=["pred_soft_max"])
				inp_summary = tf.reshape(raw_output, [-1, seq_len-self.big_frame_size, self.q_levels, 1])
				tf.summary.image("input_batch", inp_summary, 10, collections=["pred_soft_max"])

			with tf.name_scope('loss'):
				# Target
				target = tf.reshape(self.encoded_input_rnn[:, self.big_frame_size:], [-1])
				# Prediction
				prediction = tf.reshape(raw_output, [-1, self.q_levels])
				# Loss
				loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=target)
				reduced_loss = tf.reduce_mean(loss)
				tf.summary.scalar('loss', reduced_loss, collections=["loss"])
				if l2_regularization_strength is None:
					return reduced_loss , final_big_frame_state, final_frame_state
				else:
					# L2 regularization for all trainable parameters
					l2_loss = tf.add_n([tf.nn.l2_loss(v)
								  for v in tf.trainable_variables()
								  if not('bias' in v.name)])

					# Add the regularization term to the loss
					total_loss = (reduced_loss +
								l2_regularization_strength * l2_loss)
				
					tf.summary.scalar('l2_loss', l2_loss, collections=["loss"])
					tf.summary.scalar('total_loss', total_loss, collections=["loss"])


					return total_loss, final_big_frame_state, final_frame_state

	# def loss_SampleRnn_DEBUG(self,
	# 	train_input_batch_rnn,
	# 	train_big_frame_state,
	# 	train_frame_state,
	# 	l2_regularization_strength=None,
	# 	name='sample'):
		
	# 	poids = tf.get_variable("poids", [self.batch_size*self.seq_len, self.dim], dtype=tf.float32)
	# 	total_loss = tf.reduce_mean(tf.matmul(tf.reshape(train_input_batch_rnn, [1, -1]), poids))
	# 	final_big_frame_state = tf.zeros((self.batch_size, self.dim))
	# 	final_frame_state = tf.zeros((self.batch_size, self.dim))
	# 	return total_loss, train_big_frame_state, train_frame_state