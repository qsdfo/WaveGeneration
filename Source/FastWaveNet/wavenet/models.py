import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from multiprocessing.pool import ThreadPool

import wavenet.utils as utils

from wavenet.layers import (_causal_linear, _output_linear, conv1d,
					dilated_conv1d)


class Model(object):
	def __init__(self,
		num_time_samples,
		num_channels=1,
		num_classes=256,
		num_blocks=2,
		num_layers=14,
		num_hidden=128,
		filter_width=2,
		gpu_fraction=1.0):
		
		self.num_time_samples = num_time_samples
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.num_blocks = num_blocks
		self.num_layers = num_layers
		self.num_hidden = num_hidden
		self.filter_width = filter_width
		self.gpu_fraction = gpu_fraction
		
	def build_training_graph
		inputs = tf.placeholder(tf.float32, shape=(None, num_time_samples-1, num_channels))
		targets = tf.placeholder(tf.int32, shape=(None, num_time_samples-1))

		h = inputs
		hs = []
		for b in range(num_blocks):
			for i in range(num_layers):
				rate = 2**i
				name = 'b{}-l{}'.format(b, i)
				h = dilated_conv1d(h, num_hidden, filter_width=self.filter_width, rate=rate, name=name)
				hs.append(h)

		outputs = conv1d(h,
						 num_classes,
						 filter_width=1,
						 gain=1.0,
						 activation=None,
						 bias=True)

		costs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs)
		cost = tf.reduce_mean(costs)

		train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		sess.run(tf.global_variables_initializer())

		self.inputs = inputs
		self.targets = targets
		self.outputs = outputs
		self.hs = hs
		self.costs = costs
		self.cost = cost
		self.train_step = train_step
		self.sess = sess


		self.bins = np.linspace(-1, 1, self.model.num_classes)

		inputs = tf.placeholder(tf.float32, [batch_size, input_size],
								name='inputs')

		print('Make Generator.')

		count = 0
		h = inputs

		init_ops = []
		push_ops = []
		for b in range(self.model.num_blocks):
			for i in range(self.model.num_layers):
				rate = 2**i
				name = 'b{}-l{}'.format(b, i)
				if count == 0:
					state_size = 1
				else:
					state_size = self.model.num_hidden
					
				q = tf.FIFOQueue(rate,
								 dtypes=tf.float32,
								 shapes=(batch_size, state_size))
				init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

				state_ = q.dequeue()
				push = q.enqueue([h])
				init_ops.append(init)
				push_ops.append(push)

				h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
				count += 1

		outputs = _output_linear(h)

		out_ops = [tf.nn.softmax(outputs)]
		out_ops.extend(push_ops)

		self.inputs = inputs
		self.init_ops = init_ops
		self.out_ops = out_ops
		
		# Initialize queues.
		self.model.sess.run(self.init_ops)


	def _train(self, inputs, targets):
		feed_dict = {self.inputs: inputs, self.targets: targets}
		cost, _ = self.sess.run([self.cost, self.train_step], feed_dict=feed_dict)
		return cost

	def _validate(self, inputs, targets):
		feed_dict = {self.inputs: inputs, self.targets: targets}
		cost, = self.sess.run([self.cost], feed_dict=feed_dict)
		return cost

	def train(self, train_chunk_list, valid_chunk_list, batch_size, valid_freq, generate_freq):
		number_strips = 3
		validation_order = 2
		losses = []
		val_tab = []

		overfitting = False
		step_counter = 0
		chunk_counter = 0
		
		pool = ThreadPool(processes=2)
		async_train = pool.apply_async(utils.make_batch, (train_chunk_list, chunk_counter, batch_size, self.num_classes))
		async_valid = pool.apply_async(utils.make_batch, (valid_chunk_list, 0, len(valid_chunk_list), self.num_classes))
		inputs, targets, chunk_counter = async_train.get()
		inputs_val, targets_val, _ = async_valid.get()

		while not overfitting:
			# Load inputs and targets)
			async_train = pool.apply_async(utils.make_batch, (train_chunk_list, chunk_counter, batch_size, self.num_classes))
			
			# Train step
			cost = self._train(inputs, targets)
			losses.append(cost)
			
			# Validation step
			if step_counter % valid_freq == 0:
				valid_step = step_counter//valid_freq
				# Validate
				valid_cost = self._validate(inputs_val, targets_val)
				val_tab.append(valid_cost)
				if valid_step > (number_strips + validation_order):
					# Perform at least 10 epoch
					overfitting = utils.up_criterion(val_tab, valid_step-1, 3, 2)
				print("Step {:d} - loss : {:.4f} - val : {:.4f}".format(step_counter, cost, valid_cost))

			# Generate
			if (step_counter % generate_freq == 0) or overfitting:


			# increment step counter
			step_counter += 1
		return

class Generator(object):
	def __init__(self, model, batch_size=1, input_size=1):
		self.model = model
		self.bins = np.linspace(-1, 1, self.model.num_classes)

		inputs = tf.placeholder(tf.float32, [batch_size, input_size],
								name='inputs')

		print('Make Generator.')

		count = 0
		h = inputs

		init_ops = []
		push_ops = []
		for b in range(self.model.num_blocks):
			for i in range(self.model.num_layers):
				rate = 2**i
				name = 'b{}-l{}'.format(b, i)
				if count == 0:
					state_size = 1
				else:
					state_size = self.model.num_hidden
					
				q = tf.FIFOQueue(rate,
								 dtypes=tf.float32,
								 shapes=(batch_size, state_size))
				init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

				state_ = q.dequeue()
				push = q.enqueue([h])
				init_ops.append(init)
				push_ops.append(push)

				h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
				count += 1

		outputs = _output_linear(h)

		out_ops = [tf.nn.softmax(outputs)]
		out_ops.extend(push_ops)

		self.inputs = inputs
		self.init_ops = init_ops
		self.out_ops = out_ops
		
		# Initialize queues.
		self.model.sess.run(self.init_ops)

	def run(self, input_var, num_samples):
		predictions = []
		for step in range(num_samples):

			feed_dict = {self.inputs: input_var}
			output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0] # ignore push ops
			value = np.argmax(output[0, :])

			input_var = np.array(self.bins[value])[None, None]
			predictions.append(input_var)

			if step % 1000 == 0:
				predictions_ = np.concatenate(predictions, axis=1)
				# plt.plot(predictions_[0, :], label='pred')
				# plt.legend()
				# plt.xlabel('samples from start')
				# plt.ylabel('signal')
				# plt.show()

		predictions_ = np.concatenate(predictions, axis=1)
		return predictions_
