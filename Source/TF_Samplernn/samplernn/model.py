import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from .ops import mu_law_encode

class SampleRnnModel(object):
  def __init__(self, batch_size, big_frame_size, frame_size,
             q_levels, rnn_type, dim, n_rnn, seq_len, emb_size, autoregressive_order):
    self.batch_size = batch_size
    self.big_frame_size = big_frame_size
    self.frame_size = frame_size
    self.q_levels = q_levels
    self.rnn_type = rnn_type
    self.dim = dim
    self.n_rnn = n_rnn
    self.seq_len=seq_len
    self.emb_size=emb_size
    self.autoregressive_order=autoregressive_order

    def single_cell():
      return tf.contrib.rnn.GRUCell(self.dim)
    if 'LSTM' == self.rnn_type:
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(self.dim)
    self.cell = single_cell()
    self.big_cell = single_cell()
    if self.n_rnn > 1:
      self.cell = tf.contrib.rnn.MultiRNNCell(
             [single_cell() for _ in range(self.n_rnn)])
      self.big_cell = tf.contrib.rnn.MultiRNNCell(
             [single_cell() for _ in range(self.n_rnn)])
    self.initial_state   = self.cell.zero_state(self.batch_size, tf.float32)
    self.big_initial_state   = self.big_cell.zero_state(self.batch_size, tf.float32)

  def _preprocess_audio_inputs(self, input_frames):
    input_frames = (input_frames / (self.q_levels/2.0)) - 1.0
    input_frames *= 2.0
    return input_frames

  def _upsampling_reshape(self, list_frames, upsampling_ratio):
    """Take as input a list of upsampled tensor and reshape them to fit witht the frame_size of the next level
    """
    x = tf.stack(list_frames)
    # By default, stack along dimension 0... so transpose : (num_step, batch_size, proj_dim) -> (batch_size, num_step, proj_dim)
    x = tf.transpose(x, perm=[1, 0, 2])
    x_shape = x.get_shape()
    x = tf.reshape(x,
     [x_shape[0],
      x_shape[1] * upsampling_ratio,
     -1])
    return x

  def _create_network_BigFrame(self,
    		num_steps,
    		big_frame_state,
    		big_input_sequences):
    with tf.variable_scope('BigFrame_layer'):
      big_input_sequences_shape = big_input_sequences.get_shape()
      big_input_frames = tf.reshape(big_input_sequences,[
                            big_input_sequences_shape[0],
                            big_input_sequences_shape[1] / self.big_frame_size,
                            self.big_frame_size])
      big_input_frames = self._preprocess_audio_inputs(big_input_frames)

      # Note : self.big_frame_size/self.frame_size est le ratio d'upsampling
      big_frame_outputs = []
      big_frame_proj_weights = tf.get_variable(
        "big_frame_proj_weights", [self.dim, self.dim * self.big_frame_size/self.frame_size], dtype=tf.float32)
      with tf.variable_scope("BIG_FRAME_RNN"):
        for time_step in range(num_steps):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          (big_frame_cell_output, big_frame_state) = self.big_cell(big_input_frames[:, time_step, :], big_frame_state)

          #>
          # Ici ils utilisent une projection differente pour chaque time_step upsamplee (ce qui est fait dans l'article, pour j dans [0, r-1] representant l'upsampling : c_{(t-1)*r+j} = W_j h_t )
          # Perhaps we can just repeat the prediction self.big_frame_size/self.frame_size times. That would speed things up a little bit
          big_frame_outputs.append(math_ops.matmul(big_frame_cell_output, big_frame_proj_weights))
          #>

        final_big_frame_state = big_frame_state
      big_frame_outputs = self._upsampling_reshape(big_frame_outputs, self.big_frame_size/self.frame_size)
      return big_frame_outputs,final_big_frame_state

  def _create_network_Frame(self,
    		num_steps,
    		big_frame_outputs,
    		frame_state,
    		input_sequences):
    with tf.variable_scope('Frame_layer'):
      input_sequences_shape = input_sequences.get_shape()
      input_frames = tf.reshape(input_sequences,[
                        input_sequences_shape[0],
                        input_sequences_shape[1] / self.frame_size,
                        self.frame_size])
      input_frames = self._preprocess_audio_inputs(input_frames)
     
      frame_outputs = []
      frame_proj_weights = tf.get_variable(
        "frame_proj_weights", [self.dim, self.dim * self.frame_size], dtype=tf.float32)
      frame_cell_proj_weights = tf.get_variable(
        "frame_cell_proj_weights", [self.frame_size, self.dim], dtype=tf.float32)
      with tf.variable_scope("FRAME_RNN"):
        for time_step in range(num_steps):
          if time_step > 0: tf.get_variable_scope().reuse_variables()
          # Audio sample influence
          cell_input = input_frames[:, time_step, :]
          cell_input = math_ops.matmul(cell_input, frame_cell_proj_weights)
          # Previous tier influence
          cell_input = cell_input + big_frame_outputs[:, time_step, :]
          (frame_cell_output, frame_state) = self.cell(cell_input, frame_state)

          frame_outputs.append(math_ops.matmul(frame_cell_output, frame_proj_weights))
      final_frame_state = frame_state
      frame_outputs = self._upsampling_reshape(frame_outputs, self.frame_size)   # Actually self.frame_size / 1 as the upsampling ratio of the last level is 1
      return frame_outputs, final_frame_state

  def _create_network_Sample(self,
    		frame_outputs,
    		sample_input_sequences):
    with tf.variable_scope('Sample_layer'):
      sample_input_sequences_shape = sample_input_sequences.get_shape()
      sample_shap=[sample_input_sequences_shape[0],
      	     sample_input_sequences_shape[1]*self.emb_size,
      	     1]
      embedding = tf.get_variable("embedding", [self.q_levels, self.emb_size])
      sample_input_sequences = embedding_ops.embedding_lookup(embedding, tf.reshape(sample_input_sequences,[-1]))
      # Ici les embedding de chaque frames sont mises les uns a cote des autres (a l'interieur d'un meme batch)
      sample_input_sequences = tf.reshape(sample_input_sequences, sample_shap)
     
      '''Create a convolution filter variable with the specified name and shape,
      and initialize it using Xavier initialition.'''
      filter_initializer = tf.contrib.layers.xavier_initializer_conv2d()
      # conv_shape : (kernel_size, channel, output_dim)
      sample_filter_shape = [self.emb_size*self.autoregressive_order, 1, self.dim]  # self.emb_size*self.autoregressive_order : le noyau de convolution couvre un horizon de deux samples. 
      # Cf l'argument stride dans la fonction conv1D plus bas : on fait des pas de taille emb_size a chaque fois
      sample_filter = tf.get_variable("sample_filter", sample_filter_shape,
    		initializer = filter_initializer)
      out = tf.nn.conv1d(sample_input_sequences,
      		sample_filter,
      		stride=self.emb_size,
      		padding="VALID", 
      		name="sample_conv")
      out = out + frame_outputs
      
      #> Use self.dim also for MLP ?
      sample_mlp1_weights = tf.get_variable(
        "sample_mlp1", [self.dim, self.dim], dtype=tf.float32)
      sample_mlp2_weights = tf.get_variable(
        "sample_mlp2", [self.dim, self.dim], dtype=tf.float32)
      sample_mlp3_weights = tf.get_variable(
        "sample_mlp3", [self.dim, self.q_levels], dtype=tf.float32)
      #>

      #>
      # Pas de biases ?
      out = tf.reshape(out, [-1,self.dim])
      out = math_ops.matmul(out, sample_mlp1_weights)
      out = tf.nn.relu(out)
      out = math_ops.matmul(out, sample_mlp2_weights)
      out = tf.nn.relu(out)
      out = math_ops.matmul(out, sample_mlp3_weights)
      out = tf.reshape(out, [-1, sample_input_sequences_shape[1]-self.autoregressive_order+1, self.q_levels])
      #>
      return out

  def _create_network_SampleRnn(self,
    		train_big_frame_state,
    		train_frame_state):
    with tf.name_scope('SampleRnn_net'):
      #big frame 
      big_input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)\
    				[:,:-self.big_frame_size,:]
      big_frame_num_steps = (self.seq_len-self.big_frame_size)/self.big_frame_size
      big_frame_outputs,\
      final_big_frame_state = \
    		self._create_network_BigFrame(num_steps = big_frame_num_steps, 
    				big_frame_state = train_big_frame_state,
      				big_input_sequences = big_input_sequences)
      #frame 
      input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)[:, 
                              self.big_frame_size-self.frame_size:-self.frame_size, :]
      frame_num_steps = (self.seq_len-self.big_frame_size)/self.frame_size
      frame_outputs, final_frame_state  = \
    	self._create_network_Frame(num_steps = frame_num_steps,
      				big_frame_outputs = big_frame_outputs,
    				frame_state = train_frame_state,
      				input_sequences = input_sequences)
      #sample
      sample_input_sequences = self.encoded_input_rnn[:, 
                               self.big_frame_size-self.autoregressive_order:-1, :]
      sample_output = self._create_network_Sample(frame_outputs,
      			sample_input_sequences=sample_input_sequences)
      return sample_output, final_big_frame_state, final_frame_state
  def loss_SampleRnn(self,
    	             train_input_batch_rnn,
    	             train_big_frame_state,
    	             train_frame_state,
                     l2_regularization_strength=None,
                     name='sample'):
    with tf.name_scope(name):
      # Process input
      self.encoded_input_rnn = mu_law_encode(train_input_batch_rnn, self.q_levels)
        
      # Train
      raw_output, final_big_frame_state, final_frame_state = \
        self._create_network_SampleRnn( train_big_frame_state, train_frame_state)

      with tf.name_scope('loss'):
        # Target
        target = tf.reshape(self.encoded_input_rnn[:, self.big_frame_size:], [-1])

        # Prediction
        prediction = tf.reshape(raw_output, [-1, self.q_levels])

        # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits( 
            logits=prediction, 
            labels=target)
        reduced_loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', reduced_loss)
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

          tf.summary.scalar('l2_loss', l2_loss)
          tf.summary.scalar('total_loss', total_loss)

          return total_loss, final_big_frame_state, final_frame_state
