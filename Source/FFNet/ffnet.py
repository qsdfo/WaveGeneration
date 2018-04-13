def ffnet(num_classes, input_shape, deep):
 
	import keras.backend as K
	from keras.layers import Input, Lambda, Add, Activation, GaussianNoise, BatchNormalization

	#nb_filters = num_classes
	inp = Input(shape=input_shape, name=‘input_part’)
	#x = BatchNormalization()(inp)
	l = {}
	r = {}
	cl = {}
	cr = {}
	x = inp
	#x = GaussianNoise(0.1)(inp)
	i = deep
	l[i] = Lambda(lambda x: x[:,2**i:,:], output_shape=(2**i,1,))(x)
	r[i] = Lambda(lambda x: x[:,:2**i,:], output_shape=(2**i,1,))(x)
	cl[i] = Conv1D(2**i, kernel_size=1, padding=‘valid’, name=‘conv_l_%d’ % (2 ** i))(l[i])
	cr[i] = Conv1D(2**i, kernel_size=1, padding=‘valid’, name=‘conv_r_%d’ % (2 ** i))(r[i])
	x = Add()([cl[i],cr[i]])
	x = Activation(‘relu’)(x)
	x = Conv1D(2**i, kernel_size=1, padding=‘valid’, name=‘conv_sum_%d’ % (2 ** i))(x)
	x = Activation(‘relu’)(x)
	
	for i in reversed(range(deep)):
		print(i)
		l[i] = Lambda(lambda x: x[:,2**i:,:], output_shape=(2**i,2**(i+1),))(x)
		r[i] = Lambda(lambda x: x[:,:2**i,:], output_shape=(2**i,2**(i+1),))(x)
		cl[i] = Conv1D(2**i, kernel_size=1, padding=‘valid’, name=‘conv_l_%d’ % (2 ** i))(l[i])
		cr[i] = Conv1D(2**i, kernel_size=1, padding=‘valid’, name=‘conv_r_%d’ % (2 ** i))(r[i])
		x = Add()([cl[i],cr[i]])
		x = Activation(‘relu’)(x)
		x = Conv1D(2**i, kernel_size=1, padding=‘valid’, name=‘conv_sum_%d’ % (2 ** i))(x)
		x = Activation(‘relu’)(x)
		
	y = Flatten()(x)
	#y = Dense(1000, activation = ‘relu’)(y)
	y = Dense(500, activation = ‘relu’)(y)
	y = Dense(num_classes)(y)    
	model = Model(inp, y)      

	return model