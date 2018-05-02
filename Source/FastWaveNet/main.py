from time import time

from wavenet.utils import make_batch
from wavenet.models import Model, Generator

from IPython.display import Audio


if __name__=="__main__":
	inputs, targets = make_batch('assets/voice.wav')
	num_time_samples = inputs.shape[1]
	num_channels = 1
	gpu_fraction = 0.9

	model = Model(num_time_samples=num_time_samples,
	              num_channels=num_channels,
	              gpu_fraction=gpu_fraction)

	Audio(inputs.reshape(inputs.shape[1]), rate=44100)

	tic = time()
	model.train(inputs, targets)
	toc = time()

	print('Training took {} seconds.'.format(toc-tic))

	generator = Generator(model)

	# Get first sample of input
	input_ = inputs[:, 0:1, 0]

	tic = time()
	predictions = generator.run(input_, 32000)
	toc = time()
	print('Generating took {} seconds.'.format(toc-tic))

	Audio(predictions, rate=44100)