import time
import os
import shutil
import argparse
import random

import build_db
import wavenet.utils as utils
from wavenet.models import Model, Generator

# PREFIX="/home/leo"
PREFIX="/fast-1/leo"
# PREFIX="/sb/project/ymd-084-aa/leo"

DATA_DIRECTORY=PREFIX+'/WaveGeneration/Data/contrabass_no_cond/ordinario'
LOGDIR_ROOT=PREFIX+'/WaveGeneration/FastWaveNet/logdir/0'

# Data
SAMPLE_RATE=8000
SAMPLE_SIZE=2**10
SLIDING_RATIO=0.75
SILENCE_THRESHOLD=0.01
# Architecture
NUM_LAYERS=4
NUM_BLOCKS=1
NUM_HIDDEN=50
FILTER_WIDTH=2
Q_LEVELS=256        # Quantification for the amplitude of the audio samples
# Training
BATCH_SIZE=16
VALID_FREQ=1
GENERATE_FREQ=10

def get_arguments():
	parser = argparse.ArgumentParser(description='SampleRnn example network')
	# Framework
	parser.add_argument('--data_dir',         type=str,   default=DATA_DIRECTORY)
	parser.add_argument('--logdir_root',      type=str,   default=LOGDIR_ROOT)
	# Data
	parser.add_argument('--sample_rate',      type=int,   default=SAMPLE_RATE)
	parser.add_argument('--sample_size',      type=int,   default=SAMPLE_SIZE)
	parser.add_argument('--sliding_ratio',    type=float, default=SLIDING_RATIO)
	parser.add_argument('--silence_threshold',type=float,   default=SILENCE_THRESHOLD)
	# Architecture
	parser.add_argument('--num_layers',       type=int,   default=NUM_LAYERS)
	parser.add_argument('--num_blocks',       type=int,   default=NUM_BLOCKS)
	parser.add_argument('--num_hidden',       type=int,   default=NUM_HIDDEN)
	parser.add_argument('--filter_width',     type=int,   default=FILTER_WIDTH)
	parser.add_argument('--q_levels',         type=int,   default=Q_LEVELS)
	# Training	
	parser.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
	parser.add_argument('--valid_freq',       type=int,   default=VALID_FREQ)
	parser.add_argument('--generate_freq',       type=int,   default=GENERATE_FREQ)
	# Debug
	parser.add_argument('--summary',  type=bool, default=False)
	return parser.parse_args()

def main():
	##############################
	# Get args	
	args = get_arguments()
	##############################

	##############################
	# Build data chunk
	config_str = "_".join([str(args.sample_rate), str(args.sample_size), str(args.sliding_ratio), str(args.silence_threshold)])
	files_dir = args.data_dir
	npy_dir = files_dir + '/' + config_str
	lock_file_db = files_dir + '/lock'
	# Check if exists
	while(os.path.isfile(lock_file_db)):
		# Wait for the end of construction by another process
		time.sleep(1)
	if not os.path.isdir(npy_dir):
		try:
			# Build if not
			ff = open(lock_file_db, 'w')
			build_db.main(files_dir, npy_dir, args.sample_rate, args.sample_size, args.sliding_ratio, args.silence_threshold)
			ff.close()
		except:
			shutil.rmtree(npy_dir)
		finally:
			os.remove(lock_file_db)
		# data_statistics.bar_activations(save_dir, save_dir, sample_size_padded)
	##############################

	##############################
	# Init dirs
	utils.init_directory(args.logdir_root)
	if args.summary:
		logdir_summary = os.path.join(args.logdir_root, 'summary')
		utils.init_directory(logdir_summary)
	# Save	
	logdir_save = os.path.join(args.logdir_root, 'save')
	# Wave
	logdir_wav = os.path.join(args.logdir_root, 'wav')
	utils.init_directory(logdir_wav)
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
	model = Model(num_time_samples=args.sample_size,
     	num_channels=1,
     	num_classes=args.q_levels,
		num_blocks=args.num_blocks,
     	num_layers=args.num_layers,
     	num_hidden=args.num_hidden,
     	filter_width=args.filter_width,
     	gpu_fraction=0.9)
	print("TTT: Instanciate network : {}".format(time.time()-ttt))
	##############################
	
	##############################
	# Train
	tic = time.time()
	model.train(training_chunks, valid_chunks, args.batch_size, args.valid_freq, args.generate_freq)
	toc = time.time()
	print('Training took {} seconds.'.format(toc-tic))
	##############################

	##############################
	# Generate
	generator = Generator(model)
	# Get first sample of input
	input_ = inputs[:, 0:1, 0]
	tic = time()
	predictions = generator.run(input_, 32000)
	toc = time()
	print('Generating took {} seconds.'.format(toc-tic))
	##############################
	return

if __name__=="__main__":
	main()	