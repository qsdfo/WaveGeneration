import fnmatch
import copy
import librosa
import os
import random
import shutil
import pickle as pkl
import numpy as np
import progressbar

import data_statistics


def randomize_files(files):
  files_idx = [i for i in range(len(files))]
  random.shuffle(files_idx)

  for idx in range(len(files)):
    yield files[files_idx[idx]]


def find_files(directory, pattern='*.wav'):
  '''Recursively finds all files matching the pattern.'''
  files = []
  for root, dirnames, filenames in os.walk(directory):
      for filename in fnmatch.filter(filenames, pattern):
          files.append(os.path.join(root, filename))
  return files


def load_generic_audio(files, sample_rate):
	'''Generator that yields audio waveforms from the directory.'''
	print("files length: {}".format(len(files)))
	randomized_files = randomize_files(files)
	for filename in randomized_files:
		audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
		audio = audio.reshape(-1, 1)
		yield audio, filename


def trim_silence(audio, threshold):
  '''Removes silence at the beginning and end of a sample.'''
  energy = librosa.feature.rmse(audio)
  frames = np.nonzero(energy > threshold)
  indices = librosa.core.frames_to_samples(frames)[1]
  # Note: indices can be an empty array, if the whole audio was silence.
  return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def main(audio_dir, save_dir, sample_rate, sample_size=None, sliding_ratio=None, silence_threshold=None):
	# Go through the dataset multiple times
	audio_list = []
	files = find_files(audio_dir)
	iterator = load_generic_audio(files, sample_rate)
	
	chunk_index = 0
	if os.path.isdir(save_dir):
		shutil.rmtree(save_dir)
	os.makedirs(save_dir)

	for audio_copy,filename in progressbar.progressbar(iterator):
		# Init
		start_time = 0
		audio = copy.deepcopy(audio_copy)
			
		if silence_threshold is not None:
			# Remove silence
			audio = trim_silence(audio[:, 0], silence_threshold)
			audio = audio.reshape(-1, 1)
			if audio.size == 0:
				print(("Warning: " + filename + " was ignored as it contains only silence. Consider decreasing trim_silence threshold, or adjust volume of the audio.").format())

		while len(audio) >= (start_time+sample_size):
			piece = audio[start_time:start_time+sample_size]
			# Save piece
			np.save(os.path.join(save_dir, str(chunk_index)),piece)
			chunk_index += 1
			start_time = int(start_time + sliding_ratio*sample_size)

		# Instead of zero padding, take the last batch the remainder of the last batch if not too long
		remaining_samples = len(audio) - start_time
		if remaining_samples > sample_size / 3:
			piece = audio[-sample_size:]
			if len(piece) != sample_size:
				# Case when len(audio) < sample_size
				continue
			np.save(os.path.join(save_dir, str(chunk_index)),piece)
			
	return

if __name__ == '__main__':
	audio_dir='/fast-1/leo/WaveGeneration/Data/contrabass_no_cond/ordinario'
	# audio_dir='/Users/leo/Recherche/WaveGeneration/Data/contrabass_no_cond/ordinario'
	sample_rate=8000
	sample_size=2**14+8
	# sample_size=2**10+8
	sliding_ratio=0.75
	silence_threshold=0.01

	config_str = "_".join([sample_rate, sample_size, sliding_ratio, silence_threshold])
	save_dir = audio_dir + '/' + config_str

	main(audio_dir, save_dir, sample_rate, sample_size, sliding_ratio, silence_threshold)
	data_statistics.bar_activations(save_dir, save_dir, sample_size)