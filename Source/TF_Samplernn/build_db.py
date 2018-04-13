import fnmatch
import copy
import librosa
import os
import random
import shutil
import cPickle as pkl
import numpy as np
import progressbar


def randomize_files(files):
  files_idx= [i for i in xrange(len(files))]
  random.shuffle(files_idx)

  for idx in xrange(len(files)):
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


def main(audio_dir, save_dir, sample_rate, sample_size=None, silence_threshold=None):
	# Go through the dataset multiple times
	audio_list = []
	files = find_files(audio_dir)
	iterator = load_generic_audio(files, sample_rate)
	
	chunk_index = 0
	chunk_dir = save_dir + '/chunk'
	if os.path.isdir(chunk_dir):
		shutil.rmtree(chunk_dir)
	os.makedirs(chunk_dir)

	i = 0
	with progressbar.ProgressBar(max_value=len(files)) as bar:
		for audio_copy,filename in iterator:
			audio = copy.deepcopy(audio_copy)

			if silence_threshold is not None:
				# Remove silence
				audio = trim_silence(audio[:, 0], silence_threshold)
				audio = audio.reshape(-1, 1)
				if audio.size == 0:
					print("Warning: " + filename + " was ignored as it contains only "
						"silence. Consider decreasing trim_silence "
						"threshold, or adjust volume of the audio.").format()
				pad_elements = sample_size - 1 - (audio.shape[0] + sample_size - 1) % sample_size
				audio = np.concatenate([audio,
					np.full((pad_elements, 1), 0.0, dtype='float32')],
					axis=0)

			while len(audio) >= sample_size:
				piece = audio[:sample_size, :]
				# Save piece
				np.save(os.path.join(chunk_dir, str(chunk_index)),piece)
				audio = audio[sample_size:, :]
				chunk_index += 1
	        bar.update(i)
	        i+=1

if __name__ == '__main__':
	audio_dir='/home/aciditeam-leo/Aciditeam/WaveGeneration/Data/contrabass_no_cond/ordinario' 
	sample_rate=8000
	sample_size=4104
	silence_threshold=0.01

	# Save these parameters
	save_dir = os.path.join(audio_dir, str(sample_rate) + '_' + str(sample_size) + '_' + str(silence_threshold))
	if os.path.isdir(save_dir):
		shutil.rmtree(save_dir)
	os.makedirs(save_dir)
	params={}
	params["sample_rate"]=sample_rate
	params["sample_size"]=sample_size
	params["silence_threshold"]=silence_threshold
	pkl.dump(params, open(save_dir + '/params.pkl', 'wb'))
	
	main(audio_dir, save_dir, sample_rate, sample_size, silence_threshold)