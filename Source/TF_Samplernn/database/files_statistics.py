import copy
import re
from TF_Samplernn.database.build_db import find_files, load_generic_audio

def main(root, db, sample_rate):
	audio_dir = root + '/' + db
	files = find_files(audio_dir)
	iterator = load_generic_audio(files, sample_rate)

	number_file = len(files)
	total_length_seconds = 0
	total_length_sample = 0
	number_of_notes = 0
	notes = set()
	silence_threshold = None

	for audio_copy, filename in iterator:
		audio = copy.deepcopy(audio_copy)
	
		if silence_threshold is not None:
			# Remove silence
			audio = trim_silence(audio[:, 0], silence_threshold)
			audio = audio.reshape(-1, 1)
			if audio.size == 0:
				print(("Warning: " + filename + " was ignored as it contains only silence. Consider decreasing trim_silence threshold, or adjust volume of the audio.").format())

		total_length_seconds += len(audio)/sample_rate
		total_length_sample += len(audio)

		split_filename = re.split("-", filename)
		if db in ['contrabass', 'violin']:
			pitch_class = re.sub(r"[0-9]", "", split_filename[-3])
		else:
			pitch_class = re.sub(r"[0-9]", "", split_filename[-2])
		notes.add(pitch_class)
	print(notes)
	return number_file, total_length_seconds, total_length_sample, len(notes)


if __name__ == '__main__':
	db_list =["oboe", "clarinet", "violin", "contrabass", "flute"]
	root = "/fast-1/leo/WaveGeneration/Data/single_instrument"
	for db in db_list:
		number_file, total_length_seconds, total_length_sample, number_of_notes = main(root, db, 16000)
		print(("######## {}\nnumber files: {}\nlength samples: {}\nmean duration: {:.3f}\nnotes: {}").format(db, number_file, total_length_sample, total_length_seconds/number_file, number_of_notes))