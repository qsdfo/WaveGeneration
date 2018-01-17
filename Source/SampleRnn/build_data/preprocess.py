#!/usr/bin/env python2
# -*- coding: utf-8 -*-

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
#### ORIGINAL SCRIPT FROM SAMPLERNN CODE
# import os
# import sys
# import subprocess

# RAW_DATA_DIR=str(sys.argv[1])
# OUTPUT_DIR=os.path.join(RAW_DATA_DIR, "parts")
# os.makedirs(OUTPUT_DIR)
# print RAW_DATA_DIR
# print OUTPUT_DIR

# # Step 1: write all filenames to a list
# with open(os.path.join(OUTPUT_DIR, 'preprocess_file_list.txt'), 'w') as f:
#     filenames = glob.glob(RAW_DATA_DIR + '/*.wav')
#     for filename in filenames:
#         f.write("file '" + dirpath + '/' + filename + "'\n")

# # Step 2: concatenate everything into one massive wav file
# os.system("ffmpeg -f concat -safe 0 -i {}/preprocess_file_list.txt {}/preprocess_all_audio.wav".format(OUTPUT_DIR, OUTPUT_DIR))

# # # get the length of the resulting file
# length = float(subprocess.check_output('ffprobe -i {}/preprocess_all_audio.wav -show_entries format=duration -v quiet -of csv="p=0"'.format(OUTPUT_DIR), shell=True))

# # # Step 3: split the big file into 8-second chunks
# for i in xrange(int(length)//8 - 1):
#     os.system('ffmpeg -ss {} -t 8 -i {}/preprocess_all_audio.wav -ac 1 -ab 16k -ar 16000 {}/p{}.flac'.format(i, OUTPUT_DIR, OUTPUT_DIR, i))

# # # Step 4: clean up temp files
# os.system('rm {}/preprocess_all_audio.wav'.format(OUTPUT_DIR))
# os.system('rm {}/preprocess_file_list.txt'.format(OUTPUT_DIR))
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################

# Import wav from database and split them in parts to be processed by the algo
import os
import sys
import re
import glob
import shutil

import numpy as np
import scikits.audiolab
from scikits.audiolab import Format, Sndfile


def main(RAW_DATA_DIR, chunk_length, slide_length):
	OUTPUT_DIR=os.path.join(RAW_DATA_DIR, "parts")
	
	if os.path.isdir(OUTPUT_DIR):
		answer = raw_input("Parts already exist.\nType y to rebuild from scratch : ")
		if answer != 'y':
			print("NOT REBUILT !")
			sys.exit()
		else:
			shutil.rmtree(OUTPUT_DIR)
	os.makedirs(OUTPUT_DIR)
	
	print RAW_DATA_DIR
	print OUTPUT_DIR

	chunk_counter = 0
	format = Format('wav')

	wav_files = glob.glob(RAW_DATA_DIR + '/*.wav')

	for filename in wav_files:
		wav_file_path = filename
		path_no_extension = re.sub(ur'\.wav$', '', filename)
		# Read wav and csv files
		wave_info = scikits.audiolab.wavread(wav_file_path)
		data_wave = wave_info[0]
		samplerate = wave_info[1]
		len_wave = len(data_wave)

		# Cut them in chunk_length seconds chunks.
		# Sliding of slide_length seconds
		# Zero padding at the end
		time_counter = 0
		start_index = 0
		while(start_index < len_wave):
			end_index = (time_counter + chunk_length) * samplerate
			if end_index > len_wave:
				chunk_wave = np.concatenate((data_wave[start_index:], np.zeros((end_index-len_wave))))
			else:
				chunk_wave = data_wave[start_index:end_index]

			time_counter += slide_length
			start_index = time_counter * samplerate

			# Write the chunks
			outwave_name = OUTPUT_DIR + '/p' + str(chunk_counter) + '.wav'
			f = Sndfile(outwave_name, 'w', format, 1, samplerate)  # 1 stands for the number of channels
			f.write_frames(chunk_wave)

			chunk_counter += 1
	return

if __name__ == '__main__':
	chunk_length = 1
	slide_length = 0.75
	RAW_DATA_DIR=str(sys.argv[1])
	main(RAW_DATA_DIR, chunk_length, slide_length)