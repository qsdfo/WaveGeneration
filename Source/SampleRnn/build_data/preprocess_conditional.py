# Import wav from database and split them in parts to be processed by the algo

import os
import sys
import re
import glob
import shutil

import numpy as np
import scikits.audiolab
from scikits.audiolab import Format, Sndfile


def main(RAW_DATA_DIR, CHUNK_LENGTH, SLIDE_LENGTH):
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
        cond_file_path = path_no_extension + '.csv'
        # Read wav and csv files
        wave_info = scikits.audiolab.wavread(wav_file_path)
        data_wave = wave_info[0]
        samplerate = wave_info[1]
        if os.path.isfile(cond_file_path):
            data_cond = np.loadtxt(cond_file_path, dtype='int8', delimiter=";")
        else:
            raise OSError('Conditioning file does not exists for file ' + path_no_extension)

        # Integrity check
        len_wave = len(data_wave)
        assert len(data_cond) == len_wave, 'Wave and conditioning files do not have the same length'

        # Cut them in CHUNK_LENGTH seconds chunks.
        # Sliding of SLIDE_LENGTH seconds
        # Zero padding at the end
        time_counter = 0
        start_index = 0
        while(start_index < len_wave):
            end_index = (time_counter + CHUNK_LENGTH) * samplerate
            if end_index > len_wave:
                chunk_wave = np.concatenate((data_wave[start_index:], np.zeros((end_index-len_wave))))
                chunk_cond = np.concatenate((data_cond[start_index:], np.zeros((end_index-len_wave))))
            else:
                chunk_wave = data_wave[start_index:end_index]
                chunk_cond = data_cond[start_index:end_index]

            time_counter += SLIDE_LENGTH
            start_index = time_counter * samplerate

            # Write the chunks
            outwave_name = OUTPUT_DIR + '/p' + str(chunk_counter) + '.wav'
            outcsv_name = OUTPUT_DIR + '/p' + str(chunk_counter) + '.npy'
            f = Sndfile(outwave_name, 'w', format, 1, samplerate)  # 1 stands for the number of channels
            f.write_frames(chunk_wave)
            np.save(outcsv_name, chunk_cond)

            chunk_counter += 1
    return
    
if __name__ == '__main__':
    CHUNK_LENGTH = 8
    SLIDE_LENGTH = 2
    RAW_DATA_DIR=str(sys.argv[1])
    main(RAW_DATA_DIR, CHUNK_LENGTH, SLIDE_LENGTH)