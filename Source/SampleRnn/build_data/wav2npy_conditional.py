import numpy as np
import scikits.audiolab
import re
import random
import shutil
import time
import os
import glob

def main(data_path):
    __RAND_SEED = 123
    def __fixed_shuffle(inp_list):
        if isinstance(inp_list, list):
            random.seed(__RAND_SEED)
            random.shuffle(inp_list)
            return
        #import collections
        #if isinstance(inp_list, (collections.Sequence)):
        if isinstance(inp_list, numpy.ndarray):
            numpy.random.seed(__RAND_SEED)
            numpy.random.shuffle(inp_list)
            return
        # destructive operations; in place; no need to return
        raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

    
    data_path = os.path.join(data_path, 'parts')
    save_path = re.sub('parts', 'matrices', data_path)
    
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print data_path
    print save_path

    paths = sorted(glob.glob(data_path+"/*.wav"))
    __fixed_shuffle(paths)

    arr = [(scikits.audiolab.wavread(p)[0]).astype('float16') for p in paths]
    np_arr = np.array(arr)

    ######################################################
    # Here load conditional vectors
    # It has to be a numpy array of shape : (np_arr.shape[0], np_arr.shape[1], COND_DIM)
    # It's a one-hot vector, so for all i,j, sum()
    num_files = np_arr.shape[0]
    lenght_file = np_arr.shape[1]
    cond = [np.load(re.sub(ur'\.wav$', '.npy', p)).astype('int8') for p in paths]
    np_cond_int = np.array(cond)
    COND_DIM = np.amax(np_cond_int) + 1
    np_cond = np.zeros(np_cond_int.shape + (COND_DIM,))
    for x in range(num_files):
        for y in range(lenght_file):
            ind_one = np_cond_int[x,y]
            np_cond[x,y,ind_one] = 1
    
    # Test integrity (not exhaustive but very few chances for a matrix to pass this test and not be a condition matrix)
    assert np.all(np.sum(np_cond, axis=2) == np.ones((num_files, lenght_file)))
    ######################################################

    num_files = len(np_arr)
    num_train = int(max(0.8*num_files, num_files-2*256))
    num_test = int((num_files - num_train) / 2)
    num_valid = int(num_test)

    np.save(os.path.join(save_path, 'music_train.npy'), np_arr[:num_train])
    np.save(os.path.join(save_path, 'music_valid.npy'), np_arr[num_train:num_train+num_test])
    np.save(os.path.join(save_path, 'music_test.npy'), np_arr[-num_test:])
    np.save(os.path.join(save_path, 'music_train_condition.npy'), np_cond[:num_train])
    np.save(os.path.join(save_path, 'music_valid_condition.npy'), np_cond[num_train:num_train+num_test])
    np.save(os.path.join(save_path, 'music_test_condition.npy'), np_cond[-num_test:])

    with open('number_conditions.txt', 'wb') as f:
        f.write(str(COND_DIM))

if __name__ == '__main__':
    import sys
    data_path=str(sys.argv[1])
    main(data_path)