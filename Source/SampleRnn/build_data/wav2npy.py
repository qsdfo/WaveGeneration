#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import scikits.audiolab

import random
import re
import time
import os
import shutil
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

    np.save(os.path.join(save_path, 'all_music.npy'), np_arr)
    np.save(os.path.join(save_path, 'music_train.npy'), np_arr[:-2*256])
    np.save(os.path.join(save_path, 'music_valid.npy'), np_arr[-2*256:-256])
    np.save(os.path.join(save_path, 'music_test.npy'), np_arr[-256:])
    return

if __name__ == '__main__':
    data_path=str(sys.argv[1])
    main(data_path)