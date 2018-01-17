#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import wav2npy
import sys
import preprocess

if __name__ == '__main__':
	chunk_length = 1
	slide_length = 1
	RAW_DATA_DIR=str(sys.argv[1])
	preprocess.main(RAW_DATA_DIR, chunk_length, slide_length)
	wav2npy.main(RAW_DATA_DIR)