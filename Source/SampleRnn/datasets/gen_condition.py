RAW_DATA_DIR=str(sys.argv[1])
print RAW_DATA_DIR
cond_files = glob.glob(RAW_DATA_DIR + '/*.csv')

for cond_file in cond_files:
    data_cond = np.loadtxt(cond_file_path, dtype='int8', delimiter=";")
    np.save(outcsv_name, chunk_cond)