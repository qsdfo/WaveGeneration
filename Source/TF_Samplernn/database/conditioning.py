import numpy as np

def generate_condition_from_audio(audio):
	conditioning = np.random.randint(2, size=audio.shape[0])
	return conditioning

def generate_condition_from_db(database_name):
	return