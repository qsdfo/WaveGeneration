import tensorflow as tf
import numpy as np

if __name__ == '__main__':
	import pdb; pdb.set_trace()

	x = [0, 1, 2, 3]

	x_PH = tf.placeholder(tf.int32)
	y_N = x_PH + 1
	
	with tf.Session() as sess:
		y = sess.run(y_N, feed_dict={x_PH: x})