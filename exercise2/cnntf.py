########################  Syed Sagheer Hussain    ##############################
########################  Matriculation: 4353497  ##############################
########################  Embedded System Masters ## #############################


from __future__ import print_function

import tensorflow as tf
import numpy as np

class CnnTensorflow():

	def model(features, labels, mode):
		
		inputlayer = tf.reshape(features["x"], [-1, 28, 28, 1])

		conv1 = tf.layers.conv2d (
					inputs = inputlayer,
					filters = 16,
					kernel_size = 5,
					padding = "same",
					activation = tf.nn.relu)
		
		pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=2, strides=1)
		conv2 = tf.layers.conv2d (
					inputs = pool1, 
					filters = 16,
					kernel_size = 5,
					padding = "same",
					activation = tf.nn.relu)
		pool2 = tf.layers.max_pooling2d (inputs = conv2, pool_size=2, strides=1)
		pool2_flat = tf.reshape(pool2, [-1, 26 * 26 * 16])
		dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
		dropout = tf.layers.dropout (
				inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
		logits = tf.layers.dense(inputs=dense, units=10)
		predictions = {
			"classes": tf.argmax(input=logits, axis=1),
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
			train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=predictions["classes"])}
			
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



