import os
import random as rnd
import numpy as np
import tensorflow as tf
#import tensorflow.keras.backend as K
#import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import pandas as pd
#tf.config.experimental_run_functions_eagerly(True)


# Check the TensorFlow documentation (https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric) to see how to create custom metrics. 
# Specifically, the following metrics were adapted from the 'BinaryTruePositives' class of the TensorFlow documentation. 

class Specificity(tf.keras.metrics.Metric):

	def __init__(self, name='specificity', **kwargs):
		super(Specificity, self).__init__(name=name, **kwargs)
		self.tn = self.add_weight(name='tn', initializer='zeros')
		self.fp = self.add_weight(name='fp', initializer='zeros')
		self.fn = self.add_weight(name='fn', initializer='zeros')
		self.tp = self.add_weight(name='tp', initializer='zeros')

	def update_state(self, y_true, y_pred, sample_weight=None):
		
		y_pred = tf.round(y_pred)
		y_pred = tf.cast(y_pred, tf.bool)
		
		y_true = tf.cast(y_true, tf.bool)

		values_tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
		values_tn = tf.cast(values_tn, self.dtype)
		values_fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))	
		values_fp = tf.cast(values_fp, self.dtype)
		#values_fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))	
		#values_fn = tf.cast(values_fn, self.dtype)
		#values_tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))	
		#values_tp = tf.cast(values_tp, self.dtype)
		if sample_weight is not None:
			sample_weight = tf.cast(sample_weight, self.dtype)
			sample_weight_tn = tf.broadcast_weights(sample_weight, values_tn)		
			sample_weight_fp = tf.broadcast_weights(sample_weight, values_fp)	
			#sample_weight_fn = tf.broadcast_weights(sample_weight, values_fn)		
			#sample_weight_tp = tf.broadcast_weights(sample_weight, values_tp)
			values_tn = tf.multiply(values_tn, sample_weight_tn)
			values_fp = tf.multiply(values_fp, sample_weight_fp)
			#values_fn = tf.multiply(values_fn, sample_weight_fn)
			#values_tp = tf.multiply(values_tp, sample_weight_tp)
		self.tn.assign_add(tf.reduce_sum(values_tn))
		self.fp.assign_add(tf.reduce_sum(values_fp))
		#self.fn.assign_add(tf.reduce_sum(values_fn))
		#self.tp.assign_add(tf.reduce_sum(values_tp))

	def result(self):
		return self.tn / (self.tn + self.fp)

"""
 
class BalancedAccuracy(tf.keras.metrics.Metric):

	def __init__(self, name='bal_acc', **kwargs):
		super(BalancedAccuracy, self).__init__(name=name, **kwargs)
		self.tn = self.add_weight(name='tn', initializer='zeros')
		self.fp = self.add_weight(name='fp', initializer='zeros')
		self.fn = self.add_weight(name='fn', initializer='zeros')
		self.tp = self.add_weight(name='tp', initializer='zeros')

	def update_state(self, y_true, y_pred, sample_weight=None):
		
		y_pred = tf.round(y_pred)
		y_pred = tf.cast(y_pred, tf.bool)
		
		y_true = tf.cast(y_true, tf.bool)

		values_tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
		values_tn = tf.cast(values_tn, self.dtype)
		values_fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))	
		values_fp = tf.cast(values_fp, self.dtype)
		values_fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))	
		values_fn = tf.cast(values_fn, self.dtype)
		values_tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))	
		values_tp = tf.cast(values_tp, self.dtype)
		if sample_weight is not None:
			sample_weight = tf.cast(sample_weight, self.dtype)
			sample_weight_tn = tf.broadcast_weights(sample_weight, values_tn)		
			sample_weight_fp = tf.broadcast_weights(sample_weight, values_fp)	
			sample_weight_fn = tf.broadcast_weights(sample_weight, values_fn)		
			sample_weight_tp = tf.broadcast_weights(sample_weight, values_tp)
			values_tn = tf.multiply(values_tn, sample_weight_tn)
			values_fp = tf.multiply(values_fp, sample_weight_fp)
			values_fn = tf.multiply(values_fn, sample_weight_fn)
			values_tp = tf.multiply(values_tp, sample_weight_tp)
		self.tn.assign_add(tf.reduce_sum(values_tn))
		self.fp.assign_add(tf.reduce_sum(values_fp))
		self.fn.assign_add(tf.reduce_sum(values_fn))
		self.tp.assign_add(tf.reduce_sum(values_tp))

	def result(self):
		sens = self.tp / (self.tp + self.fn)
		spec = self.tn / (self.tn + self.fp)
		return (sens + spec)/2

"""

class GMean(tf.keras.metrics.Metric):

	def __init__(self, name='gmean', **kwargs):
		super(GMean, self).__init__(name=name, **kwargs)
		self.tn = self.add_weight(name='tn', initializer='zeros')
		self.fp = self.add_weight(name='fp', initializer='zeros')
		self.fn = self.add_weight(name='fn', initializer='zeros')
		self.tp = self.add_weight(name='tp', initializer='zeros')

	def update_state(self, y_true, y_pred, sample_weight=None):
		
		y_pred = tf.round(y_pred)
		y_pred = tf.cast(y_pred, tf.bool)
		
		y_true = tf.cast(y_true, tf.bool)

		values_tn = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
		values_tn = tf.cast(values_tn, self.dtype)
		values_fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))	
		values_fp = tf.cast(values_fp, self.dtype)
		values_fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))	
		values_fn = tf.cast(values_fn, self.dtype)
		values_tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))	
		values_tp = tf.cast(values_tp, self.dtype)
		if sample_weight is not None:
			sample_weight = tf.cast(sample_weight, self.dtype)
			sample_weight_tn = tf.broadcast_weights(sample_weight, values_tn)		
			sample_weight_fp = tf.broadcast_weights(sample_weight, values_fp)	
			sample_weight_fn = tf.broadcast_weights(sample_weight, values_fn)		
			sample_weight_tp = tf.broadcast_weights(sample_weight, values_tp)
			values_tn = tf.multiply(values_tn, sample_weight_tn)
			values_fp = tf.multiply(values_fp, sample_weight_fp)
			values_fn = tf.multiply(values_fn, sample_weight_fn)
			values_tp = tf.multiply(values_tp, sample_weight_tp)
		self.tn.assign_add(tf.reduce_sum(values_tn))
		self.fp.assign_add(tf.reduce_sum(values_fp))
		self.fn.assign_add(tf.reduce_sum(values_fn))
		self.tp.assign_add(tf.reduce_sum(values_tp))

	def result(self):
		sens = self.tp / (self.tp + self.fn)
		spec = self.tn / (self.tn + self.fp)
		return tf.sqrt(sens * spec)


# LRFinder callback based on:
# https://github.com/WittmannF/LRFinder/blob/master/keras_callback.py
# https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/

"""

class LRFinder(tf.keras.callbacks.Callback):
	def __init__(self, lr_start, lr_end, beta=0.9, stop_multiplier=None, reload_weights=True):
		self.lr_start = lr_start
		self.lr_end = lr_end
		self.beta = beta
		if stop_multiplier is None:
			self.stop_multiplier = -20*beta/3 + 10	# 4  if beta = 0.9
								# 10 if beta = 0 
		else:
			self.stop_multiplier = stop_multiplier
		self.reload_weights = reload_weights
		super(LRFinder, self).__init__()
	def on_train_begin(self, logs=None):
		self.n_iters = self.params['epochs'] * self.params['steps']
		self.iter = 0
		self.lrs = [] # Learning rates list.
		self.losses = [] # Losses list.

		#Let t the iteration index (iter) and m the learning rate multiplier that we want to find out.

		#t=0: lr = lr_start
		#t=1: lr = m * lr_start
		#t=2: lr = m^2 * lr_start
		#...
		#t=n: lr = m^n * lr_start = lr_end

		#Thus:

		#m = (lr_end/lr_start) ^ (1/n) = exp((-1/n) * (ln(lr_end) - ln(lr_start))).

		self.m = np.exp((1 / (self.n_iters - 1)) * (np.log(self.lr_end) - np.log(self.lr_start)))
		if self.reload_weights:
			self.model.save_weights('tmp.h5')
	def on_train_batch_begin(self, batch, logs=None):
		self.model.optimizer.learning_rate = (self.m ** self.iter) * self.lr_start
		self.lrs.append(self.model.optimizer.learning_rate.numpy())
	def on_train_batch_end(self, batch, logs=None):
		self.iter += 1
		loss = logs['loss']
		if self.iter > 1:
			loss = self.beta * self.losses[-1] + (1 - self.beta) * loss
		if self.iter == 1 or loss < self.best_loss:
			self.best_loss = loss
		if loss > self.stop_multiplier * self.best_loss: # Stop criteria. 
			self.higher_loss = self.stop_multiplier * self.best_loss
			self.model.stop_training = True
		self.losses.append(loss/(1 - self.beta ** self.iter))
	def on_train_end(self, logs=None):
		self.model.history.history.clear()
		self.model.history.history['lrs'] = self.lrs
		self.losses = [self.higher_loss if loss > self.higher_loss else loss for loss in self.losses]
		self.model.history.history['losses'] = self.losses
		fig = plt.figure()
		plt.plot(self.lrs, self.losses)
		plt.title('Learning rate finder')
		plt.xlabel('Learning rate')
		plt.ylabel('Loss')
		plt.xscale('log')
		plt.grid(True)
		plt.savefig('lr_finder.png')
		plt.close(fig)
		if self.reload_weights:
			self.model.load_weights('tmp.h5')

"""

storage_path = '/mnt/EES-Babylon'

# ======================================
# LOADING PREPROCESSED IMAGES FROM DISK: 
# ======================================

print('Loading preprocessed images from disk...')

npzfiles_path = os.path.join(storage_path, 'NPZFILES')
npzfile_name = os.path.join(npzfiles_path, 'ISIC2019_images_vgg19.npz') 
npzdata = np.load(npzfile_name)
train_data = npzdata['train_data']
train_labels = npzdata['train_labels']
valid_data = npzdata['valid_data']
valid_labels = npzdata['valid_labels']

print('Done!')

# =========================================
# LOADING CONVNET'S ARCHITECTURE FROM DISK: 
# =========================================

print('Loading convnet\'s architecture from disk...')

jsonfiles_path = os.path.join(storage_path, 'JSONFILES')
jsonfile_name = os.path.join(jsonfiles_path, 'model.json')
jsonfile = open(jsonfile_name, 'r')
jsondata = jsonfile.read()
convnet = tf.keras.models.model_from_json(jsondata)
jsonfile.close()

print('Done!')

#=====================================
# LOADING CONVNET'S WEIGHTS FROM DISK:
#=====================================

print('Loading convnet\'s weights from disk...')

h5files_path = os.path.join(storage_path, 'H5FILES')
h5file_name = os.path.join(h5files_path, 'model.h5')
convnet.load_weights(h5file_name)

print('Done!')

# =======================
# CONVNET OLD EVALUATION:
# =======================

convnet.compile(
	loss=tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
	metrics=[GMean(), tf.keras.metrics.Recall(name='sens'), Specificity(name='spec')]
)

print('Convnet evaluation before step 4: ')

convnet.evaluate(valid_data, valid_labels)

#========
# STEP 4:
#========

print('The number os trainable weights before unfreeze some layers in the base network is', len(convnet.trainable_weights))

conv_base = convnet.layers[0]
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
	if layer.name == 'block5_conv1':
		set_trainable = True
	if set_trainable:
		layer.trainable = True
	else:
		layer.trainable = False

print('The number os trainable weights after unfreeze some layers in the base network is', len(convnet.trainable_weights))

#========
# STEP 5:
#========

batch_size = 32
initial_epoch = 0

# checks if there are prior executions of steps 4 and 5 and load the last one if that is the case. 
h5files = [file for file in sorted(os.listdir(h5files_path)) if file.startswith('weights-steps45')]
if len(h5files) > 0: # if there is at least one h5 file in the directory... 
	last_h5filename = os.path.join(h5files_path, h5files[-1])
	convnet.load_weights(last_h5filename)
	initial_epoch = int(last_h5filename[-5:-3])-1 # -1 to repeat the last execution.

convnet.compile(
	loss=tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.RMSprop(lr=1e-5),
	metrics=[GMean(), tf.keras.metrics.Recall(name='sens'), Specificity(name='spec')]
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(h5files_path, 'weights-steps45-{epoch:02d}.h5'), monitor='val_gmean', verbose=1, save_best_only=True, mode='max')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_gmean', patience=10, mode='max', restore_best_weights=True)
	
history = convnet.fit(
        x=train_data, 
        y=train_labels,
        batch_size=batch_size,
        epochs=100,
	verbose=2,
	callbacks=[checkpoint_cb, early_stopping_cb],
        validation_data=(valid_data, valid_labels),
	initial_epoch=initial_epoch
)

# =======================
# CONVNET NEW EVALUATION:
# =======================

print('Model evaluation after step 5:')

convnet.evaluate(valid_data, valid_labels)
