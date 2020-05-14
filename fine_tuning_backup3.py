import os
import random as rnd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
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

rnd.seed(5489)
np.random.seed(5489)
tf.random.set_seed(5489)

print('Working with images of melanoma (class 1) and nevus (class 0) classes from ISIC 2019 dataset...')

npzfile = np.load('/mnt/EES-Babylon/MAT_FILES/ISIC2019_images_vgg19.npz')
train_labels = npzfile['train_labels']
valid_labels = npzfile['valid_labels']

print('Training samples from class 0: ', np.sum(train_labels==0))
print('Training samples from class 1: ', np.sum(train_labels==1))
print('Validation samples from class 0: ', np.sum(valid_labels==0))
print('Validation samples from class 1: ', np.sum(valid_labels==1))

print('Loading data from npz file...')

train_data = npzfile['train_data']
valid_data = npzfile['valid_data']

print('Done!')

print('Shape of train_data: ', train_data.shape)
print('Shape of train_labels: ', train_labels.shape)
print('Shape of valid_data: ', valid_data.shape)
print('Shape of valid_labels: ', valid_labels.shape)

"""

# Saving some train and valid images. 

for i in range(5):
	train_random_idx = rnd.randint(0, train_data.shape[0]-1)
	train_sample_img = tf.keras.preprocessing.image.array_to_img(train_data[train_random_idx])
	plt.imshow(train_sample_img)
	plt.title('Train image ' + str(train_random_idx) + ', Label = ' + str(train_labels[train_random_idx]))
	plt.savefig('train_sample_img' + str(i+1) + '.png')
	
for i in range(5):
        valid_random_idx = rnd.randint(0, valid_data.shape[0]-1)
        valid_sample_img = tf.keras.preprocessing.image.array_to_img(valid_data[valid_random_idx])
        plt.imshow(valid_sample_img)
        plt.title('Valid image ' + str(valid_random_idx) + ', Label = ' + str(valid_labels[valid_random_idx]))
        plt.savefig('valid_sample_img' + str(i+1) + '.png')

"""

conv_base = tf.keras.applications.VGG19(
	weights='imagenet',
	include_top=False,
	input_shape=(224,224,3))

#conv_base.summary()			

model = tf.keras.models.Sequential()
model.add(conv_base)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print('This is the number os trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights after freezing the conv base', len(model.trainable_weights))

#model.summary()

batch_size = 32

# Begin of: Searching for the optimal leraning rate

"""

model.compile(
	#loss='binary_crossentropy',
	loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=1, gamma=0),
	optimizer=tf.keras.optimizers.SGD()
)

epochs_findlr = 3

history = model.fit(
	x=train_data,
	y=train_labels,
	epochs=epochs_findlr,
	verbose=0,
	batch_size=batch_size,
	validation_data=None,
	callbacks=[LRFinder(1e-5, 1)]
)

lrs = history.history['lrs']
losses = history.history['losses']
pos_min = np.argmin(losses)
ideal_lr = lrs[pos_min]/10

print('Ideal learning rate = {:.10f}'.format(ideal_lr))

"""
	
# End of: Searching for the optimal learning rate. 

# Begin of: Training:

model.compile(
	#loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=1, gamma=0),
	loss=tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
	metrics=[GMean(), tf.keras.metrics.Recall(name='sens'), Specificity(name='spec')]
)

#nv_weight = np.sum(train_labels==0)/train_labels.shape[0]
#mel_weight = np.sum(train_labels==1)/train_labels.shape[0]
#class_weight = {0:nv_weight,1:mel_weight}

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_gmean', patience=10, mode='max', restore_best_weights=True)

history = model.fit(
        x=train_data, 
        y=train_labels,
        batch_size=batch_size,
        epochs=30,
	verbose=2,
        validation_data=(valid_data,valid_labels),
	#class_weight=class_weight,
	callbacks=[early_stopping_cb]
)

# End of: Training

print('Better result on the validation set:')
model.evaluate(valid_data, valid_labels)

"""

END----------------------------------------------------------------------------------------------------------------

train_data = tf.keras.preprocessing.image_dataset_from_directory(
	train_dir, 
	labels='inferred',
	label_mode='binary',
	class_names=None,
	color_mode='rgb',
	batch_size=32,
	image_size=(224,224),
	shuffle=True,
	seed=5489,
	validation_split=None,
	subset=None,
	interpolation='bilinear',
	follow_links=False
)

valid_data = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir, 
        labels='inferred',
        label_mode='binary',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(224,224),
        shuffle=True,
        seed=5489,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False
)
"""


