import os
import random as rnd
import numpy as np
import tensorflow as tf
#import tensorflow.keras.backend as K
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

rnd.seed(5489)
np.random.seed(5489)
tf.random.set_seed(5489)

storage_path = '/mnt/EES-Babylon'

# ==========================================
# LOADING THE PREPROCESSED IMAGES FROM DISK: 
# ==========================================

print('Working on Melanoma and Nevus images from ISIC 2019 dataset.')
print('Class 0 = Nevus, Class 1 = Melanoma.')

npzfiles_path = os.path.join(storage_path, 'NPZFILES')
npzfile_name = os.path.join(npzfiles_path, 'ISIC2019_images_vgg19.npz') 
npzdata = np.load(npzfile_name)

train_labels = npzdata['train_labels']
valid_labels = npzdata['valid_labels']

print('Training images from class 0: ', np.sum(train_labels==0))
print('Training images from class 1: ', np.sum(train_labels==1))
print('Validation images from class 0: ', np.sum(valid_labels==0))
print('Validation images from class 1: ', np.sum(valid_labels==1))

print('Loading preprocessed images from disk...')

train_data = npzdata['train_data']
valid_data = npzdata['valid_data']

print('Done!')

print('Shape of train_data: ', train_data.shape)
print('Shape of valid_data: ', valid_data.shape)

#==========================================
# LOADING THE ALREADY-TRAINED BASE NETWORK:
#==========================================

convbase = tf.keras.applications.VGG19(
	include_top=False,
	weights='imagenet',
	input_shape=(224,224,3),
	pooling='max',
)
#convbase.summary()			

#========
# STEP 1:
#========

convnet = tf.keras.models.Sequential()
convnet.add(convbase)
convnet.add(tf.keras.layers.Dense(256, activation='relu'))
convnet.add(tf.keras.layers.Dropout(0.5, input_shape=(256,)))
convnet.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#convnet.summary()

print('Saving convnet\'s architecture in the disk...')

jsonfiles_path = os.path.join(storage_path, 'JSONFILES')
jsonfname = os.path.join(jsonfiles_path, 'convnet.json')
jsondata = convnet.to_json()
with open(jsonfname, 'w') as jsonfile:
	jsonfile.write(jsondata)
	jsonfile.close()

print('Done!')

#========
# STEP 2:
#========

print('The number os trainable tensors before freezing the convbase part of convnet is', len(convnet.trainable_weights))
convbase.trainable = False
print('The number of trainable tensors after freezing the convbase part of convnet is', len(convnet.trainable_weights))

#========
# STEP 3:
#========

# Hyperparameters:
batch_size = 32
nb_epochs = 30

# The focal loss (reference paper: https://arxiv.org/pdf/1708.02002.pdf) is used in this training stage.
# In the reference paper, the authors suggest set the last layer bias to approximately -4.6 = -log((1-pi)/pi), with pi = 0.1. 
lastlayer_weights = convnet.layers[-1].get_weights()
lastlayer_weights[1] = np.array([-4.6], dtype='float32') # lastlayer_weights[1] = bias.
convnet.layers[-1].set_weights(lastlayer_weights)

# Checks if there are prior executions of steps 1-3 and load the last execution if that is the case. 
initial_epoch = 0
h5files_path = os.path.join(storage_path, 'H5FILES')
h5files = [file for file in sorted(os.listdir(h5files_path)) if file.startswith('weights-steps123')]
if len(h5files) > 0: # if there is at least one h5 file in the directory... 
	last_h5filename = os.path.join(h5files_path, h5files[-1])
	convnet.load_weights(last_h5filename)
	initial_epoch = int(last_h5filename[-5:-3])-1 # -1 to repeat the last execution.

convnet.compile(
	loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=2.0),
	optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
	metrics=[GMean(), tf.keras.metrics.Recall(name='sens'), Specificity(name='spec')]
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(h5files_path, 'weights-steps123-{epoch:02d}.h5'), monitor='val_gmean', verbose=1, save_best_only=True, mode='max')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_gmean', patience=10, mode='max', restore_best_weights=True)

history = convnet.fit(
        x=train_data, 
        y=train_labels,
        batch_size=batch_size,
        epochs=nb_epochs,
	verbose=2,
	callbacks=[checkpoint_cb, early_stopping_cb],
        validation_data=(valid_data, valid_labels),
	initial_epoch=initial_epoch,
)
