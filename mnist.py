import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Based on:
# https://github.com/WittmannF/LRFinder/blob/master/keras_callback.py
# https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
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
		"""
		Let t the iteration index (iter) and m the learning rate multiplier that we want to find out.

		t=0: lr = lr_start
		t=1: lr = m * lr_start
		t=2: lr = m^2 * lr_start
		...
		t=n: lr = m^n * lr_start = lr_end

		Thus:

		m = (lr_end/lr_start) ^ (1/n) = exp((-1/n) * (ln(lr_end) - ln(lr_start))).

		"""
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

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#plt.imshow(X_test[0])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
model.add(tf.keras.layers.Dense(300, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))

#model.summary()
model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer=tf.keras.optimizers.SGD(),
	metrics=['accuracy'], 
)

nb_epochs = 3
batch_size = 32

history = model.fit(
	X_train, 
	y_train, 
	epochs=nb_epochs,
	verbose=0, 
	batch_size=batch_size,
	validation_data=None,
	callbacks=[LRFinder(1e-5, 1, 0.9)],
)
lrs = history.history['lrs']
losses =  history.history['losses']
pos_min = np.argmin(losses)
ideal_lr = lrs[pos_min]/10
print('Ideal learning rate = {:.10f}'.format(ideal_lr))

model.compile(
	loss='sparse_categorical_crossentropy',
	#optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_opt, rho=0.9, epsilon=1e-8),
	optimizer=tf.keras.optimizers.Adam(learning_rate=ideal_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
	#optimizer=tf.keras.optimizers.SGD(learning_rate=lr_opt, momentum=0.9, nesterov=True),
	#optimizer=tf.keras.optimizers.SGD(learning_rate=lr_opt),
	#optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
	metrics=['accuracy'],
)

history = model.fit(
	X_train,
	y_train,
	epochs=10,
	batch_size = 32,
	validation_data = (X_valid, y_valid),
)

model.evaluate(X_test, y_test)

