import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

"""

def balanced_accuracy(y_true, y_pred):
	import tensorflow as tf
	tp = tf.keras.metrics.TruePositives()
	tp.update_state(y_true, y_pred)
	tp_ = tp.result()
	tn = tf.keras.metrics.TrueNegatives()
	tn.update_state(y_true, y_pred)
	tn_ = tn.result()
	fp = tf.keras.metrics.FalsePositives()
	fp.update_state(y_true, y_pred)
	fp_ = fp.result()
	fn = tf.keras.metrics.FalseNegatives()
	fn.update_state(y_true, y_pred)
	fn_ = fn.result()
	p_ = tp_ + fn_
	n_ = tn_ + fp_
	return 0.5*(tp_/p_ + tn_/n_)

"""

train_dir = '/mnt/EES-Babylon/ISIC_2019_Standarized/train'
valid_dir = '/mnt/EES-Babylon/ISIC_2019_Standarized/valid'

train_data, train_labels = get_dataset(train_dir, 'vgg16')
valid_data, valid_labels = get_dataset(valid_dir, 'vgg16')

conv_base = VGG16(weights='imagenet',
	include_top=False,
	input_shape=(224,224,3))

#conv_base.summary()			

#conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

#model.summary()

#for layer in model.layers:
#	print(layer.name, ' -> trainable? ', layer.trainable)


#model.compile(loss='binary_crossentropy',
#	optimizer=optimizers.RMSprop(lr=2e-5),
#	metrics=['accuracy'])


from clr import LRFinder

num_samples = train_data.shape[0]
batch_size = 32
minimum_lr = 1e-6
maximum_lr = 1e-1

lr_callback = LRFinder(num_samples, batch_size,
                       minimum_lr, maximum_lr,
                       # validation_data=(X_val, Y_val),
                       lr_scale='exp', save_dir='/home/esflores/KFT_ISIC2019')

model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True))

model.fit(train_data, train_labels, epochs=1, batch_size=batch_size, callbacks=[lr_callback])

#train_labels[i] == 0 => melanoma
#train_labels[i] == 1 => nevi
#mel_weight = np.sum(train_labels==1)/train_labels.shape[0]
#nv_weight = np.sum(train_labels==0)/train_labels.shape[0]
#class_weight = {0:mel_weight,1:nv_weight}

"""

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
        x=train_data, 
	y=train_labels,
	batch_size=32,
        epochs=30,
        validation_data=(valid_data,valid_labels),
	callbacks=[early_stopping_cb]
)
#shuffle=False

#model.save('my_model.h5')


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig('my_training.png')


model.evaluate(valid_data, valid_labels)

"""

"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('accuracy.png')

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('loss.png')

"""

"""

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


