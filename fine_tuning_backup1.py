from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt

#tf.compat.v1.disable_eager_execution()

train_dir = '/mnt/EES-Babylon/ISIC_2019_Standarized/train'
valid_dir = '/mnt/EES-Babylon/ISIC_2019_Standarized/valid'

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(224,224),
	batch_size=64,
	class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(224,224),
        batch_size=64,
        class_mode='binary')

for data_batch, labels_batch in train_generator:
	print('data batch shape: ', data_batch.shape)
	print('labels batch shape: ', labels_batch.shape)
	break

conv_base = VGG16(weights='imagenet',
	include_top=False,
	input_shape=(224,224,3))

conv_base.summary()			

#conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#conv_base.trainable = False

for layer in model.layers:
	print(layer.name, ' -> trainable? ', layer.trainable)

model.compile(loss='binary_crossentropy',
	optimizer=optimizers.RMSprop(lr=2e-5),
	metrics=['acc'])

history = model.fit(
        train_generator, 
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=30,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples/valid_generator.batch_size,
	max_queue_size=10,
	workers=1,
	use_multiprocessing=False
)
#shuffle=False

#model.save('my_model.h5')

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

See pg 147. 

"""


