import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time

def get_dataset(dataset_path, target_net):

	if target_net.casefold() == 'densenet121' or target_net.casefold() == 'densenet169' or target_net.casefold() == 'densenet201':
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.densenet import preprocess_input
	elif target_net.casefold() == 'efficientnetb0':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb1':	
		img_rows, img_cols, img_chs = 240, 240, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb2':	
		img_rows, img_cols, img_chs = 260, 260, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb3':	
		img_rows, img_cols, img_chs = 300, 300, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb4':	
		img_rows, img_cols, img_chs = 380, 380, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb5':	
		img_rows, img_cols, img_chs = 456, 456, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb6':	
		img_rows, img_cols, img_chs = 528, 528, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'efficientnetb7':	
		img_rows, img_cols, img_chs = 600, 600, 3
		from tensorflow.keras.applications.efficientnet import preprocess_input	
	elif target_net.casefold() == 'inceptionresnetv2':	
		img_rows, img_cols, img_chs = 299, 299, 3
		from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
	elif target_net.casefold() == 'inceptionv3':	
		img_rows, img_cols, img_chs = 299, 299, 3
		from tensorflow.keras.applications.inception_v3 import preprocess_input
	elif target_net.casefold() == 'mobilenet':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.mobilenet import preprocess_input
	elif target_net.casefold() == 'mobilenetv2':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
	elif target_net.casefold() == 'nasnetlarge':	
		img_rows, img_cols, img_chs = 331, 331, 3
		from tensorflow.keras.applications.nasnet import preprocess_input		
	elif target_net.casefold() == 'nasnetmobile':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.nasnet import preprocess_input	
	elif target_net.casefold() == 'resnet50' or target_net.casefold() == 'resnet101' or target_net.casefold() == 'resnet152':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.resnet import preprocess_input
	elif target_net.casefold() == 'resnet50v2':
		img_rows, img_cols, img_chs = 299, 299, 3
		from tensorflow.keras.applications.resnet_v2 import preprocess_input
	elif target_net.casefold() == 'resnet101v2' or target_net.casefold() == 'resnet152v2':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.resnet_v2 import preprocess_input
	elif target_net.casefold() == 'vgg16':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.vgg16 import preprocess_input
	elif target_net.casefold() == 'vgg19':	
		img_rows, img_cols, img_chs = 224, 224, 3
		from tensorflow.keras.applications.vgg19 import preprocess_input
	elif target_net.casefold() == 'xception':	
		img_rows, img_cols, img_chs = 299, 299, 3
		from tensorflow.keras.applications.xception import preprocess_input
	else:
		print('target_net does not match any of the options!')
		sys.exit()

	# Discovering the dataset size:
	folders_list = sorted(os.listdir(dataset_path))
	dataset_len = 0
	for folder_name in folders_list:
		folder_path = os.path.join(dataset_path, folder_name)
		imgs_list = os.listdir(folder_path)
		dataset_len += len(imgs_list)

	display_rows, display_columns = os.popen('stty size', 'r').read().split()
	print('='.center(int(display_columns), '='))
	print('Getting the images from the folder {}.'.format(dataset_path))
	print('These images will be reshaped to have {} rows, {} columns and {} channels as well as preprocessed to feed the DCNN {} properly.'.format(img_rows, img_cols, img_chs, target_net))	
	print('='.center(int(display_columns), '='))
	time.sleep(15)

	# Arranging dataset as a numpy array:
	X = np.zeros(shape=(dataset_len, img_rows, img_cols, img_chs), dtype='f4')
	labels = np.zeros(shape=(dataset_len,))
	dataset_idx = 0
	label = 0
	for folder_name in folders_list:
		print('='.center(int(display_columns), '='))
		print('\'{}\' folder associated with label {}.'.format(folder_name, label))
		print('='.center(int(display_columns), '='))
		time.sleep(10)
		folder_path = os.path.join(dataset_path, folder_name)
		imgs_list = sorted(os.listdir(folder_path))
		for img_name in imgs_list:
			print('Working on image {} of {}...'.format(dataset_idx + 1, dataset_len))
			img_full_name = os.path.join(folder_path, img_name)
			img = load_img(img_full_name, target_size=(img_rows, img_cols))
			x = img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			X[dataset_idx,:,:,:] = x
			labels[dataset_idx] = label
			dataset_idx += 1
		label += 1 

	print('Done!')
	return X, labels

# MAIN CODE:

target_net = 'vgg19'
dst_dir = '/mnt/EES-Babylon/MAT_FILES'

train_data, train_labels = get_dataset('/mnt/EES-Babylon/ISIC_2019_Standardized/train', target_net)
valid_data, valid_labels = get_dataset('/mnt/EES-Babylon/ISIC_2019_Standardized/valid', target_net)

np.savez(os.path.join(dst_dir, 'ISIC2019_images_' + target_net), train_data=train_data, train_labels=train_labels, valid_data=valid_data, valid_labels=valid_labels)
