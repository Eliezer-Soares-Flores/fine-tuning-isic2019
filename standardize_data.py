import os, shutil
import pandas as pd
import numpy as np

"""

This is just an adaptation of that script presented in page 132 of the Chollet's book, Deep Learning with Python. 
Here, the goal is to produce a 'standardized' organization of directories/subdirectories for the 'Melanoma' and 'Nevus' images from the ISIC 2019 dataset by using a stratified holdout partitioning, with approximately 80% for training and the remainder for validation.  

Ps.: You can use the Linux command 'ls | wc -l' to verify the number of images contained in a given directory. 

"""

original_dataset_dir = '/mnt/EES-Babylon/ISIC_2019_Training_Input'

base_dir = '/mnt/EES-Babylon/ISIC_2019_Standardized'
if not os.path.exists(base_dir):
	os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
	os.mkdir(train_dir)

valid_dir = os.path.join(base_dir, 'valid')
if not os.path.exists(valid_dir):
	os.mkdir(valid_dir)

train_mel_dir = os.path.join(train_dir, 'mal')
if not os.path.exists(train_mel_dir):
	os.mkdir(train_mel_dir)

train_nv_dir = os.path.join(train_dir, 'ben')
if not os.path.exists(train_nv_dir):
	os.mkdir(train_nv_dir)

valid_mel_dir = os.path.join(valid_dir, 'mal')
if not os.path.exists(valid_mel_dir):
	os.mkdir(valid_mel_dir)

valid_nv_dir = os.path.join(valid_dir, 'ben')
if not os.path.exists(valid_nv_dir):
	os.mkdir(valid_nv_dir)

# Selecting only Melanoma and Nevi samples:

df = pd.read_csv('/mnt/EES-Babylon/ISIC_2019_Training_GroundTruth.csv')
df_mel = df.loc[(df['MEL']==1)]
df_nv = df.loc[(df['NV']==1)]

n_mel_imgs = df_mel.shape[0]
n_nv_imgs = df_nv.shape[0]
n_imgs = n_mel_imgs + n_nv_imgs

np.random.seed(5489)
random_mel_idxs = np.random.permutation(n_mel_imgs)
random_nv_idxs = np.random.permutation(n_nv_imgs)

sep_mel_pos = int(0.8*n_mel_imgs)
sep_nv_pos = int(0.8*n_nv_imgs)

train_mel_idxs = random_mel_idxs[:sep_mel_pos]
valid_mel_idxs = random_mel_idxs[sep_mel_pos:]

train_nv_idxs = random_nv_idxs[:sep_nv_pos]
valid_nv_idxs = random_nv_idxs[sep_nv_pos:]

count = 0

for idx in train_mel_idxs:
	fname = df_mel.iloc[idx,0] + '.jpg'
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_mel_dir, fname)
	shutil.copyfile(src, dst)
	count+=1
	print('Image n.{:d} of {:d} successfully copied.'.format(count, n_imgs))

for idx in train_nv_idxs:
	fname = df_nv.iloc[idx,0] + '.jpg'
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_nv_dir, fname)
	shutil.copyfile(src, dst)	
	count+=1
	print('Image n.{:d} of {:d} successfully copied.'.format(count, n_imgs))

for idx in valid_mel_idxs:
	fname = df_mel.iloc[idx,0] + '.jpg'
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(valid_mel_dir, fname)
	shutil.copyfile(src, dst)	
	count+=1
	print('Image n.{:d} of {:d} successfully copied.'.format(count, n_imgs))

for idx in valid_nv_idxs:
	fname = df_nv.iloc[idx,0] + '.jpg'
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(valid_nv_dir, fname)
	shutil.copyfile(src, dst)
	count+=1
	print('Image n.{:d} of {:d} successfully copied.'.format(count, n_imgs))

