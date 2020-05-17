import tensorflow as tf
import os

json_fpath = '/mnt/EES-Babylon/JSONFILES'
h5_fpath = '/mnt/EES-Babylon/H5FILES_BACKUP'
json_fname = 'convnet.json'
h5_fname = 'weights-steps123-13.h5'

print('Loading convnet\'s architecture from disk...')

json_ffname = os.path.join(json_fpath, json_fname)
json_file = open(json_ffname, 'r')
json_data = json_file.read()
convnet = tf.keras.models.model_from_json(json_data)
json_file.close()

print('Done!')

print('Loading convnet\'s weights from disk...')

h5_ffname = os.path.join(h5_fpath, h5_fname)
convnet.load_weights(h5_ffname)

print('Done!')
