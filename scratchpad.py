import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

hello=tf.constant('Hello, TensorFlow!')
sess=tf.Session()
print(sess.run(hello))

import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import tensorflow as tf 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import tensorflow as tf
from keras import backend as K

num_cores = 4
CPU=True
GPU=False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session) 