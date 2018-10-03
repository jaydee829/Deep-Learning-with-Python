import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

max_features=2000
max_len=500

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
x_train=sequence.pad_sequences(x_train, maxlen=max_len)
x_test=sequence.pad_sequences(x_test, maxlen=max_len)

model=keras.models.Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len,name='embed'))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

callbacks=[keras.callbacks.TensorBoard(log_dir='my_log_dir',histogram_freq=1,)]

history=model.fit(x_train,y_train,epochs=20,batch_size=128,validation_split=0.2,callbacks=callbacks)
