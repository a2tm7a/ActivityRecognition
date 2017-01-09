
# coding: utf-8

# In[1]:

import h5py
import numpy as np
np.random.seed(1337)
from keras.optimizers import SGD


# In[2]:

h5f = h5py.File('preprocessed_data_winSize100_winShift10.h5','r')
training_data = h5f['training_data'][:]
training_output = h5f['training_output'][:]
testing_data = h5f['testing_data'][:]
testing_output = h5f['testing_output'][:]

h5f = h5py.File('preprocessed_data_winSize100_winShift10_mean_sd.h5','r')
mean = h5f['mean'][:]
sd = h5f['sd'][:]
h5f.close()

print(training_data.shape,training_output.shape,testing_data.shape,testing_output.shape)

print("Concatenating training")
training = np.concatenate([training_data, training_output.reshape(training_output.shape[0],1)], axis=1)
print("Deleting training")
del training_data
del training_output
print("Shuffling training")
np.random.shuffle(training)

print("Concatenating testing")
testing = np.concatenate([testing_data, testing_output.reshape(testing_output.shape[0],1)], axis=1)
print("Deleting testing")
del testing_data
del testing_output
print("Shuffling Testing")
np.random.shuffle(testing)

print("Getting back train data and output")
training_data=training[:,:-1]
training_output=training[:,-1]
print("Deleting training")
del training

print("Getting back test data and output")
testing_data=testing[:,:-1]
testing_output=testing[:,-1]
print("Deleting testing")
del testing

print(training_data.shape,training_output.shape,testing_data.shape,testing_output.shape)


# In[8]:

testing_data=testing_data-mean
training_data=training_data-sd


# In[17]:


batch_size = 512
nb_classes = 7
nb_epoch = 10
lr = 0.00005

sgd = SGD(lr=lr)

training_data = training_data.astype('float32')
testing_data = testing_data.astype('float32')


# In[10]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


# In[11]:

Y_train = np_utils.to_categorical(training_output, nb_classes)
Y_test = np_utils.to_categorical(testing_output, nb_classes)
print(Y_test.shape)


# In[18]:

model = Sequential()
model.add(Dense(512, input_shape=(training_data.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()


# In[19]:

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(training_data, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(testing_data, Y_test))
score = model.evaluate(testing_data, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:



