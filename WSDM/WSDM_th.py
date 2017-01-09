from __future__ import print_function
import h5py
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
h5f = h5py.File('preprocessed_data_WSDM.h5','r')
segments = h5f['data'][:]
labels= h5f['output'][:]
print(segments.shape, labels.shape)
h5f.close()


# In[15]:

print(len(segments))
print(labels.shape)

if K.image_dim_ordering() == 'th':
    reshaped_segments = segments.reshape(len(segments), 1,90, 3)
    input_shape = (1, 90, 3)
else:
    reshaped_segments = segments.reshape(len(segments),90, 3,1)
    input_shape = (90, 3, 1)

reshaped_segments = segments.reshape(len(segments), input_shape[0],input_shape[1],input_shape[2])


# In[16]:

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
print(train_test_split)


# In[17]:

train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]
print(train_x.shape,train_y.shape)


# In[ ]:



batch_size = 10
nb_classes = 6
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 90, 3
# number of convolutional filters to use
nb_filters = 180
# size of pooling area for max pooling
pool_size = (20, 1)
# convolution kernel size
kernel_size = (60, 3)

print(train_x.shape)
#input_shape=(1,90,3)


sgd = SGD(lr=0.0001)

accuracy=0
lr = 0.0001
weights=[]
first_run=True
flag=True


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,1)))

model.add(Convolution2D(1080, 6, 1,
                        border_mode='valid'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size,strides=(2,1)))

model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

epoch = 1


# In[ ]:

while flag:
    sgd = SGD(lr=lr)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=1,
              verbose=1, validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    if first_run:
        first_run=False
    
    if score[1] - accuracy > 0.005:
        epoch += 1
        # TODO: Save weights
        weights=model.
    else:
        # TODO: Copy previous weighs
        lr/=2


# In[ ]:




# In[ ]:




# In[ ]:



