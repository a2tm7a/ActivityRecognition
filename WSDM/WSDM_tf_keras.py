from __future__ import print_function
import h5py
import numpy as np
import logging

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD

np.random.seed(1337)

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


# # In[16]:

train_test_split = np.random.rand(len(segments)) < 0.70
print(train_test_split)


# In[17]:

train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]
print(train_x.shape,train_y.shape)


# train_x = segments[train_test_split]
# train_y = labels[train_test_split]
# test_x = segments[~train_test_split]
# test_y = labels[~train_test_split]
# print(train_x.shape,train_y.shape)



batch_size = 512
nb_classes = 6
nb_epoch = 10
nb_neg_cycles = 3

# input image dimensions
img_rows, img_cols = 90, 3
# number of convolutional filters to use

nb_filters = 180
# size of pooling area for max pooling
pool_size = (20, 1)
# convolution kernel size
kernel_size = (1, 3)

# train_x=train_x.reshape(16956,270)
# test_x=test_x.reshape(test_x.shape[0],270)
print(train_x.shape)

logging.basicConfig(filename= "./WSDM_Keras_batch_size"+"("+str(batch_size)+")"+".log", level=logging.INFO, format='%(message)s')
logging.info("Parameters")


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size,strides=(2,1)))

# model.add(Convolution2D(1080, 6, 1,
#                         border_mode='valid'))
# model.add(Activation('relu'))
# #model.add(MaxPooling2D(pool_size=pool_size,strides=(2,1)))

model.add(Flatten())

# model.add(Dense(6, input_shape=(train_x.shape[1],)))
# model.add(Dense(128))
# model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

logging.info(str(model.summary()))

epoch = 1

final_acc=0.0
lr = 0.0001
weights=[]
first_run=True
flag=True
iter_count = 1
count_neg_iter=0

#The flag to keep the loop running
run_flag=True
weights=[]

#Check if it is the first iteration
first_iter=True

while run_flag:

    #Give ramdom weights in first iteration and previous weights to other
    if first_iter:
        first_iter=False
    else:
        model.set_weights(np.asarray(weights))


    logging.info("\n\nIteration:"+str(iter_count))
    print ("Learning Rate : ",lr)
    logging.info("Learning Rate: "+str(lr))
    
    sgd = SGD(lr=lr)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=1,
              verbose=1, validation_data=(test_x, test_y))
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    current_acc = score[1]

    logging.info(current_acc)

    if(current_acc - final_acc > 0):        
        iter_count = iter_count + 1

        #Update the weights if the accuracy is greater than .001
        weights=model.get_weights()
        print ("Updating the weights")
        logging.info("Updating the weights")
        #Updating the final accuracy
        final_acc=current_acc
        #Setting the count to 0 again so that the loop doesn't stop before reducing the learning rate n times consecutivly
        count_neg_iter = 0

    else:
        #If the difference is not greater than 0.005 reduce the learning rate
        lr=lr/2.0
        print ("Reducing the learning rate by half")
        logging.info("Reducing the learning rate by half")
        count_neg_iter = count_neg_iter + 1
        
        #If the learning rate is reduced consecutively for nb_neg_cycles times then the loop should stop
        if(count_neg_iter>nb_neg_cycles):
            run_flag=False
            model.set_weights(np.asarray(weights))

print('Final accuracy:', final_acc)

logging.info(final_acc)