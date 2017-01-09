from __future__ import print_function
import h5py
import numpy as np
h5f = h5py.File('preprocessed_data_WSDM.h5','r')
segments = h5f['data'][:]
labels= h5f['output'][:]
print(segments.shape, labels.shape)
h5f.close()


# In[ ]:




# In[3]:

print(len(segments))
print(labels.shape)

# if K.image_dim_ordering() == 'th':
#     reshaped_segments = segments.reshape(len(segments), 1,90, 3)
#     input_shape = (1, 90, 3)
# else:
#     reshaped_segments = segments.reshape(len(segments),90, 3,1)
#     input_shape = (90, 3, 1)

reshaped_segments = segments.reshape(len(segments), 1,90, 3)


# In[4]:

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
print(train_test_split)


# In[5]:

train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]
print(train_x.shape,train_y.shape)


# In[6]:

# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution1D, MaxPooling1D
# from keras.utils import np_utils
# from keras import backend as K

# batch_size = 128
# nb_classes = 6
# nb_epoch = 12

# # input image dimensions
# img_rows, img_cols = 90, 3
# # number of convolutional filters to use
# nb_filters = 32
# # size of pooling area for max pooling
# pool_size = (2, 2)
# # convolution kernel size
# kernel_size = (40, 3)



# In[7]:

# model = Sequential()

# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

# model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(test_x, test_y))
# score = model.evaluate(test_x, test_y, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# # batch_size = 512
# # nb_classes = 6
# # nb_epoch = 5


# In[8]:

# model = Sequential()
# model.add(Dense(100, input_shape=(270,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

# history = model.fit(train_x, train_y,
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=1, validation_data=(test_x, test_y))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


# In[9]:

input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 5

total_batchs = train_x.shape[0] // batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)
	
def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1, 1, 1, 1], padding='VALID')
	
def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))
    
def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], 
                          strides=[1, 1, stride_size, 1], padding='VALID')


# In[14]:

import tensorflow as tf


# In[15]:

X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])
print(type(X),X.get_shape())
c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
print(type(c),c.get_shape())
p = apply_max_pool(c,20,2)
print(type(p),p.get_shape())
c = apply_depthwise_conv(p,6,depth*num_channels,depth//10)
print(type(c),c.get_shape())

shape = c.get_shape().as_list()
print(shape)

c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])
print(c_flat.get_shape())

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)


# In[16]:

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[17]:

with tf.Session() as session:
    tf.initialize_all_variables().run()
    for epoch in range(training_epochs):
        cost_history = np.empty(shape=[1],dtype=float)
        for b in range(total_batchs):    
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)
        print("Epoch: ",epoch," Training Loss: ",c," Training Accuracy: ",
              session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
    
    print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))


# In[ ]:




# In[ ]:




# In[ ]:



