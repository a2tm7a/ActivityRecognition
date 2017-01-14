import h5py
import numpy as np
np.random.seed(1337)
from keras.optimizers import SGD
import logging

# In[2]:

h5f = h5py.File('../preprocessed_data_winSize100_winShift10.h5','r')
training_data = h5f['training_data'][:]
training_output = h5f['training_output'][:]
testing_data = h5f['testing_data'][:]
testing_output = h5f['testing_output'][:]

h5f = h5py.File('../preprocessed_data_winSize100_winShift10_mean_sd.h5','r')
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


print(training_data.shape,training_output.shape)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


# In[11]:
nb_classes = 7

training_output = np_utils.to_categorical(training_output, nb_classes)
testing_s = np_utils.to_categorical(testing_output, nb_classes)
print(training_output.shape)


batch_size = 512

nb_epoch = 10
nb_neg_cycles = 3

print(training_data.shape)

logging.basicConfig(filename= "./WSDM_Keras_batch_size"+"("+str(batch_size)+")"+".log", level=logging.INFO, format='%(message)s')
logging.info("Parameters")


model = Sequential()


model.add(Dense(1024, input_shape=(training_data.shape[1],)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
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

    model.fit(training_data, training_output, batch_size=batch_size, nb_epoch=1,
              verbose=1, validation_data=(testing_data, testing_output))
    score = model.evaluate(testing_data, testing_output, verbose=0)
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