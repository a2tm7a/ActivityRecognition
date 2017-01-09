
# coding: utf-8

# In[1]:

import h5py
import numpy as np
np.random.seed(1337)


# In[2]:

h5f = h5py.File('preprocessed_data_winSize100_winShift10.h5','r')
training_data = h5f['training_data'][:]
training_output = h5f['training_output'][:]
testing_data = h5f['testing_data'][:]
testing_output = h5f['testing_output'][:]

print(training_data.shape,training_output.shape,testing_data.shape,testing_output.shape)

mean = np.mean(training_data,axis=0)
sd = np.std(training_data,axis=0)
import h5py
h5f = h5py.File('preprocessed_data_winSize100_winShift10_mean_sd.h5', 'w')
h5f.create_dataset('mean', data=mean)
h5f.create_dataset('sd', data=sd)
h5f.close()


# In[17]:

h5f = h5py.File('preprocessed_data_winSize100_winShift10_mean_sd.h5','r')
mean = h5f['mean'][:]
sd = h5f['sd'][:]
h5f.close()
print(mean,sd)
