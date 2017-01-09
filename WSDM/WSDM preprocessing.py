from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')
from keras import backend as K


# In[123]:

def read_data(file_path):
    column_names = ['user-id','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path,header = None, names = column_names)
    #print(data)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma
    
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


# In[89]:

dataset = read_data('/home/amit/Papers/Activity based recognition/DataSet/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
#print(dataset)


# In[90]:

print("Normalising x")
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
print("Normalising y")
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
print("Normalising z")
dataset['z-axis'] = feature_normalize(dataset['z-axis'])


# In[93]:

for activity in np.unique(dataset["activity"]):
    subset = dataset[dataset["activity"] == activity][:180]
    plot_activity(activity,subset)


# In[96]:

def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += int(size / 2)
        
def segment_signal(data,window_size = 90):
    segments = np.empty((0,window_size,3))
    #segments = np.empty((0,270))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(start%1000==0):
            print(start)
            #print(np.dstack([x,y,z]).ravel())
        if(len(dataset["timestamp"][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])
    return segments, labels


# In[ ]:




# In[97]:

segments, labels = segment_signal(dataset)


# In[101]:

print(segments.shape)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
import h5py
h5f = h5py.File('preprocessed_data_WSDM.h5', 'w')
h5f.create_dataset('data', data=segments)
h5f.create_dataset('output', data=labels)
h5f.close()


# In[2]: