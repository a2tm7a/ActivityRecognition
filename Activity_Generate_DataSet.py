
# coding: utf-8


import pandas as pd
import numpy as np


# Read all the participants and sort them
files=[]
import os
for file in os.listdir("./Dataset"):
    if file.endswith(".csv"):
        files.append(file)
files.sort()
print(files)
    


# FORMATTING THE DATA
mapping = {'walking': 0, 'standing': 1, 'jogging': 2, 'sitting': 3, 'biking': 4, 'upstairs': 5, 'downstairs': 6}

training_data=None
training_output=None
testing_data=None
testing_output=None

# No of training participants
ratio=7

count=0
for file in files:
    count+=1
    df = pd.read_csv('./Dataset/'+file, index_col=False, header=1);  
    #print(df.describe())
    print(file)
    data=None
    for i in range(1,3):
        dff=df.ix[:,14*i-13:14*i-10]
        dff['activity'] = df['Activity']
        
        dff=dff.replace({'activity': mapping})

        if data is None:
            data=dff.as_matrix()
        else:
            data=np.append(data,dff.as_matrix(),axis=0)

    print(data.shape)
    #print(data)
    
    linear_data_x=None
    linear_data_y=None
    window_size=100
    window_shift=10
    label = 0
    index=0
    data_flag = False
    
    print("Clubbing data")
    while data_flag!=True:
        while int(data[index+window_size -1][3]) == int(label) and data_flag!=True:

            if linear_data_x is None:
                linear_data_x = data[index:index+window_size,:3].ravel()
                linear_data_y = int(label)
                
            else:
                linear_data_x = np.vstack((linear_data_x,data[index:index+window_size,:3].ravel()))
                linear_data_y = np.hstack((linear_data_y,int(label)))
                #print(index,index+window_size-1,int(label))
            index+=window_shift
            
            if index>= data.shape[0] or index+window_size>=data.shape[0]:
                data_flag=True

                break
        
        label+=1
        print(index,label)
        #print(label)
        
        if int(label) == 7:
            label=0
        
        #print(label,data[index][12])
        while int(data[index][3]) != int(label):
            #print(index,int(data[index][12]),int(label))
            index+=1
            if index >= data.shape[0]:
                data_flag = True
                break 
    
    #print(count)
    if count<=ratio:
        if training_data is None:
            training_data=linear_data_x
            training_output=linear_data_y
        else:
            training_data=np.vstack((training_data,linear_data_x))
            training_output=np.append(training_output,linear_data_y)
    else:
        if testing_data is None:
            testing_data=linear_data_x
            testing_output=linear_data_y
        else:
            testing_data=np.vstack((testing_data,linear_data_x))
            testing_output=np.append(testing_output,linear_data_y)

    print(training_data.shape,training_output.shape)
    if testing_data is not None:
        print(testing_data.shape,testing_output.shape)


import h5py
h5f = h5py.File('preprocessed_data_winSize100_winShift10.h5', 'w')
h5f.create_dataset('training_data', data=training_data)
h5f.create_dataset('testing_data', data=testing_data)
h5f.create_dataset('training_output', data=training_output)
h5f.create_dataset('testing_output', data=testing_output)


h5f.close()

h5f = h5py.File('preprocessed_data_winSize100_winShift10.h5','r')
b = h5f['testing_data'][:]
print(b.shape)

print(np.mean(training_data,axis=0),np.std(training_data,axis=0))
print(np.mean(testing_data,axis=0),np.std(testing_data,axis=0))