#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[23]:


file = open('/home/ashiq/Documents/MCA/s4/Project/AMBTMD_latest/bike_bus.csv')
lines = file.readlines()

processedList5 = []

for i, line in enumerate(lines):
    try:
        line = line.strip()
        line = line.split(',')
        last = line[4]
        last = last.strip()
        if last == '':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4]]
        processedList5.append(temp)
    except:
        print('Error at line number: ', i)


# In[24]:


processedList5


# In[25]:


columns = ['time','x','y','z','target']


# In[26]:


data = pd.DataFrame(data=processedList5, columns=columns)


# In[30]:


data.head()


# In[31]:


data.shape


# In[32]:


data.info()


# In[33]:


data.isnull().sum()


# In[34]:


data['target'].value_counts()


# In[35]:


###Balance the above data


# In[36]:


data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')


# In[37]:


data.info()


# In[38]:


Fs = 20


# In[39]:


targets = data['target'].value_counts().index


# In[40]:


targets


# In[43]:


def plot_target(target, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(target)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for target in targets:
    data_for_plot = data[(data['target'] == target)][:Fs*10]
    plot_target(target, data_for_plot)


# In[57]:


df = data.drop(['time'], axis=1).copy()
df.head()
Bus = df[df['target']=='Bus'].head(3555).copy()
Bike = df[df['target']=='Bike'].head(3555).copy()


# In[58]:


Bus


# In[59]:


Bike


# In[60]:


balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Bus, Bike])
balanced_data.shape


# In[61]:


balanced_data['target'].value_counts()


# In[62]:


balanced_data.head()


# In[63]:


from sklearn.preprocessing import LabelEncoder


# In[64]:


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['target'])
balanced_data.head(7110)


# In[65]:


label.classes_


# In[55]:


##### standardize above data 


# In[66]:


x = balanced_data[['x','y','z']]
y = balanced_data['label']


# In[67]:


scaler = StandardScaler()
x = scaler.fit_transform(x)

scaled_x = pd.DataFrame(data = x, columns = ['x', 'y', 'z'])
scaled_x['label'] = y.values

scaled_x


# In[68]:


#### Frame Preperation


# In[69]:


import scipy.stats as stats


# In[70]:


Fs = 20
frame_size = Fs*4 
hop_size = Fs*2


# In[71]:


def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels


# In[72]:


x, y = get_frames(scaled_x, frame_size, hop_size)


# In[73]:


x.shape, y.shape


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# In[75]:


x_train.shape, x_test.shape


# In[76]:


x_train[0].shape, x_test[0].shape


# In[78]:


x_train = x_train.reshape(140, 80, 3, 1)
x_test = x_test.reshape(36, 80, 3, 1)


# In[79]:


x_train[0].shape, x_test[0].shape


# In[80]:


### 2D CNN model 


# In[81]:


model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation = 'relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation = 'softmax'))


# In[82]:


model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[83]:


history = model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), verbose=1)


# In[84]:


def plot_learningCurve(history, epochs):
  # Plot training & validation accuracy values
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()


# In[85]:


plot_learningCurve(history, 10)


# In[86]:


###Confusion Matrix


# In[87]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[88]:


y_pred = model.predict_classes(x_test)


# In[89]:


mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))


# In[ ]:




