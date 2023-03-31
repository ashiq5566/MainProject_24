#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow==2.0.0


# In[2]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[4]:


#pd.read_csv('/home/ashiq/Documents/MCA/s4/Project/human_activity_recoganition_cnn/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')


# In[5]:


file = open('/home/ashiq/Documents/MCA/s4/Project/human_activity_recoganition_cnn/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
lines = file.readlines()

processedList = []

for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        last = last.strip()
        if last == '':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedList.append(temp)
    except:
        print('Error at line number: ', i)


# In[6]:


processedList


# In[7]:


columns = ['user','activity','time','x','y','z']


# In[8]:


data = pd.DataFrame(data=processedList, columns=columns)


# In[9]:


data.head()


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


data['activity'].value_counts()


# In[14]:


###Balance the above data


# In[15]:


data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')


# In[16]:


data.info()


# In[17]:


Fs = 20


# In[18]:


activities = data['activity'].value_counts().index


# In[19]:


activities


# In[20]:


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in activities:
    data_for_plot = data[(data['activity'] == activity)][:Fs*10]
    plot_activity(activity, data_for_plot)


# In[21]:


df = data.drop(['user','time'], axis=1).copy()
df.head()


# In[22]:


Walking = df[df['activity']=='Walking'].head(3555).copy()
Jogging = df[df['activity']=='Jogging'].head(3555).copy()
Upstairs = df[df['activity']=='Upstairs'].head(3555).copy()
Downstairs = df[df['activity']=='Downstairs'].head(3555).copy()
Sitting = df[df['activity']=='Sitting'].head(3555).copy()  
Standing = df[df['activity']=='Standing'].copy()  


# In[23]:


balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
balanced_data.shape


# In[24]:


balanced_data['activity'].value_counts()


# In[25]:


balanced_data.head()


# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
balanced_data.head()


# In[28]:


label.classes_


# In[29]:


##### standardize above data 


# In[30]:


x = balanced_data[['x','y','z']]
y = balanced_data['label']


# In[31]:


scaler = StandardScaler()
x = scaler.fit_transform(x)

scaled_x = pd.DataFrame(data = x, columns = ['x', 'y', 'z'])
scaled_x['label'] = y.values

scaled_x


# In[32]:


#### Frame Preperation


# In[33]:


import scipy.stats as stats


# In[34]:


Fs = 20
frame_size = Fs*4 
hop_size = Fs*2


# In[35]:


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


# In[36]:


x, y = get_frames(scaled_x, frame_size, hop_size)


# In[37]:


x.shape, y.shape


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)


# In[39]:


x_train.shape, x_test.shape


# In[40]:


x_train[0].shape, x_test[0].shape


# In[41]:


x_train = x_train.reshape(425, 80, 3, 1)
x_test = x_test.reshape(107, 80, 3, 1)


# In[42]:


x_train[0].shape, x_test[0].shape


# In[43]:


### 2D CNN model 


# In[44]:


model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation = 'relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation = 'softmax'))


# In[45]:


model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[46]:


history = model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), verbose=1)


# In[47]:


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


# In[48]:


plot_learningCurve(history, 10)


# In[49]:


###Confusion Matrix


# In[50]:


pip install mlxtend


# In[51]:


pip install -U scikit-learn


# In[52]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[53]:


y_pred = model.predict_classes(x_test)


# In[54]:


mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))


# In[55]:


model.save_weights('model.h5')


# In[ ]:




