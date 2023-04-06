from django.shortcuts import render
from .forms import CustomerForm
from tensorflow import keras
from keras.models import load_model
import numpy
import csv
import json
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats


file = open('/home/ashiq/Documents/MCA/s4/Project/AMBTMD_latest/bike_bus.csv')
lines = file.readlines()

processedList10 = []

for i, line in enumerate(lines):
    try:
        line = line.strip()
        line = line.split(',')
        last = line[4]
        last = last.strip()
        if last == '':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4]]
        processedList10.append(temp)
    except:
        print('Error at line number: ', i)

columns = ['time','x','y','z','target']
data = pd.DataFrame(data=processedList10, columns=columns)
data.isnull().sum()

data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')

Fs = 20

targets = data['target'].value_counts().index

df = data.drop(['time'], axis=1).copy()
df.head()
Bus = df[df['target']=='Bus'].head(9427).copy()
Bike = df[df['target']=='Bike'].head(9427).copy()
Walking = df[df['target']=='Walking'].head(9427).copy()

balanced_data = pd.DataFrame()
balanced_data = pd.concat([Bus, Bike, Walking])
balanced_data.shape

label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['target'])
balanced_data.head()

print(label.classes_)

x = balanced_data[['x','y','z']]
y = balanced_data['label']

scaler = StandardScaler()
x = scaler.fit_transform(x)

scaled_x = pd.DataFrame(data = x, columns = ['x', 'y', 'z'])
scaled_x['label'] = y.values

scaled_x

inp = scaled_x.to_numpy()

Fs = 20
frame_size = Fs*4 
hop_size = Fs*2

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

x, y = get_frames(scaled_x, frame_size, hop_size)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)

x_train = x_train.reshape(564, 80, 3, 1)
x_test = x_test.reshape(142, 80, 3, 1)

### 2D CNN model 

model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = x_train[0].shape))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation = 'relu'))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation = 'softmax'))


model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])    

history = model.fit(x_train, y_train, epochs = 10, validation_data = (x_test, y_test), verbose=1)


y_pred = np.argmax(model.predict(x_test), axis=-1)
print(y_pred)

# input = pd.read_csv('/home/ashiq/Documents/MCA/s4/Project/AMBTMD_latest/Datasets/test_input.csv')
# input_array = input.to_numpy()
# features = input_array
# features = features.reshape(-1, 80, 3, 1)
# prediction1 = model.predict(features)
# print(prediction1)






# model = keras.models.load_model('/home/ashiq/Documents/MCA/s4/Project/AMBTMD/savedModels/my_model.h5')

    
def predictor(request):
    if request.method=='POST':
        form = CustomerForm(request.POST, request.FILES)
        if form.is_valid():
                file = request.FILES['file']
                input_array = np.genfromtxt(file, delimiter=',')
                features = input_array
                features = features.reshape(-1, 80, 3, 1)
                prediction1 = model.predict(features)
                
                class_names = label.classes_
                print(class_names)
                class_index = np.argmax(prediction1)
                class_name = class_names[class_index]
                print(class_name)
                print("The predicted class is:", class_name)
                # data = []
                # decoded_file = file.read().decode('utf-8').splitlines()
                # reader = csv.reader(decoded_file)
                # for row in reader:
                #     data.append(row)
                # data_array = numpy.array(data)
                # print(data_array)
                # dt = data_array.reshape(-1, 80, 3, 1)
                # prediction = model.predict(dt)    
                return render(request, 'success.html',{'class_name':class_name})
    else:
        form = CustomerForm()
    context = {
        'form':form,
    }
    
    return render(request, 'main.html',context)

def forminfo(request):
    return render(request, 'result.html')

# y_pred
