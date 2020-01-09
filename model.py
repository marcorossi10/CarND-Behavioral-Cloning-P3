import csv
import cv2
import numpy as np

lines = []
images = []
measurements = []

# with open('DrivingData/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)

# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = 'DrivingData/IMG/' + filename
#     image = cv2.imread(current_path)
#     images.append(image)
#     measurement = float(line[3])
#     measurements.append(measurement)
    
with open('DrivingData3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Here I append all the images and the measurements also their flipped version
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'DrivingData3/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append((-1.0)*measurement)

    
print(len(measurements), len(images))
#Conversion to numpy array as Keras requires
X_train = np.array(images)
y_train = np.array(measurements)

#Neural network definition
#from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()
#Preprocessing layers
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Convolutional Layer 
model.add(Conv2D(filters = 6, kernel_size = (5, 5)))
model.add(Activation('relu'))
model.add(Dropout(0.15)) #Specify the probability to DROP nodes

#Pooling
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.15)) #Specify the probability to DROP nodes

#Convolutional Layer 
model.add(Conv2D(filters = 16, kernel_size = (5, 5)))
model.add(Activation('relu'))
model.add(Dropout(0.15)) #Specify the probability to DROP nodes

#Pooling
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.15)) #Specify the probability to DROP nodes

#Flatten layer
model.add(Flatten())

#Fully Connected layer
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.15)) #Specify the probability to DROP nodes

#Fully Connected layer
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dropout(0.15)) #Specify the probability to DROP nodes

#Fully Connected layer
model.add(Dense(1))

#Configures the model for training
model.compile(loss='mse', optimizer='adam') 
#Trains the model for a fixed number of epochs (iterations on a dataset)
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs=2)

#plot_model(model, to_file='network_layers.png')

model.save('model.h5')




    