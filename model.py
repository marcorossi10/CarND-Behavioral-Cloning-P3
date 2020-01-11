import csv
import cv2
import numpy as np
from scipy import ndimage

lines = []
images = []
measurements = []
    
with open('DrivingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Here I append all the images and the measurements also their flipped version
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'DrivingData/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append((-1.0)*measurement)

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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
#Convolutional Layer 
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
#Convolutional Layer 
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
#Convolutional Layer 
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
#Convolutional Layer 
model.add(Conv2D(64, (3, 3), activation="relu"))
#Convolutional Layer 
model.add(Conv2D(64, (3, 3), activation="relu"))
#Dropout
model.add(Dropout(0.5)) #Remember that this is the dropout probability
#Flatten layer
model.add(Flatten())
#Fully Connected layer
model.add(Dense(100))
#Fully Connected layer
model.add(Dense(50))
#Fully Connected layer
model.add(Dense(10))  
#Fully Connected layer
model.add(Dense(1))

#Configures the model for training
model.compile(loss='mse', optimizer='adam') 
#Trains the model for a fixed number of epochs (iterations on a dataset)
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs=2, verbose = 1)

model.save('model.h5')
    