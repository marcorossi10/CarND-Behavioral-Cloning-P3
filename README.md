# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/center_2020_01_10_17_38_14_243.jpg "RecoverFromRight"

Overview
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model (from line 37 to line 62 of model.py) consisted of the following layers:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |              160x320x3 RGB image            |
|  Lambda layer   |      Normalization of the input data        |
|  Cropping layer |         Crop unuseful parts of the images   |
|5x5 Convolutional|   Filters=24, 2x2 stride, VALID padding     |
|      RELU       |               Activation layer              |
|5x5 Convolutional|   Filters=36, 2x2 stride, VALID padding     |
|      RELU       |               Activation layer              |
|5x5 Convolutional|   Filters=48, 2x2 stride, VALID padding     |
|      RELU       |               Activation layer              |
|3x3 Convolutional|   Filters=64, 1x1 stride, VALID padding     |
|      RELU       |               Activation layer              |
|3x3 Convolutional|   Filters=64, 1x1 stride, VALID padding     |
|      RELU       |               Activation layer              |
|     DROPOUT     |          0.5 of drop probability            |
|   Flattening    |                                             |
| Fully connected |                 output 100                  |
| Fully connected |                 output 50                   |
| Fully connected |                 output 10                   |
| Fully connected |                 output 1                    |

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 42 to 50).

The model includes RELU layers to introduce nonlinearity (code line 42, 44, 46, 48, 50), and the data is normalized in the model using a Keras lambda layer (code line 39). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 52). A drop probability of 0.5 is selected.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16-25). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road so that the the network could learn how to behave when the vehicle was getting closer to the road edges.

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first architecture I implemented was resembling the same architecture (a modified LeNet) used in the previous project where I had to classify different traffic signs. I was curious to see how it would have behaved.

After acquiring some data of center driving and running the simulator, the car was not able to keep the center of the road and was always driving on the right side of the track close to the road edge.
So I tried to acquire data where the car was recovering from the right side of the road but this was not enough.

Thus, I decided to implement the neural network proposed by the NVIDIA autonomous driving team adding a dropout layer to reduce overfitting.

In order to gauge how well the model was working, I also split my image and steering angle data into a training and validation set. The results on training and validation data were good, leading to a small mean squared error loss (<0.009).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially on tight curves. Thus, I acquired more data in these "difficult" curves in order to teach to the neural network how to generete bigger steering angles for these situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is shown in the table in the previous section.

#### 3. Creation of the Training Set & Training Process

I created my data set by starting with the provided data and evaluating the results. With this set of data the car was performing already pretty good except for tight curves. So, as already explained in the previous section I augmented the starting data with new data on these challenging curves.
The following picture shows the starting point of such manoeuvre: the data acquisition was started from that moment and stopped only once the car got to the center of the road in a smooth way.

![alt text][image1]

It is important to underline that all the new data has been acquired by using the mouse as steering input and not the keyboard. In this way, smoother angles could be generated instead of having only a set of "impulses" from the keyboard.

To augment the data set, I also flipped all the images and changed the sign of the steering angles. This approach is a good way to augment and balance the data, especially if the track contains many curves on only one side.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

### Video acquisition

At these two following links it is possible to find the recorded videos:

* Video with the car driving at 9 miles per hour: https://github.com/marcorossi10/CarND-Behavioral-Cloning-P3/blob/master/run1_9M_h.mp4
* Video with the car driving at 25 miles per hour: https://github.com/marcorossi10/CarND-Behavioral-Cloning-P3/blob/master/run2_25M_h.mp4

I decided to record also a video where the car is driving faster than the default speed set in the drive.py file.
This scenario is of course more challenging and it can be seen how the predicted model is still performing very well.