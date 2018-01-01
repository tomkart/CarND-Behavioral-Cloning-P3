# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-cnn-architecture.png "Model Visualization"
[image2]: ./examples/center_2017_12_03_14_28_28_822.jpg "Center"
[image3]: ./examples/center_2017_12_03_14_34_10_122.jpg "Recovery Image"
[image4]: ./examples/center_2017_12_03_14_34_13_057.jpg "Recovery Image"
[image5]: ./examples/center_2017_12_03_14_34_19_149.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/center_2017_12_03_14_33_53_540.jpg "reverse"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 (same as model_5_v2.h5) containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I am using Nvidia model.

```sh
#NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

It consists of a 5 convolution and then 3 fully connected layers  (code line 68-77) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 81). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

To reduce overfitting, I have added dropout in the Nvidia model

```sh
#NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
```


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (2 laps), recovering from the left and right sides of the road (1 lap), and in reverse direction (1 lap).

For details about how I created the training data, see the next section.

I have also corrected the image color in model.py, so that it will be the same as the image read in by drive.py.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network, LeNet model. The car went off track in a short distance.

Then I tried with the Nvidia model with 3 epoch, the car was able to drive itself further but went off track after 25s. (run3.mp4)
Next, I tried with 5 epoch, it drove most of the track but went off after crossing the bridge. (run5.mp4) That turn has different materials on the right.  

To improve the driving behavior in that cases, I drove that turn in Training mode again to add more training data for that part of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. (run5_v2.mp4)

#### 2. Final Model Architecture


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to center from the side. These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded the driving in reverse for 1 lap as well.

![alt text][image8]

To augment the data sat, I also flipped images in the model.



After the collection process, I had 30254 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was above 3 and under 5 as evidenced with the previous testing result (run3.mp4 vs run5.mp4) and training loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Finally it can drive itself through the track 1 with final model with dropout. (video.mp4 / run3_dropout.mp4) 
