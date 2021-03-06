#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* [model.h5](https://cadence.box.com/s/kx14zqrbiyl4hfajoc52sbrh0it7qws6) containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* data shows a small sample (100 images) of the training data

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based heavily off of Alexnet. The model essentially consists of two convolution layers, followed by a flatten layer, then three fully connected layers of 1000, 100, and finally 1.

The model includes RELU layers in between each of the aforementioned layers to introduce nonlinearity.

For the preprocessing, the images are cropped and then normalized using two Keras layers.

####2. Attempts to reduce overfitting in the model

The model contains a gaussian layer in order to reduce overfitting.

In addition, the model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used with an Adam optimizer and a binary crossentorpy loss function.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I simply drove around the track keeping the car roughly in the center, with some wiggle to "show" the car how to recover from the left and right sides of the road. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar Alexnet, like in the previous labs.  I thought this model might be appropriate because it worked in the past.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the left and right camera images with an offset on the training data but not on the validation data.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle almost goes off the track, but the car does recover.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

My model is based heavily off of Alexnet. The model essentially consists of two convolution layers, followed by a flatten layer, then three fully connected layers of 1000, 100, and finally 1.

Reveiew Response 1: My layer is relatively simple and quite derivate of Alexnet. Specifically, there are three 'sections' to my model: a preprocessing section, a convolutional section, and a fully connected section.

The first section I would consider the preprocessing section. First, there is a cropping layer. The top and bottom 50 pixels of the image are removed, resulting in a N x 60 x 320 x 3 matrix. That is followed by a gaussian noise layer, with sigma = 0.5. Finally, the matrix is normalized. Now, all the values in the matrix are between -0.5 and 0.5.

The section section is the convolutional section. There are two iterations of this section. First, the convolution layer uses 32 filters, with a 3x3 kernal and border, by default, is 'same'. Therefore, this layer ends up with a size of N x 58 x 318 x 32. The 2x2 pooling layer that follows, cuts the number of rows and number of cols in half; therefore, N x 29 x 159 x 32. This is followed by a RELU activation layer to reduce nonlinearity.

The convolutional section is then repeated with the exact same three layers. The convolution layer reduces the number of rows and columns by 2, then the pooling layer cuts the number of rows and column in half, and the activation layer has no change. This hidden layer has a size of N x (29-2)/2 x (159-2)/2 x 32, or N x 13 x 78 x 32;  non-integers are truncated.

Between the second and final section is a flattening layer. 13*78*32=32448. Ergo, this hidden layer has a size of N x 32448.

The final section consists of three fully connected layers. First is a dense layer of size 1024, then a dense layer of size 128, followed by a dense layer of size 1, the output of which is N x 1, or, the predicted steering angles for N frames. N.B. - in between each dense layer is a RELU activation layer. Again, this is to introduce nonlinearity. 

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on the track using center lane driving with occasional wiggles to provide some recovery data.

I did not use track two.

To augment the data set, I flipped the images and angles. This way, the data wouldn't skew to the left or right. I also added the left and right camera images with an offset to help teach the car how to correct itself.

I did not use the left and right camera images in the validation set.

I used roughly 2000 images to train the model. 

The data were randomly shuffle. 20% of the data were kept as a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. After 4 epochs, the accuracy on both the training and validation sets were quite low. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Review Response 2: In response to my review and after researching into Keras optimizers, it does appear that adam is the generally preferred optimizer compared to RMSprop. In all tests, it does allow for faster training. That being said - after changing the code and retraining my model, there was no discernable differnce in performance.

