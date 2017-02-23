import os
import csv
import cv2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

dataFolder = 'data/'
dataPath = dataFolder + 'driving_log.csv'

samples = []
with open(dataPath) as dataFile:
    logreader = csv.reader(dataFile)
    headers = next(logreader) #get rid of header row..
    for row in logreader:
        centerImage = row[0].strip()
        leftImage = row[1].strip()
        rightImage = row[2].strip()
        steering = row[3]
        if cv2.imread(dataFolder + centerImage).any():
        	samples.append(row)

writer = csv.writer(open(dataFolder + 'driving_log_nick.csv', 'w'))
for row in data:
    if counter[row[0]] >= 4:
        writer.writerow(row)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def parseSample( data_folder, batch_sample, image_number, image_offset ):
    image_name = data_folder + batch_sample[image_number].strip()
    #print(image_name)
    image = cv2.imread(image_name)
    angle = float(batch_sample[3]) + image_offset
    return image, angle

def generator(data_folder, samples, batch_size = 128, LRoffset = 3):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image, angle = parseSample( data_folder, batch_sample, 0, 0 )                
                images.append(image)
                angles.append(angle)
                
                if( LRoffset >=0 ):
                    image, angle = parseSample( data_folder, batch_sample, 1, LRoffset )
                    
                    images.append(image)
                    angles.append(angle)
                    
                    image, angle = parseSample( data_folder, batch_sample, 2, -1*LRoffset )
                    
                    images.append(image)
                    angles.append(angle)
                    
            # trim image to only see section with road
            #yield images, angles
            X = np.array(images)
            y = np.array(angles)
            #yield(X_train, y_train)
            yield sklearn.utils.shuffle(X, y)

# compile and train the model using the generator function
train_generator = generator(dataFolder, train_samples, batch_size=4)
validation_generator = generator(dataFolder, validation_samples, batch_size=4)

#train_data = next(train_generator)
#validation_data = next(train_generator)

# set up cropping2D layer
model = Sequential()
#Crop
model.add(Cropping2D(cropping=((50,50), (0,0)), input_shape=(160,320,3)))

#Normalize 
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#b = max' - a * max or 
#a = (max'-min')/(max-min)
#newvalue = a * value + b

#Rest of the model
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('linear'))

model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.2))
model.add(Activation('linear'))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('linear'))

model.add(Dense(1))
model.add(Activation('linear'))

#myX = next(train_generator)[0]
#myy = next(train_generator)[1]
#print(myX)
#print(myy)
#print(len(next(train_generator[1]))
#print(len(set(train_samples[1])))
#print(train_samples[1])
model.compile(loss='mean_absolute_error', optimizer='rmsprop')
#print( model.input_shape )
#print( model.inputs )
#print( model.outputs )
#print( model.output_shape )


#history_object = model.fit(myX, myy,
	#samples_per_epoch=len(train_samples),
	#validation_data=validation_generator,
	#nb_val_samples=len(validation_samples),
	#nb_epoch=3,
    #verbose=2)

#model.compile(loss='mse', optimizer='sgd')
history_object = model.fit_generator(train_generator,
	samples_per_epoch=len(train_samples),
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples),
	nb_epoch=3,
    verbose=2)

#print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#score = model.evaluate_generator(validation_generator)

model.save('model.h5')