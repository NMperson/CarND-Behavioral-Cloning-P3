import csv
import cv2
import argparse

import math
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GaussianNoise

import h5py

def getData( dataPath ):
    samples = []
    steering = []
    with open(dataPath) as dataFile:
        logreader = csv.reader(dataFile)
        headers = next(logreader) #get rid of header row..
        for row in logreader:
            #centerImage = row[0].strip()
            #leftImage = row[1].strip()
            #rightImage = row[2].strip()
            #steering.append( float(row[3]) )
            #if cv2.imread(dataFolder + centerImage).any():
            samples.append(row)
    print(len(samples))
    return samples

def column(matrix, i):
    return [row[i] for row in matrix]

def parseSample( batch_sample, image_number, image_offset ):
    image_name = batch_sample[image_number].strip()
    image = cv2.imread(image_name)
    angle = float(batch_sample[3]) + image_offset
    return image, angle

#returns a batch of batch_size*6
def generator( samples, batch_size = 128, LRoffset = 0.5):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            #print(batch_samples)

            images = []
            angles = []
            for batch_sample in batch_samples:
                image, angle = parseSample( batch_sample, 0, 0 )                
                images.append(image)
                angles.append(angle)

                images.append(cv2.flip(image,1))
                angles.append(-1*angle)
                
                if( LRoffset > 0 ):
                    image, angle = parseSample( batch_sample, 1, LRoffset )
                    images.append(image)
                    angles.append(angle)

                    images.append(cv2.flip(image,1))
                    angles.append(-1*angle)
                    
                    image, angle = parseSample( batch_sample, 2, -1*LRoffset )
                    images.append(image)
                    angles.append(angle)

                    images.append(cv2.flip(image,1))
                    angles.append(-1*angle)
                    
            X = np.array(images)
            y = np.array(angles)

            yield sklearn.utils.shuffle(X, y)

# compile and train the model using the generator function

def printPredStats(samples, ith = 0, number = 64):#, offset = 0):
    all_prediction_names = column(samples, ith)#[0:N]
    step = math.floor(len(all_prediction_names)/number)
    prediction_names = []
    prediction_images = []
    predictions = []
    for i in range(1,len(samples),step):
        prediction_name = all_prediction_names[i].strip()
        prediction_image =  cv2.imread(prediction_name)
        prediction_images.append(prediction_image)
        prediction_names.append(prediction_name)
    predictions = model.predict(np.asarray(prediction_images))
    for angle, image in zip( predictions.tolist(), prediction_names ):
        a = str(angle[0])#+offset)
        f = image.split("\\")[-1]
        print( a + ":\t" + f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'dataFolder',
        type=str,
        help='Path to main image folder. Will also be name of h5 file.'

    )
    parser.add_argument(
        'dataFolder2',
        type=str,
        nargs='?',
        default='',
        help='Optional path to second folder of images.'
    )
    parser.add_argument(
        'dataFolder3',
        type=str,
        nargs='?',
        default='',
        help='Optional path to third folder of images.'
    )
    args = parser.parse_args()

    dataPath = args.dataFolder + '/driving_log.csv'
    samples = getData( dataPath )
    if( args.dataFolder2 ):
        dataPath = args.dataFolder2 + '/driving_log.csv'
        samples.extend( getData( dataPath ) )
    if( args.dataFolder3 ):
        dataPath = args.dataFolder3 + '/driving_log.csv'
        samples.extend( getData( dataPath ) )
    
    print("Images:")
    print(len(samples))

    #samples.append(getData( 'fistlap/driving_log.csv '))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    #offset = 0.5
    train_generator = generator( train_samples,
    	batch_size=40, LRoffset = 0.5)
    validation_generator = generator( validation_samples,
    	batch_size=40, LRoffset = 0)

    model = Sequential()
    #Crop
    model.add(Cropping2D(cropping=((50,50), (0,0)), input_shape=(160,320,3)))
    #Noise
    model.add(GaussianNoise(0.5))
    #Normalize 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    #Convolution Layer 1
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('softmax'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #Convolution Layer 2
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    #Flatten
    model.add(Flatten())
    #Fully Connected Layer 1
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #Fully Connected Layer 2
    model.add(Dense(128))
    model.add(Activation('relu'))
    #Fully Connected Layer 3
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    #mds, mae
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
    	samples_per_epoch=720,
    	validation_data=validation_generator,
    	nb_val_samples=240,
    	nb_epoch=4,
    	verbose=1)

    print(args.dataFolder)
    printPredStats(samples, ith = 0, number = 5)#, offset = 0)
    printPredStats(samples, ith = 1, number = 5)#, offset = offset)
    printPredStats(samples, ith = 2, number = 5)#, offset = 0-offset)



    modelPath = args.dataFolder +'.h5' #'model.h5' 
    model.save(modelPath, overwrite='t')

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



