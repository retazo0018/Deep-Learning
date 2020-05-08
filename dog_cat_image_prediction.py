# Dataset is available in kaggle.
# Training set has 8000 images of dogs and cats and test set contains 2000 images of dogs and cats
# Training set accuracy - 85.16
# Test set accuracy - 81.10


# Part 1
# building Convolutional Neural Network
# import keras packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialise CNN
classifier = Sequential()
# Step 1 - Convolution
# 32 - feature detectors of size 3*3
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding second convolution layer
classifier.add(Convolution2D(32,3,3, activation='relu'))

# Adding second pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Fully connection layer
# Number of nodes in output_dim is practical ; choosing over 100 is good
classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))

# Compiling the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Part 2 -  Fitting cnn to the images
# Image augmentation - to avoid overfitting ; allows to enrich training set wothout adding
# new images by performing image rotation, shearing, scalinge etc.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000)


# To improve accuracy of the neural network in test set, and to reduce the difference
# between training set and test set accuracy
#   1. Add another convolution layer
#   2. Add fully conected layer
