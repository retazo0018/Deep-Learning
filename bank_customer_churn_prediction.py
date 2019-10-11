
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical Independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encoding country categorical variable
labelEncoder_X_1 = LabelEncoder() 
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
# encoding gender categorical variable
labelEncoder_X_2 = LabelEncoder() 
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2]) 

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# removing 1st column to avoid dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (compulsary for ANN to avoid one independent variable dominating others)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    
#ANN

import keras
from keras.models  import Sequential
from keras.layers import Dense

# intialising the ANN
classifier = Sequential()

# Adding input layer and the first hidden layer
# Average number of nodes in input layer and average number of nodes in output layer = nodes in hidde layer
# output_dim - number of nodes in hidden layer, init - initialise weights with values closer to zero, activation- rectifier function, input_dim - number of layers in input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu',input_dim=11))

# Adding second hidden layer 
classifier.add(Dense(output_dim = 6 , init = 'uniform', activation='relu'))

# Adding output layer
# if your dependent variable is multi class, then output_dim is set to number
# of classes and activation function becomes softmax
classifier.add(Dense(output_dim = 1 , init = 'uniform', activation='sigmoid'))

# Compiling the ANN - applying stochastic gradient descent 
# adam - stochaistic gradient descent, loss - lograthmic loss function since 
# it is a sigmoid activation function ; binary_crossentropy for 2 class 
# classification, categorical_crossentropy for multiclass classification
# metric = criterion to improve model performance 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
