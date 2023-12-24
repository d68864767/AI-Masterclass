# Deep Learning Tutorial

Welcome to the Deep Learning tutorial of our AI Masterclass project. In this tutorial, we will guide you through the process of building a deep learning model using TensorFlow and Keras. We will be using a Convolutional Neural Network (CNN) for this tutorial, which is a type of deep learning model particularly effective for image classification tasks.

## Prerequisites

Before we start, make sure you have the following installed:

- Python 3.6 or later
- TensorFlow 2.0 or later
- Keras
- Numpy
- Pickle

## Step 1: Import Necessary Libraries

First, we need to import the necessary libraries. We will be using TensorFlow for building our deep learning model, and Keras, a user-friendly neural network library built on top of TensorFlow.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
```

## Step 2: Load and Preprocess Data

Next, we load our data using pickle. We then normalize our data by dividing by 255. This is a common preprocessing step for image data.

```python
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
X = X/255.0
```

## Step 3: Define the Model

Now, we define our model. We are using a Sequential model, which is a linear stack of layers. We add several layers to our model including Conv2D, MaxPooling2D, and Dense layers.

```python
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))
```

## Step 4: Compile and Train the Model

After defining our model, we need to compile it. During compilation, we define the loss function and the optimizer. Then, we train our model using our data.

```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
```

And that's it! You have successfully built a deep learning model for image classification. You can now use this model to make predictions on new data. Remember, deep learning is a vast field and this tutorial just scratches the surface. Keep exploring and learning!
