```python
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define constants
INPUT_SHAPE = (32, 32, 3)  # Input shape (e.g., for a 32x32 RGB image)
CONV_FILTERS = 32  # Number of convolution filters
CONV_KERNEL_SIZE = (3, 3)  # Size of convolution kernel
POOL_SIZE = (2, 2)  # Size of pooling window
HIDDEN_DIM = 128  # Hidden layer dimension
OUTPUT_DIM = 10  # Output dimension (e.g., for 10-class classification)

# Define your convolutional neural network
class ConvolutionalNetwork(Sequential):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.add(Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu', input_shape=INPUT_SHAPE))
        self.add(MaxPooling2D(pool_size=POOL_SIZE))
        self.add(Flatten())
        self.add(Dense(HIDDEN_DIM, activation='relu'))
        self.add(Dense(OUTPUT_DIM, activation='softmax'))

# Instantiate the model
model = ConvolutionalNetwork()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```
