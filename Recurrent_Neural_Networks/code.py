```python
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Activation

# Define constants
INPUT_DIM = 100  # Input dimension
HIDDEN_DIM = 50  # Hidden layer dimension
OUTPUT_DIM = 10  # Output dimension
TIME_STEPS = 20  # Time steps for RNN

# Define your recurrent neural network
class RecurrentNetwork(Sequential):
    def __init__(self):
        super(RecurrentNetwork, self).__init__()
        self.add(SimpleRNN(HIDDEN_DIM, input_shape=(TIME_STEPS, INPUT_DIM), activation='relu'))
        self.add(Dense(OUTPUT_DIM, activation='softmax'))

# Create an instance of the RecurrentNetwork
model = RecurrentNetwork()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Note: This is a basic implementation of a recurrent neural network using the SimpleRNN layer from TensorFlow.
# In practice, you would typically use more complex layers like LSTM or GRU, which are capable of learning longer sequences.
# Additionally, you would usually train the model on some data using the fit method. This is omitted here for simplicity.
```
