```python
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# Define constants
INPUT_DIM = 100  # Input dimension
HIDDEN_DIM = 50  # Hidden layer dimension
OUTPUT_DIM = 10  # Output dimension

# Define your feedforward network
class FeedForwardNetwork(Sequential):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.add(Dense(HIDDEN_DIM, input_dim=INPUT_DIM, activation='relu'))
        self.add(Dense(OUTPUT_DIM, activation='softmax'))

# Instantiate the model
model = FeedForwardNetwork()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define your input data
X = np.random.rand(1000, INPUT_DIM)
Y = tf.keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=OUTPUT_DIM)

# Train the model
model.fit(X, Y, epochs=10, batch_size=32)
```
