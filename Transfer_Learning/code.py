```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import pickle

# Define constants
INPUT_SHAPE = (50, 50, 3)  # Input shape for the model
NUM_CLASSES = 10  # Number of output classes

# Load your data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# Normalize data
X = X/255.0

# Load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=tf.keras.Input(shape=INPUT_SHAPE))

# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(NUM_CLASSES, activation="softmax")(headModel)

# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# Compile our model
print("[INFO] compiling model...")
opt = Adam(lr=1e-4, decay=1e-4 / 20)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the head of the network for a few epochs (all other layers are frozen) -- this will allow the new FC layers to start to become initialized with actual "learned" values versus pure random
print("[INFO] training head...")
model.fit(X, y, batch_size=32, epochs=20, validation_split=0.2)
```
