```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam

# Define constants
INPUT_DIM = 20  # Input dimension for the model
NUM_CLASSES = 2  # Number of output classes

# Generate a 2-class classification problem
X, y = make_classification(n_samples=1000, n_features=INPUT_DIM, n_informative=15, n_redundant=5, n_classes=NUM_CLASSES)

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=INPUT_DIM, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Define the optimizers
optimizers = {
    'sgd': SGD(),
    'adam': Adam(),
    'rmsprop': RMSprop(),
    'adagrad': Adagrad(),
    'adadelta': Adadelta(),
    'adamax': Adamax(),
    'nadam': Nadam()
}

# Compile and train the model using different optimizers
history_dict = {}
for name, opt in optimizers.items():
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)
    history_dict[name] = history

# Plot the training loss and accuracy for different optimizers
plt.figure(figsize=(12, 6))
for name, history in history_dict.items():
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    plt.plot(val_acc, label='{} Accuracy'.format(name))
    plt.plot(val_loss, label='{} Loss'.format(name))
plt.title('Model Accuracy and Loss for Different Optimizers')
plt.ylabel('Accuracy/Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
```
