```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Reshape, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Define constants
NOISE_DIM = 100  # Noise data points
GENERATOR_OUTPUT_DIM = 784  # 28x28
GENERATOR_HIDDEN_DIM = 256
DISCRIMINATOR_INPUT_DIM = 784  # 28x28
DISCRIMINATOR_HIDDEN_DIM = 256

# Load your data
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 127.5 - 1.  # Normalize the images to [-1, 1]
X_train = np.expand_dims(X_train, axis=3)

# Define the generator
def create_generator():
    model = Sequential()
    model.add(Dense(GENERATOR_HIDDEN_DIM, input_dim=NOISE_DIM))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(GENERATOR_OUTPUT_DIM, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

# Define the discriminator
def create_discriminator():
    model = Sequential()
    model.add(Dense(DISCRIMINATOR_HIDDEN_DIM, input_dim=DISCRIMINATOR_INPUT_DIM))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Create GAN
def create_gan(discriminator, generator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile models
generator = create_generator()
discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
gan = create_gan(discriminator, generator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Training function
def train_gan(generator, discriminator, gan, epochs=10000, batch_size=128, save_interval=1000):
    for epoch in range(epochs):
        # Train discriminator
        real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        real_images = real_images.reshape(batch_size, DISCRIMINATOR_INPUT_DIM)
        noise = np.random.normal(0, 1, size=[batch_size, NOISE_DIM])
        generated_images = generator.predict(noise)
        X = np.concatenate([real_images, generated_images])
        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 0.9
        discriminator.trainable = True
        discriminator.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.normal(0, 1, size=[batch_size, NOISE_DIM])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)

        # Save generated images
        if epoch % save_interval == 0:
            save_images(epoch, generator)

# Function to save images
def save_images(epoch, generator):
    noise = np.random.normal(0, 1, size=[100, NOISE_DIM])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_images/gan_output_epoch_{0}.png'.format(epoch+1))

# Train the GAN
train_gan(generator, discriminator, gan)
```
