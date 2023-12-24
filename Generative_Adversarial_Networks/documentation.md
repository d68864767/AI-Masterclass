# Generative Adversarial Networks (GANs) Documentation

This document provides an overview of the Generative Adversarial Networks (GANs) module of the AI Masterclass project. The purpose of this module is to provide a comprehensive understanding of GANs, their architecture, and how to implement them using TensorFlow.

## Table of Contents

1. [Introduction to GANs](#introduction)
2. [GAN Architecture](#architecture)
3. [Implementation Details](#implementation)
4. [Running the Code](#running)
5. [References](#references)

## Introduction to GANs <a name="introduction"></a>

Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework. They were introduced by Ian Goodfellow et al. in 2014.

## GAN Architecture <a name="architecture"></a>

A GAN consists of two parts, a Generator and a Discriminator. The Generator takes in random noise as input and generates samples. The Discriminator takes in these samples and classifies them as real or fake. The goal of the Generator is to generate samples that the Discriminator classifies as real.

## Implementation Details <a name="implementation"></a>

The code for this module is written in Python and uses the TensorFlow library for implementing the GAN. The GAN is trained on the MNIST dataset, which consists of handwritten digits.

The Generator and Discriminator are both implemented as simple feed-forward neural networks using the Keras Sequential API. The Generator takes in a 100-dimensional noise vector and outputs a 784-dimensional vector (which corresponds to a 28x28 image). The Discriminator takes in a 784-dimensional vector and outputs a single scalar representing the probability that the input is real.

The GAN is trained using the Adam optimizer and the binary cross-entropy loss function. The Generator and Discriminator are trained alternately.

## Running the Code <a name="running"></a>

To run the code, you need to have Python and TensorFlow installed. You can then run the code using the following command:

```bash
python Generative_Adversarial_Networks/code.py
```

This will train the GAN on the MNIST dataset and save the generated images in the current directory.

## References <a name="references"></a>

1. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
2. TensorFlow Documentation: https://www.tensorflow.org/
