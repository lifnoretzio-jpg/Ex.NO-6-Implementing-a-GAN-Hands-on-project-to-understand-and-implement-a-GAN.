EX.NO. 6 — IMPLEMENTING A GAN
Page 1: Aim, Objectives and Introduction
Aim
To understand the concept of Generative Adversarial Networks (GANs) and implement a simple GAN
using Python and TensorFlow/Keras for generating synthetic handwritten-style images.
Objectives
1. To understand the architecture and working principle of a GAN.
2. To implement a Generator neural network.
3. To implement a Discriminator neural network.
4. To train both networks using adversarial learning.
5. To observe the improvement in generated images during training.
Introduction
A Generative Adversarial Network is a class of deep learning model used to generate new data samples
that resemble a given training dataset. GANs were introduced by Ian Goodfellow and colleagues in 2014.
A GAN contains two neural networks: the Generator and the Discriminator. They are trained
simultaneously in a competitive process.
The Generator receives random noise as input and transforms it into a synthetic sample. The
Discriminator receives both real samples from the dataset and generated samples and predicts whether
each sample is real or fake. The Generator attempts to fool the Discriminator, while the Discriminator
attempts to become better at detecting generated samples.
Application Areas
GANs are used in image synthesis, image-to-image translation, data augmentation, super-resolution,
fashion and artwork generation, and other generative AI applications.
GAN Hands-on Project — Page 1
EX.NO. 6 — IMPLEMENTING A GAN
Page 2: Theory and Architecture
1. Generator
The Generator, commonly denoted by G, maps a random latent vector z to a generated sample G(z).
The latent vector contains random numerical values. During training, the Generator learns a
transformation that produces samples resembling the real training distribution.
2. Discriminator
The Discriminator, denoted by D, receives a sample and outputs a value representing the probability that
the sample is real. It is trained using real samples labelled as 1 and generated samples labelled as 0.
3. Adversarial Training
The two networks have competing objectives. The Discriminator tries to correctly classify real and fake
samples, while the Generator tries to produce samples that the Discriminator classifies as real.
Architecture
Component
Input
Main Function
Generator
Random noise
Output
Creates synthetic sample
Discriminator
Conceptual Flow
Real/Fake image
Classifies authenticity
Fake image
Probability
Random Noise fi Generator fi Generated Image fi Discriminator fi Real/Fake Prediction
Real Training Image fi Discriminator fi Real Prediction
The feedback from the Discriminator is used to update the Generator and Discriminator parameters
through backpropagation and gradient-based optimization.
GAN Hands-on Project — Page 2
EX.NO. 6 — IMPLEMENTING A GAN
Page 3: Mathematical Model and Algorithm
GAN Objective Function
The original GAN formulation can be represented as a minimax objective:
minG maxD V(D,G) = Ex~pdata [log D(x)] + Ez~pz [log(1 - D(G(z)))]
Here, x represents a real data sample, z represents a random latent vector, G(z) represents the
generated sample, D(x) is the probability assigned to a real sample, and D(G(z)) is the probability
assigned to a generated sample.
Training Algorithm
Step 1: Load and preprocess the training dataset.
Step 2: Normalize image pixel values to a suitable range.
Step 3: Initialize the Generator and Discriminator.
Step 4: Sample a batch of real images.
Step 5: Generate random noise vectors and use the Generator to create fake images.
Step 6: Train the Discriminator using both real and fake images.
Step 7: Generate another batch of noise vectors.
Step 8: Train the Generator while keeping the objective of making generated samples appear real to the
Discriminator.
Step 9: Repeat the process for multiple epochs.
Step 10: Save or display generated images to observe training progress.
Losses
The Discriminator loss measures how well the Discriminator separates real and generated samples. The
Generator loss measures how successfully the Generator fools the Discriminator. GAN losses can
fluctuate during training, so generated samples are also inspected to evaluate progress.
Expected Behaviour
At the beginning, generated images usually look like random noise. As training progresses, the
Generator learns important visual patterns from the dataset and produces increasingly recognizable
samples.
GAN Hands-on Project — Page 3
EX.NO. 6 — IMPLEMENTING A GAN
Page 4: Software Requirements and Dataset
Software Requirements
• Python 3.x
• TensorFlow / Keras
• NumPy
• Matplotlib
• Jupyter Notebook or Google Colab
• A computer with sufficient RAM; GPU support can reduce training time.
Dataset
For this experiment, the MNIST handwritten digit dataset is used. MNIST contains grayscale images of
handwritten digits from 0 to 9. Each image has a resolution of 28 × 28 pixels.
Preprocessing
The images are converted to floating-point values and normalized from the original 0–255 pixel range to
approximately -1 to +1. This range works well with a Generator whose final activation uses the tanh
function.
Why MNIST?
MNIST is small, standardized, and widely used for introductory machine-learning experiments. It allows
the GAN architecture to be implemented without requiring a very large dataset or complex
image-processing pipeline.
Hardware / Execution Notes
The experiment can be executed on a CPU for demonstration purposes, although training may be slower.
Google Colab or another GPU-enabled environment can be used for faster experimentation. The number
of epochs can be increased when better visual quality is required.
Input and Output
Input: Random latent noise vectors and real MNIST training images.
Output: Newly generated 28 × 28 grayscale images that attempt to resemble handwritten digits.
GAN Hands-on Project — Page 4
EX.NO. 6 — IMPLEMENTING A GAN
Page 5: Python Program — Part I
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
# Normalize images to [-1, 1]
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5
# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_DIM = 100
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False,
                     input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1),
            padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2),
            padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2),
            padding="same", activation="tanh")
    ])
    return model
generator = build_generator()
generator.summary()
Explanation
The Generator starts with a 100-dimensional random vector. A dense layer expands this representation
into a feature map. Transposed convolution layers then upsample the feature map from 7 × 7 to 28 × 28.
The final tanh layer produces a normalized grayscale image.
GAN Hands-on Project — Page 5
EX.NO. 6 — IMPLEMENTING A GAN
Page 6: Python Program — Part II
# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(
            64, (5, 5), strides=(2, 2),
            padding="same",
            input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(
            128, (5, 5), strides=(2, 2),
            padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model
discriminator = build_discriminator()
discriminator.summary()
cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(
        tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss
def generator_loss(fake_output):
    return cross_entropy(
        tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:
        generated_images = generator(
            noise, training=True)
        real_output = discriminator(
            images, training=True)
        fake_output = discriminator(
            generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(
            real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator,
            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator,
GAN Hands-on Project — Page 6
            discriminator.trainable_variables))
    return gen_loss, disc_loss
Explanation
The Discriminator uses convolutional layers to extract image features and a final dense layer to produce
a real/fake score. Binary cross-entropy is used to calculate the Generator and Discriminator losses.
Adam optimizers update the trainable parameters
