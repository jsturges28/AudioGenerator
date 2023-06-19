import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from unet import UNET

'''
1. Feed the model the mel spectrograms for training

2. Train model

3. Return and save model weights
'''

LEARNING_RATE = 0.0005
BATCH_SIZE = 16
EPOCHS = 1

SPECTROGRAMS_PATH = os.path.abspath("./mel_spectrograms/")

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, files in os.walk(spectrograms_path):
        for file in files:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1) (num_samples, n_bins, n_frames, 1)
    return x_train


def train_unet(x_train, learning_rate, batch_size, epochs):
    unet = UNET(input_dim=[128, 64, 1],
                filters=[64, 128, 256],
                kernels=[3],
                strides=[2])
    
    unet.summary()
    unet.compile(learning_rate)
    history = unet.train(x_train, batch_size, epochs)

    unet._save_history(history, "history")

    return unet


if __name__ == '__main__':
    x_train = load_fsdd(SPECTROGRAMS_PATH)

    unet = train_unet(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    unet.save("model")

    

