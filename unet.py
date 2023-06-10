import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras import layers
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, \
    Reshape, Conv2DTranspose, Activation, Lambda, MaxPooling2D, UpSampling2D, Concatenate
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, KLDivergence
import numpy as np
import os
import pickle
import librosa

tf.compat.v1.disable_eager_execution() # Prevent evaluating operations before graph is completely built

class CustomFit(keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model

    def compile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)

        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars) # calculate d_loss / d_training_vars

        self.optimizer.apply_gradients(zip(gradients, training_vars)) # apply gradient descent by subtracting a fraction from each variable
        self.compiled_metrics.update_state(y, y_pred) # update state of model's metrics

        return {m.name : m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data

        y_pred = self.model(x, training=False)

        loss = self.loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name : m.result() for m in self.metrics}

class SpectralLogLoss(keras.losses.Loss):
    def __init__(self):
        super(SpectralLogLoss, self).__init__()

    def call(self, y_true, y_pred, eps=1e-6, norm='l1'):
        error = y_true - y_pred
        # Compute the L1 or L2 loss
        if norm == 'l1':
            return K.mean(K.abs(error), axis=[1,2,3])
        elif norm == 'l2':
            return K.mean(K.square(error), axis=[1,2,3])
        else:
            raise ValueError("Invalid norm type: must be either 'l1' or 'l2'")


class EncoderBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides):
        super(EncoderBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, strides, padding='same')
        self.conv2 = Conv2D(filters, kernel_size, strides, padding='same')
        self.maxpool1 = MaxPooling2D(pool_size=2, strides=2)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        print("Preconcat shape:", x.shape)
        maxpool_x = self.maxpool1(x)
        return maxpool_x, x

class DecoderBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, strides, padding='same')
        self.conv2 = Conv2D(filters, kernel_size, strides, padding='same')
        self.upsample1 = UpSampling2D(size=2)
        self.concat = Concatenate()

    def call(self, input_tensor, concat_tensor):
        x = self.upsample1(input_tensor)
        print("Decoder shape: ", x.shape)
        print("Concat shape: ", concat_tensor.shape)
        x = self.concat([x, concat_tensor])
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class OutputBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides):
        super(OutputBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, strides, padding='same')
        self.conv2 = Conv2D(filters, kernel_size, strides, padding='same')
        self.convfinal = Conv2D(1, kernel_size, padding='same', activation='sigmoid')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.convfinal(x)
        return x

class UNETModel(keras.Model):
    def __init__(self, input_dim, filters, kernels, strides):
        super(UNETModel, self).__init__()
        self.input_dim = input_dim
        self.encoderblock1 = EncoderBlock(filters[0], kernels[0], strides[0])
        self.encoderblock2 = EncoderBlock(filters[1], kernels[0], strides[0])
        self.convbottom1 = Conv2D(filters[2], kernels[0], strides[0], padding='same')
        self.convbottom2 = Conv2D(filters[2], kernels[0], strides[0], padding='same')
        self.decoderblock1 = DecoderBlock(filters[2], kernels[0], strides[0])
        self.decoderblock2 = DecoderBlock(filters[1], kernels[0], strides[0])
        #self.decoderblock3 = DecoderBlock(filters[0], kernels[0], strides[0])
        #self.convfinal = OutputBlock(filters[0], kernels[0], strides[0])
        self.convfinal = Conv2D(1, kernels[0], padding='same', activation='sigmoid')

    def call(self, input_tensor):
        x, concat1 = self.encoderblock1(input_tensor)
        x, concat2 = self.encoderblock2(x)
        x = self.convbottom1(x)
        x = self.convbottom2(x)
        x = self.decoderblock1(x, concat2)
        x = self.decoderblock2(x, concat1)
        x = self.convfinal(x)
        return x
    
    def model(self):
        x = Input(shape=self.input_dim)
        return Model(inputs=[x], outputs=self.call(x))
    
class UNET:
    def __init__(self, input_dim, filters, kernels, strides):
        self.input_dim = input_dim # example: [28, 28, 1]
        self.filters = filters # [2,4,8]
        self.kernels = kernels # [3, 5, 3] 3x3, 5x5, 3x3
        self.strides = strides # [1, 2, 2]
        self.model = None

        self._build() # build the model

    def summary(self):
        self.model.summary()

    def _build(self):
        tensor_stack = []
        input_tensor = Input(shape=self.input_dim, name="input")
        tensor = input_tensor

        #128x128
        tensor = Conv2D(self.filters[0], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.filters[0], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor_stack.append(tensor)
        tensor = MaxPooling2D(pool_size=2, strides=2)(tensor)

        #64x64
        tensor = Conv2D(self.filters[1], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.filters[1], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor_stack.append(tensor) # t2
        tensor = MaxPooling2D(pool_size=2, strides=2)(tensor)

        #32x32
        tensor = Conv2D(self.filters[2], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.filters[2], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
    
        tensor = UpSampling2D(size=2)(tensor)

        #64x64

        #skip connections
        t2 = tensor_stack.pop() # removes last item from tensor stack and returns it
        tensor = Concatenate()([tensor, t2]) # Concat along channels

        #64x64
        tensor = Conv2D(self.filters[1], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.filters[1], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor = UpSampling2D(size=2)(tensor)

        #128x128
        
        # skip connection
        t3 = tensor_stack.pop()

        tensor = Concatenate()([tensor, t3])

        #128x128
        tensor = Conv2D(self.filters[0], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        tensor = Conv2D(self.filters[0], kernel_size=self.kernels[0], padding='same', activation='relu')(tensor)
        
        # output
        tensor = Conv2D(1, kernel_size=self.kernels[0], padding='same', activation='sigmoid')(tensor)

        self.model = Model(inputs=input_tensor, outputs=tensor)

    def _build1(self):
        self.model = UNETModel(input_dim=self.input_dim,
                               filters=self.filters,
                               kernels=self.kernels,
                               strides=self.strides)
        
        dummy_input = Input(shape=self.input_dim)
        self.model(dummy_input)
        
    def compile(self, learning_rate=0.0001):
        optimizer=Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=SpectralLogLoss(), metrics=['MeanSquaredError', 'MeanAbsoluteError'])
        
    def train(self, x_train, batch_size, num_epochs):
        history = self.model.fit(x_train, x_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)
        return history

    def save(self, save_folder="."):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        #self._save_history(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        reconstructed_images = self.model.predict(images)
        return reconstructed_images
    
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters_unet.pkl")
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)

        unet = UNET(*parameters)
        weights_path = os.path.join(save_folder, "weights_unet.h5")
        unet.load_weights(weights_path)

        return unet

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_dim,
            self.filters,
            self.kernels,
            self.strides,
        ]
        save_path = os.path.join(save_folder, "parameters_unet.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights_unet.h5")
        self.model.save_weights(save_path)

    def _save_history(self, history, save_folder="."):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        history_dict = history.history
        save_path = os.path.join(save_folder, "history_unet.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(history_dict, f)
    

if __name__ == '__main__':
    unet = UNET(input_dim=[128,64,1],
                filters=[64,128,256],
                kernels=[3],
                strides=[2])
    
    unet.summary()

    