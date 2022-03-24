"""
    Sonar Classifier
    Simple Models. These models are useful for testing input pipelines and validating that
    other code changes still work.
"""

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import tensorflow_probability as tfp


def simple_model(input_shape, num_classes, activation_fcn, p, d ):
    ''' This is a simple model for testing the pipeline and initial data tests
        input_shape: size of input spectrograms
        num_classes: number of output classes to predict
        activation_fcn: final activation function, softmax for multi class, sigmoid for multi label

        Performs 2 class classificaiton at output of 2 nodes
    '''
    # input shape should be time series, bins
    #input_shape=( 120, 513, 1)
    model = keras.Sequential()
    # input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # add batch norm
    model.add(layers.BatchNormalization())

    # first Conv Layer
    model.add(layers.Conv2D(32, (5,5), activation = 'relu', padding='same', strides=2))
    model.add(layers.MaxPooling2D((2,2)))

    # Second Conv Layer
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', padding='same', strides=2))
    model.add(layers.MaxPooling2D((2,2)))

    # flatten
    model.add(layers.Flatten())

    # dense
    model.add(layers.Dense(56, activation='relu'))
    # add dropout
    model.add(layers.Dropout(0.2))

    #output layer
    model.add(layers.Dense(num_classes, activation= activation_fcn))

    return model

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


def simple_MC_model(input_shape, num_classes, activation_fcn, p, d ):
    model = keras.Sequential()
    # input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # add batch norm
    model.add(layers.BatchNormalization())

    # first Conv Layer
    model.add(layers.Conv2D(32, (5,5), activation = 'relu', padding='same', strides=2))
    model.add(PermaDropout((0.3)))
    model.add(layers.MaxPooling2D((2,2)))

    # Second Conv Layer
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', padding='same', strides=2))
    model.add(PermaDropout((0.3)))
    model.add(layers.MaxPooling2D((2,2)))

    # flatten
    model.add(layers.Flatten())

    # dense
    model.add(layers.Dense(56, activation='relu'))
    # add dropout
    model.add(PermaDropout(0.3))

    #output layer
    model.add(layers.Dense(num_classes, activation = activation_fcn))
    
    return model


def simple_BNN_VI_model(input_shape, num_classes, activation_fcn, p, d,ds_size ):
    
    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (ds_size *1.0)
    model = keras.Sequential()
    # input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # add batch norm
    model.add(layers.BatchNormalization())
    model.add(tfp.layers.Convolution2DFlipout(32,kernel_size=(5,5),padding="same",strides=2,
                                                 activation = 'relu', kernel_divergence_fn=kernel_divergence_fn,))
    # first Conv Layer
    model.add(layers.MaxPooling2D((2,2)))

    # Second Conv Layer
    model.add(tfp.layers.Convolution2DFlipout(32,kernel_size=(3,3),padding="same",strides=2,
                                                 activation = 'relu', kernel_divergence_fn=kernel_divergence_fn,))    
    model.add(layers.MaxPooling2D((2,2)))

    # flatten
    model.add(layers.Flatten())

    # dense
    model.add(tfp.layers.DenseFlipout(56, activation = 'relu', kernel_divergence_fn=kernel_divergence_fn))
    
    model.add(tfp.layers.DenseFlipout(num_classes, activation = 'softmax', kernel_divergence_fn=kernel_divergence_fn))
 
    #output layer
    return model