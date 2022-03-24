"""
    Andrew Pfau
    Sonar Classifier
    
    This file holds the baseline custom cnn model used for testing different model architectures of CNNs
    Split into a file seperate from model.py because it changes more frequently

    This is a CNN with rectangular kernels and 5 blocks

"""
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def dev_model(input_shape, num_classes, activation_fcn, hparam=None):
    """   
        A model to test different models, number of filters, layer structure, and kernels
        Modified to use tensorboard's hparams plugin
        Hparams is a list of hyperparameters to test
    """
    #print(str(hparams.keys()))
    NUM_FILTERS = 16
    # size indicates size x size, ie 5x5 
    NUM_BLOCKS = 5
    # the most parameters come from the dense layer, reducing number of dense layer nodes
    # reduces number of model parameters
    NUM_DENSE_NODES = 32

    KERNEL = (10,5)
    inputL = layers.Input(shape=input_shape)
    #add batch norm layer
    x = layers.BatchNormalization()(inputL)

    # repeat same structure NUM times, changing number of filters and kernel size in each block
    for i in range(NUM_BLOCKS):
        conv1 = layers.Conv2D(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same', use_bias=False, name='conv1_block_' + str(i))(x)
        batchNorm1 = layers.BatchNormalization()(conv1)
        act1 = layers.Activation("relu")(batchNorm1)

        conv2 = layers.Conv2D(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same', use_bias=False, name='conv2_block_'+ str(i))(act1)
        batchNorm2 = layers.BatchNormalization()(conv2)
        act2 = layers.Activation("relu")(batchNorm2)
        
        # reduce inputs to the next layer by 2 in each dimension
        if i < 4:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(act2)
        NUM_FILTERS += 16
    
    # flatten output of conv layers
    flat = layers.Flatten()(act2)
    
    # dense layer for classification
    dense = layers.Dense(NUM_DENSE_NODES, use_bias=False)(flat)
    batchNorm = layers.BatchNormalization()(dense)
    act = layers.Activation("relu")(batchNorm)
    drop= layers.Dropout(0.20, name='last')(act)
    
    # last layer with num_class nodes
    classL = layers.Dense(num_classes,activation=activation_fcn)(drop)
    
    return keras.Model(inputs= inputL, outputs = classL, name="CNN_Model")

def build_model(param, input_shape, activation_fcn, hparams=None):
    # can alter this later to handle a hparam search case and non hpararm case 
    return dev_model(input_shape, param['num_classes'], activation_fcn, hparams)
