"""
    A selection of pre-trained models to test
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def vgg_pretrained(input_shape, num_class, activation_fcn, p, d):
    input_layer = tf.keras.Input(shape=input_shape)
    preprocess_layer = tf.keras.applications.vgg16.preprocess_input(input_layer)
    pretrained_model = tf.keras.applications.VGG16(input_tensor=input_layer, input_shape=input_shape, include_top=False, weights="imagenet")
    pretrained_model.trainable = False
    pretrained = pretrained_model(preprocess_layer)
    flat = tf.keras.layers.Flatten()(pretrained) 
    
    #Add hidden and pooling layers
    batchNorm = tf.keras.layers.BatchNormalization()(flat)
    dense = tf.keras.layers.Dense(32)(batchNorm)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    #Add last layer
    class_layer = tf.keras.layers.Dense(num_class, activation = activation_fcn)(dropout)

    return keras.Model(inputs=input_layer, outputs=class_layer, name="VGG_Model")

def mobilenet_pretrained(input_shape, num_class, activation_fcn, p, d):
    input_layer = tf.keras.Input(shape=input_shape)
    preprocess_layer= tf.keras.applications.mobilenet_v2.preprocess_input(input_layer)
    pretrained_model = tf.keras.applications.MobileNetV2(input_tensor=input_layer, input_shape=input_shape, include_top=False, weights="imagenet")
    pretrained_model.trainable = False
    pretrained = pretrained_model(preprocess_layer)
    # Flatten
    flat = tf.keras.layers.Flatten()(pretrained) 
    
    #Add hidden and pooling layers
    batchNorm = tf.keras.layers.BatchNormalization()(flat)
    dense = tf.keras.layers.Dense(32)(batchNorm)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    #Add last layer
    class_layer = tf.keras.layers.Dense(num_class, activation = activation_fcn)(dropout)

    return keras.Model(inputs=input_layer, outputs=class_layer, name="MobileNet_Model")

def inception_pretrained(input_shape, num_class, activation_fcn, p, d):
    input_layer = tf.keras.Input(shape=input_shape)
    preprocess_layer = tf.keras.applications.inception_v3.preprocess_input(input_layer)
    pretrained_model = tf.keras.applications.InceptionV3(input_tensor=input_layer, input_shape=input_shape, include_top=False, weights="imagenet")
    pretrained_model.trainable = False
    pretrained = pretrained_model(preprocess_layer)
    flat = tf.keras.layers.Flatten()(pretrained) 
    
    #Add hidden and pooling layers
    batchNorm = tf.keras.layers.BatchNormalization()(flat)
    dense = tf.keras.layers.Dense(32)(batchNorm)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    #Add last layer
    class_layer = tf.keras.layers.Dense(num_class, activation = activation_fcn)(dropout)

    return keras.Model(inputs=input_layer, outputs=class_layer, name="Inception_Model")


def pretrained_model(i, num_classes, activation_fcn, model_path, d):
    """
    This function loads a pretrained model from a checkpoint and sets the top layer (classification) to trainable
    """
    pretrained = tf.keras.models.load(model_path)
    # replace last layer with the number of classes to classify
    model = tf.keras.Model(inputs=pretrained.input, outputs=pretrained.get_layer('last').output)    
    model.add(layers.Dense(num_classes, activation=activation_fcn))

    return model
