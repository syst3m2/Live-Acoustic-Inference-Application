# file to convert command line parameters into model optimizer function

import tensorflow as tf

def get_optimizer(params):
    if params['optimizer'] == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=params['learning_rate_start'])
    elif params.optimizer == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=params['learning_rate_start'])

def get_loss(loss, label_smooth):
    if loss == 'binarycrossentropy':
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smooth)
    elif loss == 'crossentropy':
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
    elif loss == 'mse':
        return tf.keras.losses.MeanSquaredError()