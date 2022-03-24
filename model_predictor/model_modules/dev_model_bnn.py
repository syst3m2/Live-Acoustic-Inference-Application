"""
    Andrew Pfau
    Thesis
    
    This file holds the dev model used for testing different model architectures of CNNs
    Split into a file seperate from model.py because it changes more frequently

"""
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import sys

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


def dev_model(input_shape, num_classes, activation_fcn, bnn=False, hparam=None,mc_dropout_prob=0.3,l2=0.0001):
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
    regularizer = tf.keras.regularizers.l2(l2)
    KERNEL = (10,5)
    inputL = layers.Input(shape=input_shape)
    #add batch norm layer
    x = layers.BatchNormalization()(inputL)

    # repeat same structure NUM times, changing number of filters and kernel size in each block
    for i in range(NUM_BLOCKS):
        conv1 = layers.Conv2D(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same', use_bias=False,kernel_regularizer=regularizer, name='conv1_block_' + str(i))(x)
        batchNorm1 = layers.BatchNormalization()(conv1)
        act1 = layers.Activation("relu")(batchNorm1)

        conv2 = layers.Conv2D(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same', use_bias=False,kernel_regularizer=regularizer, name='conv2_block_'+ str(i))(act1)
        batchNorm2 = layers.BatchNormalization()(conv2)
        act2 = layers.Activation("relu")(batchNorm2)
        
        # reduce inputs to the next layer by 2 in each dimension
        if i < 4:
            maxP = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(act2)
            if bnn:
                x = PermaDropout(mc_dropout_prob)(maxP)
            else:
                x= layers.Dropout(0.20)(maxP)

        NUM_FILTERS += 16
        
    # flatten output of conv layers
    if bnn:
        x = PermaDropout(mc_dropout_prob)(act2)
    else:
        x= layers.Dropout(0.20)(act2)

    flat = layers.Flatten()(x)
    
    # dense layer for classification
    dense = layers.Dense(NUM_DENSE_NODES, use_bias=False)(flat)
    batchNorm = layers.BatchNormalization()(dense)
    act = layers.Activation("relu")(batchNorm)
    drop= layers.Dropout(0.20, name='last')(act)
    
    # last layer with num_class nodes
    classL = layers.Dense(num_classes,activation=activation_fcn)(drop)
    
    return keras.Model(inputs= inputL, outputs = classL, name="CNN_Model")



    

def dev_model_Reparameterization(input_shape, num_classes, activation_fcn, train_size, hparam=None):
    """   
        A model to test different models, number of filters, layer structure, and kernels
        Modified to use tensorboard's hparams plugin
        Hparams is a list of hyperparameters to test
    """
    def get_kernel_posterior_fn(kernel_posterior_scale_mean=-9.0,
                                    kernel_posterior_scale_stddev=0.1,
                                    kernel_posterior_scale_constraint=0.2):
        """
        Get the kernel posterior distribution
        # Arguments
            kernel_posterior_scale_mean (float): kernel posterior's scale mean.
            kernel_posterior_scale_stddev (float): the initial kernel posterior's scale stddev.
                ```
                q(W|x) ~ N(mu, var),
                log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
                ````
            kernel_posterior_scale_constraint (float): the log value to constrain the log variance throughout training.
                i.e. log_var <= log(kernel_posterior_scale_constraint).
        # Returns
            kernel_posterior_fn: kernel posterior distribution
        """

        def _untransformed_scale_constraint(t):
            return tf.clip_by_value(t, -1000, tf.math.log(kernel_posterior_scale_constraint))

        kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
            untransformed_scale_initializer=tf.random_normal_initializer(
                mean=kernel_posterior_scale_mean,
                stddev=kernel_posterior_scale_stddev),
            untransformed_scale_constraint=_untransformed_scale_constraint)
        return kernel_posterior_fn
    
    def get_kernel_divergence_fn(train_size, w=1.0):
        """
        Get the kernel Kullback-Leibler divergence function
        # Arguments
            train_size (int): size of the training dataset for normalization
            w (float): weight to the function
        # Returns
            kernel_divergence_fn: kernel Kullback-Leibler divergence function
        """

        def kernel_divergence_fn(q, p, _):  # need the third ignorable argument
            kernel_divergence = tfp.distributions.kl_divergence(q, p) / tf.cast(train_size, tf.float32)
            return w * kernel_divergence

        return kernel_divergence_fn
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
    for i in range(5):
        conv1 = tfp.layers.Convolution2DReparameterization(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same',
                                                kernel_posterior_fn=get_kernel_posterior_fn(),
                                                kernel_divergence_fn=None, name='conv1_block_' + str(i))
        w = conv1.add_weight(name=conv1.name + '/kl_loss_weight', 
                            initializer=tf.initializers.constant(1.0), trainable=False)
        
        conv1.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)  

        conv1 = conv1(x)

        batchNorm1 = layers.BatchNormalization()(conv1)
        act1 = layers.Activation("relu")(batchNorm1)
        drop1= layers.Dropout(0.20)(act1)
        conv2 = tfp.layers.Convolution2DReparameterization(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same',
                                                kernel_posterior_fn=get_kernel_posterior_fn(),
                                                kernel_divergence_fn=None, name='conv2_block_'+ str(i))
        w = conv2.add_weight(name=conv2.name + '/kl_loss_weight', 
                            initializer=tf.initializers.constant(1.0), trainable=False)

        conv2.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w )
        conv2 = conv2(drop1)   
        batchNorm2 = layers.BatchNormalization()(conv2)
        act2 = layers.Activation("relu")(batchNorm2)
        
        # reduce inputs to the next layer by 2 in each dimension
        if i < 4:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(act2)
        NUM_FILTERS += 16
    # flatten output of conv layers
    flat = layers.Flatten()(act2)
    
   
    # dense layer for classification
    dense = tfp.layers.DenseLocalReparameterization(NUM_DENSE_NODES,kernel_divergence_fn=None,kernel_posterior_fn=get_kernel_posterior_fn())
    w = dense.add_weight(name = dense.name+'/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
    dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

    dense = dense(flat)
    batchNorm = layers.BatchNormalization()(dense)
    act = layers.Activation("relu")(batchNorm)
    drop= layers.Dropout(0.20, name='last')(act)
    
    # last layer with num_class nodes
    classL = tfp.layers.DenseLocalReparameterization(num_classes,activation=activation_fcn,kernel_divergence_fn=None,kernel_posterior_fn=get_kernel_posterior_fn())
    w = classL.add_weight(name = classL.name+'/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
    classL = classL(drop)
    classL.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)
    return keras.Model(inputs= inputL, outputs = classL)

def dev_model_tfp_flipout(input_shape, num_classes, activation_fcn, train_size, hparam=None):
    """   
        A model to test different models, number of filters, layer structure, and kernels
        Modified to use tensorboard's hparams plugin
        Hparams is a list of hyperparameters to test
    """
    def get_kernel_posterior_fn(kernel_posterior_scale_mean=-9.0,
                                kernel_posterior_scale_stddev=0.1,
                                kernel_posterior_scale_constraint=0.2):
        """
        Get the kernel posterior distribution
        # Arguments
            kernel_posterior_scale_mean (float): kernel posterior's scale mean.
            kernel_posterior_scale_stddev (float): the initial kernel posterior's scale stddev.
              ```
              q(W|x) ~ N(mu, var),
              log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
              ````
            kernel_posterior_scale_constraint (float): the log value to constrain the log variance throughout training.
              i.e. log_var <= log(kernel_posterior_scale_constraint).
        # Returns
            kernel_posterior_fn: kernel posterior distribution
        """

        def _untransformed_scale_constraint(t):
            return tf.clip_by_value(t, -1000, tf.math.log(kernel_posterior_scale_constraint))

        kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
            untransformed_scale_initializer=tf.random_normal_initializer(
                mean=kernel_posterior_scale_mean,
                stddev=kernel_posterior_scale_stddev),
            untransformed_scale_constraint=_untransformed_scale_constraint)
        return kernel_posterior_fn
    
    def get_kernel_divergence_fn(train_size, w=1.0):
        """
        Get the kernel Kullback-Leibler divergence function
        # Arguments
            train_size (int): size of the training dataset for normalization
            w (float): weight to the function
        # Returns
            kernel_divergence_fn: kernel Kullback-Leibler divergence function
        """

        def kernel_divergence_fn(q, p, _):  # need the third ignorable argument
            kernel_divergence = tfp.distributions.kl_divergence(q, p) / tf.cast(train_size, tf.float32)
            return w * kernel_divergence

        return kernel_divergence_fn

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
    for i in range(5):
        conv1 = tfp.layers.Convolution2DFlipout(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same',
                                                kernel_posterior_fn=get_kernel_posterior_fn(),
                                                kernel_divergence_fn=None, name='conv1_block_' + str(i))
        w = conv1.add_weight(name=conv1.name + '/kl_loss_weight', 
                            initializer=tf.initializers.constant(1.0), trainable=False)
        
        conv1.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)  

        conv1 = conv1(x)

        batchNorm1 = layers.BatchNormalization()(conv1)
        act1 = layers.Activation("relu")(batchNorm1)
        drop1= layers.Dropout(0.20)(act1)
        conv2 = tfp.layers.Convolution2DFlipout(NUM_FILTERS, kernel_size=KERNEL, strides=1, padding='same',
                                                kernel_posterior_fn=get_kernel_posterior_fn(),
                                                kernel_divergence_fn=None, name='conv2_block_'+ str(i))
        w = conv2.add_weight(name=conv2.name + '/kl_loss_weight', 
                            initializer=tf.initializers.constant(1.0), trainable=False)

        conv2.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w )
        conv2 = conv2(drop1)   
        batchNorm2 = layers.BatchNormalization()(conv2)
        act2 = layers.Activation("relu")(batchNorm2)
        
        # reduce inputs to the next layer by 2 in each dimension
        if i < 4:
            x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(act2)
        NUM_FILTERS += 16
    
    # flatten output of conv layers
    flat = layers.Flatten()(act2)
    
    # dense layer for classification
    dense = tfp.layers.DenseFlipout(NUM_DENSE_NODES,kernel_divergence_fn=None,kernel_posterior_fn=get_kernel_posterior_fn())
    w = dense.add_weight(name = dense.name+'/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
    dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

    dense = dense(flat)
    batchNorm = layers.BatchNormalization()(dense)
    act = layers.Activation("relu")(batchNorm)
    drop= layers.Dropout(0.20, name='last')(act)
    
    # last layer with num_class nodes
    classL = tfp.layers.DenseFlipout(num_classes,activation=activation_fcn,kernel_divergence_fn=None,kernel_posterior_fn=get_kernel_posterior_fn())
    w = classL.add_weight(name = classL.name+'/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
    classL = classL(drop)
    classL.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)
    return keras.Model(inputs= inputL, outputs = classL)


def build_model(params, input_shape, activation_fcn,ds_size,hparams=None):
    # can alter this later to handle a hparam search case and non hpararm case 
    bnn_type = params['bnn_type']
    if bnn_type not in ['none','tfp_flipout', 'dropout','tfp_reparameterization']:
        sys.exit("Specificed model type not allowed")
    if bnn_type == 'none':
        return dev_model(input_shape, params['num_classes'], activation_fcn, False, hparams)
    elif bnn_type == 'dropout':
        return dev_model(input_shape, params['num_classes'], activation_fcn, True, hparams,params['mc_dropout_prob'],params['dropout_l2'])
    elif bnn_type == 'tfp_reparameterization':
        if params['kl_term'] == -1:
            size = ds_size
        else:
            size = params['kl_term']
        return  dev_model_Reparameterization(input_shape, params['num_classes'], activation_fcn, size, hparams)
    elif bnn_type == 'tfp_flipout':
        if params['kl_term'] == -1:
            size = ds_size
        else:
            size = params['kl_term']
        return  dev_model_tfp_flipout(input_shape, params['num_classes'], activation_fcn, size, hparams)