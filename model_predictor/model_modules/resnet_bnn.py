import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow.keras.applications.resnet50 import ResNet50
import pdb
import sys
#tf.compat.v1.disable_eager_execution()


def create_resnet_model(hparams, activation_fcn,input_shape, training_set_length):
    # hparams.resnet_n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    # hparams.resnet_version = 2

    # Computed depth from supplied model parameter n
    if hparams.resnet_version == 1:
        depth = hparams.resnet_n * 6 + 2
    elif hparams.resnet_version == 2:
        depth = hparams.resnet_n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, hparams.resnet_version)
    print(model_type)
    print(activation_fcn)
    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        regularizer = tf.keras.regularizers.l2(1e-4)
        conv = tf.keras.layers.Conv2D(num_filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=regularizer)

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
        else:
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(input_shape, depth, num_classes=hparams.num_classes):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = tf.keras.layers.Input(shape=input_shape)
        batchNorm = layers.BatchNormalization()(inputs)
        x = resnet_layer(inputs=batchNorm)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = tf.keras.layers.add([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)

        outputs = tf.keras.layers.Dense(num_classes,
                                        activation=activation_fcn,
                                        kernel_initializer='he_normal')(y)

        '''
        ### marko add classifier
        hidden1 = layers.Dense(128, use_bias=False)(y)
        batch_norm1 = layers.BatchNormalization()(hidden1)
        act1 = layers.Activation("relu")(batch_norm1)
        drop1 = layers.Dropout(0.25)(act1)
        #second
        hidden2 = layers.Dense(128, use_bias=False)(drop1)
        batch_norm2 = layers.BatchNormalization()(hidden2)
        act2 = layers.Activation("relu")(batch_norm2)
        drop2 = layers.Dropout(0.25)(act2)

        outputs = layers.Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(drop2)
        '''
        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def resnet_v2(input_shape, depth, num_classes=hparams.num_classes):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = tf.keras.layers.Input(shape=input_shape)
        batchNorm = layers.BatchNormalization()(inputs)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=batchNorm,
                         num_filters=num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = tf.keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)

        outputs = tf.keras.layers.Dense(num_classes,
                                        activation=activation_fcn,
                                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    if hparams.resnet_version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    return model



# implements bayesian model via flipout variational inference
def create_flipout_resnet_model(hparams,activation_fcn, input_shape,training_set_length):
  ########
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    # hparams.resnet_version = 2

    # this method implements a bayesian resnet, which approximates the distribution
    # of weights based upon prior data utilizing a distribution
    # and negative log likelihood loss function

    # Implemented by changing convolutional layers to 2DFlipout Layers
    # and dense layers to Dense Flipout Layers
    ########

    # Computed depth from supplied model parameter n
    if hparams.resnet_version == 1:
        depth = hparams.resnet_n * 6 + 2
    elif hparams.resnet_version == 2:
        depth = hparams.resnet_n * 9 + 2

    # Model name, depth and version
    model_type = 'Bayesian_ResNet%dv%d' % (depth, hparams.resnet_version)
    print(model_type)
    print(activation_fcn)

    # get size of training set for KL Divergence Function scaling

    train_size = training_set_length

    # This method allows the user to modify the default kernel posterior function if desired.
    # Default Kernel Posterior Function is given below:
    # https://www.tensorflow.org/probability/api_docs/python/tfp/layers/default_mean_field_normal_fn

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


    # this method will retrun a normalized kernel divergence function based on the size of the training set
    # and an additional weight (default is non-trainable) that can be adjusted as needed

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


    def bayesian_resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True,
                     kl_weight=hparams.kl_weight):


        regularizer = tf.keras.regularizers.l2(1e-4)

        conv = tfp.layers.Convolution2DFlipout(num_filters,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding='same',
                                               
                                               kernel_posterior_fn=get_kernel_posterior_fn(),
                                               kernel_divergence_fn=None)
        w = conv.add_weight(name=conv.name + '/kl_loss_weight', 
                            initializer=tf.initializers.constant(1.0), trainable=False)
 
        conv.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)



        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
        else:
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
            x = conv(x)
        return x

    def bayesian_resnet_v1(input_shape, depth, num_classes=hparams.num_classes, kl_weight=hparams.kl_weight):

        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = tf.keras.layers.Input(shape=input_shape)
        batchNorm = layers.BatchNormalization()(inputs)
        x = bayesian_resnet_layer(inputs=batchNorm)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = bayesian_resnet_layer(inputs=x,
                                          num_filters=num_filters,
                                          strides=strides,
                                          kl_weight=kl_weight)
                y = bayesian_resnet_layer(inputs=y,
                                          num_filters=num_filters,
                                          activation=None,
                                          kl_weight=hparams.kl_weight)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = bayesian_resnet_layer(inputs=x,
                                              num_filters=num_filters,
                                              kernel_size=1,
                                              strides=strides,
                                              activation=None,
                                              kl_weight=kl_weight,
                                              batch_normalization=False)
                x = tf.keras.layers.add([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)



        dense = tfp.layers.DenseFlipout(num_classes,
                                        activation=activation_fcn,
                                        kernel_posterior_fn=get_kernel_posterior_fn(),
                                        kernel_divergence_fn=None,)
        w = dense.add_weight(name = dense.name+'/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
        dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

        logits = dense(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=logits)
        return model

    def bayesian_resnet_v2(input_shape, depth, num_classes=hparams.num_classes,kl_weight=hparams.kl_weight):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = tf.keras.layers.Input(shape=input_shape)
        batchNorm = layers.BatchNormalization()(inputs)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = bayesian_resnet_layer(inputs=batchNorm,
                         num_filters=num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = bayesian_resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False,
                                 kl_weight=hparams.kl_weight)
                y = bayesian_resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False,
                                 kl_weight=hparams.kl_weight)
                y = bayesian_resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False,
                                 kl_weight=hparams.kl_weight)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = bayesian_resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = tf.keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)



        dense = tfp.layers.DenseFlipout(num_classes,
                                        activation=activation_fcn,
                                        kernel_posterior_fn=get_kernel_posterior_fn(),
                                        kernel_divergence_fn=None,)
        w = dense.add_weight(name = dense.name+'/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
        dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

        logits = dense(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=logits)
        return model

    if hparams.resnet_version == 2:
        model = bayesian_resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = bayesian_resnet_v1(input_shape=input_shape, depth=depth)

    return model


# implements a bayesian resnet model using reparameterization varaitional inference
def create_bayesian_reparam_resnet_model(hparams,activation_fcn, input_shape,training_set_length):
    ########
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    # hparams.resnet_version = 2

    # this method implements a bayesian resnet, which approximates the distribution
    # of weights based upon prior data utilizing a distribution
    # and negative log likelihood loss function

    # Implemented by changing convolutional layers to Convolution2DReparameterization Layers
    # and dense layers to DenseReparameterization Layers
    ########

    # Computed depth from supplied model parameter n
    if hparams.resnet_version == 1:
        depth = hparams.resnet_n * 6 + 2
    elif hparams.resnet_version == 2:
        depth = hparams.resnet_n * 9 + 2

    # Model name, depth and version
    model_type = 'Bayesian_ResNet%dv%d' % (depth, hparams.resnet_version)
    print(model_type)
    print(activation_fcn)
    # get size of training set for KL Divergence Function scaling

    train_size = training_set_length

    # This method allows the user to modify the default kernel posterior function if desired.
    # Default Kernel Posterior Function is given below:
    # https://www.tensorflow.org/probability/api_docs/python/tfp/layers/default_mean_field_normal_fn

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

    # this method will retrun a normalized kernel divergence function based on the size of the training set
    # and an additional weight (default is non-trainable) that can be adjusted as needed

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

    def reparam_resnet_layer(inputs,
                              num_filters=16,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              batch_normalization=True,
                              conv_first=True):

        regularizer = tf.keras.regularizers.l2(1e-4)

        conv = tfp.layers.Convolution2DReparameterization(num_filters,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding='same',
                                               kernel_posterior_fn=get_kernel_posterior_fn(),
                                               kernel_divergence_fn=None)
        w = conv.add_weight(name=conv.name + '/kl_loss_weight', shape=(),
                            initializer=tf.initializers.constant(1.0), trainable=False)
        conv.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
        else:
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
            x = conv(x)
        return x

    def reparam_resnet_v1(input_shape, depth, num_classes=hparams.num_classes):

        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)


        

        inputs = tf.keras.layers.Input(shape=input_shape)
        batchNorm = layers.BatchNormalization()(inputs)
        
        x = reparam_resnet_layer(inputs=batchNorm)

        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = reparam_resnet_layer(inputs=x,
                                          num_filters=num_filters,
                                          strides=strides)
                y = reparam_resnet_layer(inputs=y,
                                          num_filters=num_filters,
                                          activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = reparam_resnet_layer(inputs=x,
                                              num_filters=num_filters,
                                              kernel_size=1,
                                              strides=strides,
                                              activation=None,
                                              batch_normalization=False)
                x = tf.keras.layers.add([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)

        dense = tfp.layers.DenseReparameterization(num_classes,
                                        activation=activation_fcn,
                                        kernel_posterior_fn=get_kernel_posterior_fn(),
                                        kernel_divergence_fn=None, )
        w = dense.add_weight(name=dense.name + '/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
        dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

        logits = dense(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=logits)
        return model

    def reparam_resnet_v2(input_shape, depth, num_classes=hparams.num_classes):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = tf.keras.layers.Input(shape=input_shape)
        batchNorm = layers.BatchNormalization()(inputs)
        
        
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = reparam_resnet_layer(inputs=batchNorm,
                                  num_filters=num_filters_in,
                                  conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = reparam_resnet_layer(inputs=x,
                                          num_filters=num_filters_in,
                                          kernel_size=1,
                                          strides=strides,
                                          activation=activation,
                                          batch_normalization=batch_normalization,
                                          conv_first=False)
                y = reparam_resnet_layer(inputs=y,
                                          num_filters=num_filters_in,
                                          conv_first=False)
                y = reparam_resnet_layer(inputs=y,
                                          num_filters=num_filters_out,
                                          kernel_size=1,
                                          conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = reparam_resnet_layer(inputs=x,
                                              num_filters=num_filters_out,
                                              kernel_size=1,
                                              strides=strides,
                                              activation=None,
                                              batch_normalization=False)
                x = tf.keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)

        dense = tfp.layers.DenseReparameterization(num_classes,
                                        activation=activation_fcn,
                                        kernel_posterior_fn=get_kernel_posterior_fn(),
                                        kernel_divergence_fn=None, )
        w = dense.add_weight(name=dense.name + '/kl_loss_weight', shape=(),
                             initializer=tf.initializers.constant(1.0), trainable=False)
        dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)

        logits = dense(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=logits)
        return model

    if hparams.resnet_version == 2:
        model = reparam_resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = reparam_resnet_v1(input_shape=input_shape, depth=depth)

    return model


# implements resnet model with MC Dropout
# MC Dropout layers are added following activation after convolutional layers
def create_MC_dropout_resnet_model(hparams, activation_fcn,input_shape,training_set_length):
    # creates a dropout layer with an "always on" dropout rate
    # def MC_dropout_layer(dropout_rate):
    # return Lambda(lambda x: tf.keras.layers.Dropout(x, dropout_rate))

    def MC_dropout_layer(input_tensor):
        return tf.keras.layers.Dropout(hparams.mc_dropout_prob)(input_tensor, training=True)

    if hparams.resnet_version == 1:
        depth = hparams.resnet_n * 6 + 2
    elif hparams.resnet_version == 2:
        depth = hparams.resnet_n * 9 + 2


    
    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, hparams.resnet_version)
    print(model_type)
    print(activation_fcn)

    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder with MC_droput
           layers added between convolutional layers and dense layers

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        regularizer = tf.keras.regularizers.l2(1e-4)
        conv = tf.keras.layers.Conv2D(num_filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=regularizer)

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
                x = MC_dropout_layer(x)
        else:
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if activation is not None:
                x = tf.keras.layers.Activation(activation)(x)
                x = MC_dropout_layer(x)
            x = conv(x)
        return x

    def resnet_v1(input_shape, depth, num_classes=hparams.num_classes):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = tf.keras.layers.add([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)

        dense = tf.keras.layers.Dense(num_classes,
                                        activation=activation_fcn,
                                        kernel_initializer='he_normal')
        logits = dense(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=logits)

        return model

    def resnet_v2(input_shape, depth, num_classes=hparams.num_classes):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = tf.keras.layers.Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = tf.keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        y = tf.keras.layers.Flatten()(x)

        dense = tf.keras.layers.Dense(num_classes,
                                        activation=activation_fcn,
                                        kernel_initializer='he_normal')
        logits = dense(y)

        # Instantiate model.
        model = tf.keras.models.Model(inputs=inputs, outputs=logits)

        return model

    if hparams.resnet_version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    return model



def build_model(param, input_shape, activation_fcn,ds_size,hparams=None):
    bnn_type = param.bnn_type
    if bnn_type not in ['none','tfp_flipout', 'dropout','tfp_reparameterization']:
        sys.exit("Specificed model type not allowed")
    if bnn_type == 'none':
        return create_resnet_model(param, activation_fcn, input_shape,ds_size)
    elif bnn_type == 'tfp_flipout':
        return create_flipout_resnet_model(param,activation_fcn, input_shape,ds_size)
    elif bnn_type == 'tfp_reparameterization':
        return create_bayesian_reparam_resnet_model(param,activation_fcn, input_shape,ds_size)
    elif bnn_type == 'dropout':
        return create_MC_dropout_resnet_model(param,activation_fcn, input_shape,ds_size)
    