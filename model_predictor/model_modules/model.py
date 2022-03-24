"""
    Andrew Pfau
    Sona Classifier

    This is the main model file that train_model, train_model_k_fold, and saved_model will reference.
    This file only builds and returns the models. All models are stored in their own files within the
    models folder.
"""


from tensorflow import keras
from tensorflow.keras import layers

# import models stored in other files
from model_modules.dev_model_bnn import build_model as build_dev_bnn_model
from model_modules.cnn_rect_model import build_model as build_cnn_rect_model
from model_modules.pretrained_models import vgg_pretrained, inception_pretrained, pretrained_model, mobilenet_pretrained
from model_modules.resnet_models import resnet_v1_model, resnet_v2_model
from model_modules.simple_models import simple_model, simple_BNN_VI_model, simple_MC_model
from model_modules.resnet_bnn import build_model as build_resnet_bnn_model
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
import tensorflow_probability as tfp


# make the model_dict a global variable in this file so that the main.py can import and reference it.
# We want the sanity_checker function in main.py to make sure that the passes model is a key in model_dict,
# this prevents having 2 lists of allowable models that have to be changed.

# When creating a new model add its name here as the key and its creation function as the value
model_dict = {"simple":simple_model, "resnet1":resnet_v1_model, "resnet2":resnet_v2_model, "vgg":vgg_pretrained, 
              "inception":inception_pretrained, "mobilenet":mobilenet_pretrained, "pretrained":pretrained_model,
              'simple_mc_model':simple_MC_model,'simple_bnn_vi_model':simple_BNN_VI_model, 'cnn_model':build_cnn_rect_model,
              'cnn_model_hparam':build_cnn_rect_model, 'dev_bnn_model':build_dev_bnn_model, 'dev_model_hparam':build_dev_bnn_model,
              'resnet_bnn':build_resnet_bnn_model}


def build_model(params, model_file, input_shape, activation_fcn,ds_size,hparams=None):
    bnn = False
    
    if params['bnn_type'] != 'none':
        bnn = True
    
    model_choice = params['model'].lower()
    
    # dev_model_hparam is the same as dev_model but includes hparam searching
    
    if model_choice == 'cnn_model' or model_choice == 'cnn_model_hparam':
        return build_cnn_rect_model(params, input_shape, activation_fcn,hparams=hparams)
    if model_choice == 'dev_bnn_model' or model_choice == 'dev_model_hparam':
        return build_dev_bnn_model(params, input_shape, activation_fcn,ds_size,hparams=hparams)
    elif model_choice == 'resnet_bnn':
        return build_resnet_bnn_model(params, input_shape, activation_fcn,ds_size,hparams=hparams)
    elif model_choice == 'simple_bnn_vi_model':
        return simple_BNN_VI_model(input_shape, params['num_classes'], activation_fcn, model_file, params['resnet_depth'],ds_size)

    return model_dict[model_choice](input_shape, params['num_classes'], activation_fcn, model_file, params['resnet_depth'])
