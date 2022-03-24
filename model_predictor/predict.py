#import os
#import sys
#sys.path.append("..")
import tensorflow as tf
from tensorflow import keras

#from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss, multilabel_confusion_matrix, roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
#import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import pickle
# imports for plotting
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss, multilabel_confusion_matrix, roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from model_modules.model_optimizers import get_optimizer, get_loss
from model_modules.model import build_model
import copy
'''
def BNN_predict(num_classes,to_test):
    pred_vi=np.zeros((len(to_test),num_classes))
    pred_max_p_vi=np.zeros((len(to_test)))
    pred_std_vi=np.zeros((len(to_test)))
    entropy_vi = np.zeros((len(to_test)))
    var=  np.zeros((len(to_test)))
    for i in range(0,len(to_test)):
        preds = to_test[i]
        pred_vi[i]=np.mean(preds,axis=0)#mean over n runs of every proba class
        pred_max_p_vi[i]=np.argmax(np.mean(preds,axis=0))#mean over n runs of every proba class
        pred_std_vi[i]= np.sqrt(np.sum(np.var(preds, axis=0)))
        var[i] =  np.sum(np.var(preds, axis=0))
        entropy_vi[i] = -np.sum( pred_vi[i] * np.log2(pred_vi[i] + 1E-14)) #Numerical Stability
    pred_vi_mean_max_p=np.array([pred_vi[i][np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    nll_vi=-np.log(pred_vi_mean_max_p)
    return pred_vi,pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var
'''
def BNN_predict(num_classes,to_test):
    pred_vi=np.zeros((len(to_test),num_classes))
    pred_max_p_vi=np.zeros((len(to_test)))
    pred_std_vi=np.zeros((len(to_test)))
    entropy_vi = np.zeros((len(to_test)))
    norm_entropy_vi =  np.zeros((len(to_test)))
    epistemic = np.zeros((len(to_test)))
    aleatoric = np.zeros((len(to_test)))
    var=  np.zeros((len(to_test)))
    for i in range(0,len(to_test)):
        preds = to_test[i]
        pred_vi[i]=np.mean(preds,axis=0)#mean over n runs of every proba class
        pred_max_p_vi[i]=np.argmax(np.mean(preds,axis=0))#mean over n runs of every proba class
        pred_std_vi[i]= np.sqrt(np.sum(np.var(preds, axis=0)))
        var[i] =  np.sum(np.var(preds, axis=0))
        entropy_vi[i] = -np.sum( pred_vi[i] * np.log2(pred_vi[i] + 1E-14)) #Numerical Stability
        epistemic[i] = np.sum(np.mean(preds**2, axis=0) - np.mean(preds, axis=0)**2)
        aleatoric[i] = np.sum(np.mean(preds*(1-preds), axis=0))
        norm_entropy_vi[i] = entropy_vi[i]/np.log2(2^num_classes)
    pred_vi_mean_max_p=np.array([pred_vi[i][np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    nll_vi=-np.log(pred_vi_mean_max_p)


    return pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric

    #results = {'pred':pred_vi,'pred_max_p':pred_max_p_vi, 'pred_vi_mean_max':pred_vi_mean_max_p, 'entropy':entropy_vi,
    #            'nll':nll_vi, 'pred_std':pred_std_vi, 'var':var, 'norm_entropy':norm_entropy_vi,'epistemic':epistemic,'aleatoric':aleatoric}
    #return results


def model_predict(params, dataset, model_file):
    if 'bnn_build' in params:
        bnn_build = params['bnn_build']
    else:
        bnn_build = False

    if bnn_build==True:
        # Need to add capability to record STD and ENTROPY with predictions
        # load the model

        # TODO make this a method shared between train and saved
        # set loss and final activation functions based on model type
        if params['model_type'] == "multi_class":
            loss_fcn = get_loss("crossentropy", params['label_smooth'])
            activation_fcn = "softmax"
        elif params['model_type'] == "multi_label":
            loss_fcn = get_loss("binarycrossentropy", params['label_smooth'])
            activation_fcn = "sigmoid"
        else:
            print("Specified model type not allowed")

        # calculate input shape parameters
        # calculate input shape parameters
        step = (params['overlap']/100) * \
            ((params['win_size'] * 0.001) * params['sample_rate'])
        time_axis = (int)((params['duration'] * params['sample_rate']) // step)
        # stft frequency axis is different than MFCC, time axises are the same
        if params['model_input'] == 'stft':
            freq_axis = (params['sample_pts']//2) + 1
        else:
            freq_axis = params['mel_bins']
        input_shape = (time_axis, freq_axis, params['channels'])

        model = build_model(params, model_file, input_shape, activation_fcn, 1)

        # compile model
        model.compile(optimizer=get_optimizer(params),
                        loss=loss_fcn,
                        metrics=[params['eval_metrics']])

        model.load_weights(model_file)

        probs = tf.stack([model.predict(dataset, verbose=1)
                        for _ in range(params['num_mc_inference'])], axis=0)

        # Update database model to include pickle file save? May not work in JSON
        #https://stackoverflow.com/questions/57642165/saving-python-object-in-postgres-table-with-pickle
        '''
        print(probs.shape)
        toSave = {
            'preds': probs,
            'trueLabels': true_l
        }
        print(toSave.keys())
        file = open(os.path.join(args.checkpoint_dir, "bnn.pkl"), 'wb')
        pickle.dump(toSave, file)
        file.close()
        '''
        
        preds = np.swapaxes(probs.numpy(),0,1)
        
        pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric = BNN_predict(5,preds)
        

        #bnn_predict = BNN_predict(5,preds)

        predict_label_list = []
        if params['model_type'] == "multi_class":
            predict_labels = pred_vi.argmax(-1)
            #predict_labels = bnn_predict['pred'].argmax(-1)

            # Returns predict class labels along with entropy and STD to measure uncertainty
            prediction_dictionary = {0:"Class A", 1:"Class B", 2:"Class C", 3:"Class D", 4:"Class E"}
            predict_labels = [prediction_dictionary[x] for x in pred_max_p_vi]
            predict_labels = np.array([predict_labels, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric])
            predict_labels = predict_labels.T

            # Convert each to dictionary
            for i in range(0,len(predict_labels)):
                predict = {'pred':predict_labels[i][0],'pred_max_p':predict_labels[i][1], 'pred_vi_mean_max':predict_labels[i][2], 'entropy':predict_labels[i][3],
                'nll':predict_labels[i][4], 'pred_std':predict_labels[i][5], 'var':predict_labels[i][6], 'norm_entropy':predict_labels[i][7],'epistemic':predict_labels[i][8],'aleatoric':predict_labels[i][9]}
                predict_label_list.append([predict])
            predict_labels = predict_label_list
            #predict_labels = predict_labels.tolist()

            #predict_labels = np.array([predict_labels, bnn_predict])
            #predict_labels = predict_labels.T
            #predict_labels = predict_labels.tolist()

        elif params['model_type'] == "multi_label":
            pred_vi_working = copy.deepcopy(pred_vi)
            #pred_vi_working = copy.deepcopy(bnn_predict['pred'])

            for item in pred_vi_working:
                if  max(item) < 0.5:
                    k = np.argmax(item)
                    for i in range(len(item)):
                        item[i] = 0
                    item[k] = 1
                else:
                    for i in range(len(item)):
                        if item[i] >=0.5:
                            item[i] = 1
                        else:
                            item[i] = 0

            predict_labels = pred_vi_working
            prediction_dictionary = {0:"Class A", 1:"Class B", 2:"Class C", 3:"Class D", 4:"Class E"}
            predict_labels = [prediction_dictionary[x] for x in predict_labels]
            predict_labels = np.array([predict_labels, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric])
            predict_labels = predict_labels.T
            #predict_labels = predict_labels.tolist()

            # Convert each to dictionary
            for i in range(0,len(predict_labels)):
                predict = {'pred':predict_labels[i][0],'pred_max_p':predict_labels[i][1], 'pred_vi_mean_max':predict_labels[i][2], 'entropy':predict_labels[i][3],
                'nll':predict_labels[i][4], 'pred_std':predict_labels[i][5], 'var':predict_labels[i][6], 'norm_entropy':predict_labels[i][7],'epistemic':predict_labels[i][8],'aleatoric':predict_labels[i][9]}
                predict_label_list.append([predict])
            predict_labels = predict_label_list

            #predict_labels = np.array([pred_vi_working, bnn_predict])
            #predict_labels = predict_labels.T
            #predict_labels = predict_labels.tolist()


    else:
        acoustic_inference_model = tf.keras.models.load_model(model_file)
        predict_probs = acoustic_inference_model.predict(dataset)
        predict_labels = predict_probs.argmax(axis=-1)
        prediction_dictionary = {0:"Class A", 1:"Class B", 2:"Class C", 3:"Class D", 4:"Class E"}
        predict_labels = [[{'pred':prediction_dictionary[x]}] for x in predict_labels]
        
    
    return predict_labels
        
    
    