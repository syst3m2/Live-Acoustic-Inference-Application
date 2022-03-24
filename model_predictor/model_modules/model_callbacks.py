import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import os
from models.clr import CyclicLR
import numpy as np

def make_callbacks(params):
    '''
        Wrapper function to read the list of callbacks passed as a parameter
        to main.py and build the callbacks list.
        Params is the arguments to main.py, callbacks are passed as a list to
        --callbacks.
    '''
    checkpoints = []

    #if 'checkpoint' in params.callbacks:
    #    checkpoints.append()
    if 'early_stop'  in params.callbacks:
        checkpoints.append(_make_early_stop(params))
    if 'checkpoint'  in params.callbacks:
        checkpoints.append(_make_checkpoint(params))
    if 'lr_schedule' in params.callbacks:
        checkpoints.append(_make_lr_scheduler(params))
    if 'clr_schedule' in params.callbacks:
        checkpoints.append(_make_clr_scheduler(params))
    if 'tensorboard' in params.callbacks:
        checkpoints.append(_make_tensorboard(params))
    if 'csv_saver'   in params.callbacks:
        checkpoints.append(_make_csv_log(params))
    if 'kl_loss_scheduler' in params.callbacks:
        checkpoints.append(_makekl_loss_scheduler(params))
    if 'kl_loss_scheduler2' in params.callbacks:
        checkpoints.append(_makekl_loss_scheduler2(params))
    if 'reduce_lr' in params.callbacks:
        checkpoints.append(_make_reduce_lr_cb(params))
    return checkpoints


def _makekl_loss_scheduler(params):
    kl_loss_scheduler = KLLossScheduler()
    return kl_loss_scheduler

def _makekl_loss_scheduler2(params):
    kl_loss_scheduler2 = KLLossScheduler2()
    return kl_loss_scheduler2

def _make_early_stop(params):
    checkpoint = tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                                min_delta=0,
                                                patience=90,
                                                verbose=1,
                                                mode='auto')
    return checkpoint

def _make_checkpoint(params):
    # this saves the entire model, weights and model graphs
    # different checkpoint file name is used for k-fold vs not k-fold
    monitor_value = 'val_accuracy'
    save_name = "checkpoint{epoch:02d}--{val_accuracy:.2f}.h5"
    
    # need to monitor something different for regression
    if "regression" in params.model_type:
        monitor_value = 'mse'
        save_name = "checkpoint{epoch:02d}--{mse:.2f}.h5"

    if(params.mode == 'train-k-fold'):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(params.checkpoint_dir,"checkpoint-kfold.h5"),
                                                    monitor= monitor_value,
                                                    verbose=1,
                                                    mode='auto',
                                                    save_best_only=True,
                                                    save_freq='epoch')
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(params.checkpoint_dir, save_name),
                                                monitor= monitor_value,
                                                verbose=1,
                                                mode='auto',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                save_freq='epoch')
    return checkpoint

def _make_lr_scheduler(params):
    checkpoint = tf.keras.callbacks.LearningRateScheduler(schedule_step, verbose = 1)
    return checkpoint

def schedule_step(epoch, lr):
    # step function, divide lr by 10 every 50 epochs
    if epoch > 0 and (epoch % 50) == 0:
        return lr * 0.1
    else:
        return lr 

def schedule_exp(epoch):
    # function to compute learning rate based on epoch for learning rate scheduler
    # Learn rate schedule, try exponential, dropping every 10 epochs '''
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.01 * (10 - epoch))

def _make_clr_scheduler(params):
    clr = CyclicLR()
    return clr

def _make_tensorboard(params):
    checkpoint = tf.keras.callbacks.TensorBoard(log_dir=params.checkpoint_dir)
    return checkpoint

def _make_csv_log(params):
    checkpoint = tf.keras.callbacks.CSVLogger( os.path.join(params.checkpoint_dir, "log.csv"), append=True, separator=';')
    return checkpoint

def _make_reduce_lr_cb(params):
    reduce_lr =  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                            factor=0.2,
                                            patience=50,
                                            cooldown=0,
                                            min_lr=0.00001,
                                            verbose=1,
                                            mode ='auto')
    return reduce_lr

class KLLossScheduler(tf.keras.callbacks.Callback):
    
    def __init__(self, n_silent_epoch=5, n_annealing_epoch=25, verbose=1):
        self.n_silent_epoch = n_silent_epoch
        self.n_annealing_epoch = n_annealing_epoch
        self.verbose = verbose
        super(KLLossScheduler, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        
    
        kl_weight = (epoch - self.n_silent_epoch + 1) / self.n_annealing_epoch
        kl_weight = np.maximum(0.0, np.minimum(kl_weight, 1.0))
        self.kl_weight = kl_weight
        if self.verbose > 0:
            print('\nEpoch: {}, KL Divergence Loss Weight = {:.6f}'.format(epoch+1, kl_weight))
        for l in self.model.layers:
            for id_w, w in enumerate(l.weights):
                if 'kl_loss_weight' in w.name:
                    l_weights = l.get_weights()
                    l.set_weights([*l_weights[:id_w], kl_weight, *l_weights[id_w+1:]])


class KLLossScheduler2(tf.keras.callbacks.Callback):
    def __init__(self, update_per_batch=True, n_silent_epoch=5, n_annealing_epoch=50,steps_per_epoch=1, verbose=0):
        self.update_per_batch = update_per_batch
        self.n_silent_epoch = n_silent_epoch
        self.n_annealing_epoch = n_annealing_epoch
        self.verbose = verbose
        self.steps_per_epoch = steps_per_epoch
        super(KLLossScheduler2, self).__init__()
    def on_train_batch_begin(self, batch, logs=None):
        if self.update_per_batch:
            
            
            #idx_total_batch = (self.epoch - self.n_silent_epoch) * self.steps_per_epoch + batch + 1
            #kl_weight = (idx_total_batch / self.steps_per_epoch) / self.n_annealing_epoch
            #kl_weight = np.maximum(0.0, np.minimum(kl_weight, 1.0))
            #self.kl_weight = kl_weight
            
            kl_weight = kl_weight = (2**(self.steps_per_epoch-batch))/((2**(self.steps_per_epoch)-1))
            kl_weight = np.maximum(0.0, np.minimum(kl_weight, 1.0))
            self.kl_weight = kl_weight
            
            if self.verbose > 0:
                print('\nBatch: {}, KL Divergence Loss Weight = {:.6f}'.format(batch+1, kl_weight))
            for l in self.model.layers:
                for id_w, w in enumerate(l.weights):
                    if 'kl_loss_weight' in w.name:
                        l_weights = l.get_weights()
                        l.set_weights([*l_weights[:id_w], kl_weight, *l_weights[id_w+1:]])
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if not self.update_per_batch:
            kl_weight = (epoch - self.n_silent_epoch + 1) / self.n_annealing_epoch
            kl_weight = np.maximum(0.0, np.minimum(kl_weight, 1.0))
            self.kl_weight = kl_weight
            if self.verbose > 0:
                print('\nEpoch: {}, KL Divergence Loss Weight = {:.6f}'.format(epoch+1, kl_weight))
            for l in self.model.layers:
                for id_w, w in enumerate(l.weights):
                    if 'kl_loss_weight' in w.name:
                        l_weights = l.get_weights()
                        l.set_weights([*l_weights[:id_w], kl_weight, *l_weights[id_w+1:]])