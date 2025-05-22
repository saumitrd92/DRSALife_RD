import os
from datetime import datetime
import tensorboard
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np
import network as net
import data_prep as dp
import keras_tuner
import argparse
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

def training_code(params):
    ################################
    ### Preprocess Training Data ###
    ################################

    dp.prep_training_data()

    # Define the Keras TensorBoard callback.
    dt_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir="logs/fit/" + dt_stamp
    params['logdir'] = logdir
    tensorboard_callback = TensorBoard(log_dir=params['logdir'])

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # Load training data 
    train_x = np.load('sim_data/training/x.npy')
    train_x_neigh = np.load('sim_data/training/x_neigh.npy')
    train_x_plus = np.load('sim_data/training/x_plus.npy')
    train_x_dot = np.load('sim_data/training/x_dot.npy')

    # restore np.load for future normal usage
    np.load = np_load_old

    ae, pipeline = net.pipeline(params)

    ############################
    ### Training Autoencoder ###
    ############################

    ae.fit([train_x,train_x_neigh], [train_x,train_x], epochs=100, batch_size=10000, verbose=0,callbacks=[tensorboard_callback],shuffle=True)

    #########################
    ### Training Pipeline ###
    #########################
    # checkpoint
    filepath="Realizations/trial_"+params['trial_id']+"_weights-improvement-{epoch:02d}.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=500)
    earlystop = EarlyStopping(monitor="loss", min_delta=1e-6, patience=500, verbose=0)
    callbacks_list = [tensorboard_callback,earlystop]
    history = pipeline.fit([train_x,train_x_neigh], train_x_dot, epochs=10000, batch_size=10000, verbose=0,shuffle=True,callbacks=callbacks_list)

    pipeline.save_weights("Realizations/trial_"+params['trial_id']+'_pipeline.h5')

    validation_loss = history.history['custom_loss'][-1]
    return validation_loss

class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters

        params={}

        params['trial_id'] = trial.trial_id

        ### Experiment Parameters ###

        params['state_features'] = 2

        ### Hyper Parameters ###
        params['hp_num_neighbors'] = 4

        # params['hp_num_complex_pairs'] = hp.Int("hp_num_complex_pairs", min_value=2, max_value=5, default=3, step=1)
        # params['hp_num_real'] = hp.Int("hp_num_real", min_value=0, max_value=10, default=3, step=1)

        params['hp_latent_space'] = hp.Int("hp_latent_space", min_value=5, max_value=50, default=24, step=1)

        params['hp_beta_units'] = hp.Choice("hp_beta_units", [8,16], default=8) #hp.Fixed("hp_beta_units", value=8)

        enc_dec_size =  hp.Choice("hp_enc_dec_size", [8,16], default=16) #hp.Fixed("hp_enc_dec_size", value=16)
        params['hp_phi_enc_units'] = enc_dec_size
        params['hp_psi_enc_units'] = enc_dec_size
        params['hp_psi_dec_units'] = enc_dec_size

        params['hp_delta_units'] = hp.Choice("hp_delta_units", [8,16], default=8) #hp.Fixed("hp_delta_units", value=8)
        
        dropout =  hp.Choice("hp_dropout", [0.1,0.2,0.3], default=0.2) # hp.Fixed("hp_dropout", value=0.1)
        params['hp_dropout'] = dropout

        return training_code(params)

# np.random.seed(30091992)
tuner = MyTuner(max_trials=50, overwrite=False, directory='tmp')

tuner.search(callbacks=[TensorBoard(log_dir="logs/fit/")])