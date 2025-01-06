import numpy as np
import pandas as pd
import random 
import multiprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import copy

from cosmopower import cosmopower_NN
from scipy import interpolate
from scipy import signal

# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu' 
print('using', device, 'device \n')

# disable eager execution mode
#tf.compat.v1.disable_eager_execution()

# setting the seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

train_params=np.load('../data/training_params/training_iBamm_U90W90W90_cosmogridv1_desy3_full_mask_sobol_6parameter_2.5e5_nodes.npz')
#train_params = np.load('../../integrated_3pcf_emulation_mcmc/NN/training_params/training_iBamm_LHS_5parameter_1e5_U90W90W90_final_prior.npz')
model_parameters = train_params.files
print(model_parameters)

features = np.log10(np.load('../data/training_features/training_iBamm_params_sobol_cosmogridv1_desy3_2.6e5_GM_U90W90W90_log.npz')['model'])
#features = np.log10(np.load('../../integrated_3pcf_emulation_mcmc/NN/training_features/training_P_params_LHS_1e5_nl_l100_final_prior.npz')['model'])

ell = np.unique(np.logspace(np.log10(2), np.log10(15000), 88).astype(int))
modes = ell
cp_nn = cosmopower_NN(parameters=model_parameters, modes=modes, n_hidden = [512, 512, 1024, 512, 512], verbose=True)
#cp_nn = cosmopower_NN(parameters=model_parameters, modes=modes, n_hidden = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], verbose=True)
with tf.device(device):
    # train
    #cp_nn.train(training_parameters=train_params, training_features=features, filename_saved_model='../data/trained_models/nn_emulator_log10iBapp_U90W90W90_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1', validation_split=0.1, learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], batch_sizes=[256, 256, 256, 256, 256], gradient_accumulation_steps = [1, 1, 1, 1, 1], patience_values = [100, 100, 100, 100, 100], max_epochs = [1000, 1000, 1000, 1000, 1000])
# learning_rates=[1e-2, 1e-2/np.sqrt(10), 1e-3, 1e-3/np.sqrt(10), 1e-4]
    cp_nn.train(training_parameters=train_params, training_features=features, filename_saved_model='../data/trained_models/nn_emulator_log10iBamm_U90W90W90_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1', validation_split=0.2, learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], batch_sizes=[128, 128, 128, 128, 128], gradient_accumulation_steps = [1, 1, 1, 1, 1], patience_values = [100, 100, 100, 100, 100], max_epochs = [1000, 1000, 1000, 1000, 1000])

