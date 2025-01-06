import time
start = time.time()

import random
import math as m
import numpy as np
import healpy as hp
import tensorflow as tf

tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_probability as tfp

from affine import *
from scipy import interpolate
from cosmopower import cosmopower_NN

# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print('using', device, 'device \n')

#### set-up parameters ####
pi = tf.constant(m.pi, dtype=tf.float32)
speed_c = tf.constant(299792.458, dtype=tf.float32)

z_source_tf = tf.constant(2.5, dtype=tf.float32)

ell = np.unique(np.logspace(np.log10(2), np.log10(15000), 88).astype(int))
ell_tf = tf.convert_to_tensor(ell)

fullell_tf = tf.linspace(2.0, 14998, 14997)
fullell_kitching_tf = tf.linspace(2.5, 14998.5, 14997)
fullell_kitching = np.linspace(2.5, 14998.5, 14997)
n_ell = tf.size(ell_tf)
n_fullell = tf.size(fullell_tf)

z_list_integ = np.linspace(0.01, 2.5, 250)
z_list_integ_tf = tf.linspace(0.01, 2.5, 250)
n_z_los = tf.size(z_list_integ_tf)

# Dimension of parameter space sampled
ndim = 14

#### Hartlap correction factor for the precision matrix ####
# Number of simulation realizations used to estimate data covariance matrix
N_s = 16000
# Number of bins in the data vector
N_d = 300
hartlap_factor = tf.constant((N_s - N_d - 2)/(N_s - 1), dtype=tf.float32)

#### Percival power index m in the posterior ####
# The factor B in the expression of m, see appendix B of https://arxiv.org/pdf/2108.10402.pdf
#B = (N_s - N_d - 2)/(N_s - N_d - 1)/(N_s - N_d - 4)
#m = ndim + 2 + (N_s - 1 + B*(N_d - ndim))/(1 + B*(N_d - ndim))
#m = tf.constant(m, dtype=tf.float32)
#N_s_tf = tf.constant(N_s, dtype=tf.float32)
A = 2/(N_s - N_d - 1)/(N_s - N_d - 4)
B = (N_s - N_d - 2)/(N_s - N_d - 1)/(N_s - N_d - 4)
percival_factor = 1 + A + B*(ndim + 1)
percival_factor = tf.constant(percival_factor, dtype=tf.float32)
ds_factor = 1 + B*(N_d - ndim)
ds_factor = tf.constant(ds_factor, dtype=tf.float32)

# Number of Markov Chains initiated simultaneously
nwalkers = 500

# Number of Markov Chain steps 
total_steps = 5000
burnin_steps = 4000

# Set up the pixel window functions and prepare it for all sampled nodes
nside = 512
# Add the pixel window function correction
pixwin = hp.pixwin(nside, lmax=np.max(ell)) 
pixwin_ell = np.arange(len(pixwin))                                         # pixwin creates window function for ell=0 to 3*nside-1
pixwin = pixwin[np.intersect1d(ell, pixwin_ell, return_indices=True)[2]]    # match pixwin function to correct ells
pixwin = np.append(pixwin, np.zeros(len(ell) - len(pixwin)))                # Add zeros to match length

# Expand the pixwin functions to all sampled nodes
pixwin = pixwin.reshape((-1,n_ell))
pixwin = tf.convert_to_tensor(pixwin, dtype=tf.float32)

#### Read in the angular bins and Fourier transform kernel for different correlations ####
r_xi = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_xi_angles_cen_arcmins_10_250_15_bins.tab", usecols=(0))
r_min_xi = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_xi_angles_min_arcmins_10_250_15_bins.tab", usecols=(0))
r_max_xi = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_xi_angles_max_arcmins_10_250_15_bins.tab", usecols=(0))
r_zeta_U50W50 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W50_angles_cen_arcmins_10_85_10_bins.tab", usecols=(0))
r_min_zeta_U50W50 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W50_angles_min_arcmins_50_10_85_10_bins.tab", usecols=(0))
r_max_zeta_U50W50 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W50_angles_max_arcmins_50_10_85_10_bins.tab", usecols=(0))
r_zeta_U70W70 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W70_angles_cen_arcmins_10_131_12_bins.tab", usecols=(0))
r_min_zeta_U70W70 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W70_angles_min_arcmins_70_10_131_12_bins.tab", usecols=(0))
r_max_zeta_U70W70 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W70_angles_max_arcmins_70_10_131_12_bins.tab", usecols=(0))
r_zeta_U90W90 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W90_angles_cen_arcmins_10_162_13_bins.tab", usecols=(0))
r_min_zeta_U90W90 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W90_angles_min_arcmins_90_10_162_13_bins.tab", usecols=(0))
r_max_zeta_U90W90 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W90_angles_max_arcmins_90_10_162_13_bins.tab", usecols=(0))
r_zeta_U110W110 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W110_angles_cen_arcmins_10_201_14_bins.tab", usecols=(0))
r_min_zeta_U110W110 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W110_angles_min_arcmins_110_10_201_14_bins.tab", usecols=(0))
r_max_zeta_U110W110 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W110_angles_max_arcmins_110_10_201_14_bins.tab", usecols=(0))
r_zeta_U130W130 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W130_angles_cen_arcmins_10_250_15_bins.tab", usecols=(0))
r_min_zeta_U130W130 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W130_angles_min_arcmins_130_10_250_15_bins.tab", usecols=(0))
r_max_zeta_U130W130 = np.loadtxt("../data/testing_grids/updated_testing/angular_bins/alpha_iZ_W130_angles_max_arcmins_130_10_250_15_bins.tab", usecols=(0))

n_angular_bins_xi = len(r_xi)
n_angular_bins_iZ50 = len(r_zeta_U50W50)
n_angular_bins_iZ70 = len(r_zeta_U70W70)
n_angular_bins_iZ90 = len(r_zeta_U90W90)
n_angular_bins_iZ110 = len(r_zeta_U110W110)
n_angular_bins_iZ130 = len(r_zeta_U130W130)

A_2pt_U50W50 = np.loadtxt('../data/testing_grids/updated_testing/angular_bins/A2pt_bin_averaged_iZ_W50_alpha_10_85_10.dat', usecols=(3), unpack=True)
A_2pt_U70W70 = np.loadtxt('../data/testing_grids/updated_testing/angular_bins/A2pt_bin_averaged_iZ_W70_alpha_10_131_12.dat', usecols=(3), unpack=True)
A_2pt_U90W90 = np.loadtxt('../data/testing_grids/updated_testing/angular_bins/iZ_A2pt_W+90_alpha_10_162_13.dat', usecols=(3), unpack=True)
A_2pt_U110W110 = np.loadtxt('../data/testing_grids/updated_testing/angular_bins/A2pt_bin_averaged_iZ_W110_alpha_10_201_14.dat', usecols=(3), unpack=True)
A_2pt_U130W130 = np.loadtxt('../data/testing_grids/updated_testing/angular_bins/A2pt_bin_averaged_iZ_W130_alpha_10_250_15.dat', usecols=(3), unpack=True)
FT_plus_binave_zeta_U50W50 = np.load('../data/testing_grids/updated_testing/angular_bins/iZp_W50_10_85_10_bin_averaged_values.npy').T
FT_minus_binave_zeta_U50W50 = np.load('../data/testing_grids/updated_testing/angular_bins/iZm_W50_10_85_10_bin_averaged_values.npy').T
FT_plus_binave_zeta_U70W70 = np.load('../data/testing_grids/updated_testing/angular_bins/iZp_W70_10_131_12_bin_averaged_values.npy').T
FT_minus_binave_zeta_U70W70 = np.load('../data/testing_grids/updated_testing/angular_bins/iZm_W70_10_131_12_bin_averaged_values.npy').T
FT_plus_binave_zeta_U90W90 = np.load('../data/testing_grids/updated_testing/angular_bins/iZp_W90_10_162_13_bin_averaged_values.npy').T
FT_minus_binave_zeta_U90W90 = np.load('../data/testing_grids/updated_testing/angular_bins/iZm_W90_10_162_13_bin_averaged_values.npy').T
FT_plus_binave_zeta_U110W110 = np.load('../data/testing_grids/updated_testing/angular_bins/iZp_W110_10_201_14_bin_averaged_values.npy').T
FT_minus_binave_zeta_U110W110 = np.load('../data/testing_grids/updated_testing/angular_bins/iZm_W110_10_201_14_bin_averaged_values.npy').T
FT_plus_binave_zeta_U130W130 = np.load('../data/testing_grids/updated_testing/angular_bins/iZp_W130_10_250_15_bin_averaged_values.npy').T
FT_minus_binave_zeta_U130W130 = np.load('../data/testing_grids/updated_testing/angular_bins/iZm_W130_10_250_15_bin_averaged_values.npy').T
FT_plus_binave_xi = np.load('../data/testing_grids/updated_testing/angular_bins/xip_10_250_15_bin_averaged_values.npy').T
FT_minus_binave_xi = np.load('../data/testing_grids/updated_testing/angular_bins/xim_10_250_15_bin_averaged_values.npy').T

A_2pt_U50W50 = tf.convert_to_tensor(A_2pt_U50W50, dtype=tf.float32)
A_2pt_U70W70 = tf.convert_to_tensor(A_2pt_U70W70, dtype=tf.float32)
A_2pt_U90W90 = tf.convert_to_tensor(A_2pt_U90W90, dtype=tf.float32)
A_2pt_U110W110 = tf.convert_to_tensor(A_2pt_U110W110, dtype=tf.float32)
A_2pt_U130W130 = tf.convert_to_tensor(A_2pt_U130W130, dtype=tf.float32)
FT_plus_binave_zeta_U50W50 = tf.convert_to_tensor(FT_plus_binave_zeta_U50W50, dtype=tf.float32)
FT_minus_binave_zeta_U50W50 = tf.convert_to_tensor(FT_minus_binave_zeta_U50W50, dtype=tf.float32)
FT_plus_binave_zeta_U70W70 = tf.convert_to_tensor(FT_plus_binave_zeta_U70W70, dtype=tf.float32)
FT_minus_binave_zeta_U70W70 = tf.convert_to_tensor(FT_minus_binave_zeta_U70W70, dtype=tf.float32)
FT_plus_binave_zeta_U90W90 = tf.convert_to_tensor(FT_plus_binave_zeta_U90W90, dtype=tf.float32)
FT_minus_binave_zeta_U90W90 = tf.convert_to_tensor(FT_minus_binave_zeta_U90W90, dtype=tf.float32)
FT_plus_binave_zeta_U110W110 = tf.convert_to_tensor(FT_plus_binave_zeta_U110W110, dtype=tf.float32)
FT_minus_binave_zeta_U110W110 = tf.convert_to_tensor(FT_minus_binave_zeta_U110W110, dtype=tf.float32)
FT_plus_binave_zeta_U130W130 = tf.convert_to_tensor(FT_plus_binave_zeta_U130W130, dtype=tf.float32)
FT_minus_binave_zeta_U130W130 = tf.convert_to_tensor(FT_minus_binave_zeta_U130W130, dtype=tf.float32)
FT_plus_binave_xi = tf.convert_to_tensor(FT_plus_binave_xi, dtype=tf.float32)
FT_minus_binave_xi = tf.convert_to_tensor(FT_minus_binave_xi, dtype=tf.float32)

covariance = np.load('../data/testing_grids/updated_testing/cov/cov_2PCF.npy')
covariance = covariance*ds_factor/percival_factor
#covariance = np.cov(data_matrix, rowvar=False, ddof=1)*f_sky_des/f_sky_lsst
inv_covariance = np.linalg.inv(covariance)
inv_covariance = tf.convert_to_tensor(inv_covariance, dtype=tf.float32)

data_vec_2PCF = np.load("../data/testing_grids/updated_testing/cosmogridv1_fiducial_2PCF_data_vector.npy")
#data_vec = data_vec_U90W90[:90]
data_vec = tf.convert_to_tensor(data_vec_2PCF, dtype=tf.float32)

#### prior boudaries for cosmological parameters ####
Omega_min = 0.1
Omega_max = 0.5
As_min = np.log(10**10 * 0.5e-9)
As_max = np.log(10**10 * 7.0e-9)
w0_min = -3.5
w0_max = -0.33
h_min = 0.55
h_max = 0.91
cmin_min = 0.5
cmin_max = 6.0
#delta_z_min = -0.1
#delta_z_max = 0.1
#m_min = -0.1
#m_max = 0.1
A_IA_NLA_min = -5.0
A_IA_NLA_max = 5.0
#alpha_IA_NLA_min = -5.0
#alpha_IA_NLA_max = 5.0

mu_delta_z1 = 0.0
mu_delta_z2 = 0.0
mu_delta_z3 = 0.0
mu_delta_z4 = 0.0
mu_m1 = -0.006
mu_m2 = -0.02
mu_m3 = -0.024
mu_m4 = -0.037
std_delta_z1 = 0.018
std_delta_z2 = 0.015
std_delta_z3 = 0.011
std_delta_z4 = 0.017
std_m1 = 0.009
std_m2 = 0.008
std_m3 = 0.008
std_m4 = 0.008

mu_delta_z1_tf = tf.constant(mu_delta_z1, dtype=tf.float32)
mu_delta_z2_tf = tf.constant(mu_delta_z2, dtype=tf.float32)
mu_delta_z3_tf = tf.constant(mu_delta_z3, dtype=tf.float32)
mu_delta_z4_tf = tf.constant(mu_delta_z4, dtype=tf.float32)
mu_m1_tf = tf.constant(mu_m1, dtype=tf.float32)
mu_m2_tf = tf.constant(mu_m2, dtype=tf.float32)
mu_m3_tf = tf.constant(mu_m3, dtype=tf.float32)
mu_m4_tf = tf.constant(mu_m4, dtype=tf.float32)
std_delta_z1_tf = tf.constant(std_delta_z1, dtype=tf.float32)
std_delta_z2_tf = tf.constant(std_delta_z2, dtype=tf.float32)
std_delta_z3_tf = tf.constant(std_delta_z3, dtype=tf.float32)
std_delta_z4_tf = tf.constant(std_delta_z4, dtype=tf.float32)
std_m1_tf = tf.constant(std_m1, dtype=tf.float32)
std_m2_tf = tf.constant(std_m2, dtype=tf.float32)
std_m3_tf = tf.constant(std_m3, dtype=tf.float32)
std_m4_tf = tf.constant(std_m4, dtype=tf.float32)

#### Read in emulators and data scalers ####
cp_nn_P = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10P_minus4_to_plus6_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1_updated')
cp_nn_iBapp_U50W50 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBapp_U50W50W50_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBamm_U50W50 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBamm_U50W50W50_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBapp_U70W70 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBapp_U70W70W70_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBamm_U70W70 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBamm_U70W70W70_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBapp_U90W90 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBapp_U90W90W90_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBamm_U90W90 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBamm_U90W90W90_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBapp_U110W110 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBapp_U110W110W110_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBamm_U110W110 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBamm_U110W110W110_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBapp_U130W130 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBapp_U130W130W130_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')
cp_nn_iBamm_U130W130 = cosmopower_NN(restore=True, restore_filename='../data/trained_models/nn_emulator_log10iBamm_U130W130W130_desy3_2.5e5nodes_full_mask_sobol_6params_80ell_cosmogridv1')

iBamm_shift_constant_U50W50 = np.loadtxt('../data/training_params/input_param_scaler/constant_shift_iBamm_U50W50_cosmogridv1_desy3_full_mask_sobol_6parameter_2.5e5_nodes.dat')
iBamm_shift_constant_U50W50 = tf.constant(iBamm_shift_constant_U50W50, dtype=tf.float32)

iBamm_shift_constant_U70W70 = np.loadtxt('../data/training_params/input_param_scaler/constant_shift_iBamm_U70W70_cosmogridv1_desy3_full_mask_sobol_6parameter_2.5e5_nodes.dat')
iBamm_shift_constant_U70W70 = tf.constant(iBamm_shift_constant_U70W70, dtype=tf.float32)

iBamm_shift_constant_U90W90 = np.loadtxt('../data/training_params/input_param_scaler/constant_shift_iBamm_U90W90_cosmogridv1_desy3_full_mask_sobol_6parameter_2.5e5_nodes.dat')
iBamm_shift_constant_U90W90 = tf.constant(iBamm_shift_constant_U90W90, dtype=tf.float32)

iBamm_shift_constant_U110W110 = np.loadtxt('../data/training_params/input_param_scaler/constant_shift_iBamm_U110W110_cosmogridv1_desy3_full_mask_sobol_6parameter_2.5e5_nodes.dat')
iBamm_shift_constant_U110W110 = tf.constant(iBamm_shift_constant_U110W110, dtype=tf.float32)

iBamm_shift_constant_U130W130 = np.loadtxt('../data/training_params/input_param_scaler/constant_shift_iBamm_U130W130_cosmogridv1_desy3_full_mask_sobol_6parameter_2.5e5_nodes.dat')
iBamm_shift_constant_U130W130 = tf.constant(iBamm_shift_constant_U130W130, dtype=tf.float32)

emulator_H = tf.saved_model.load('../data/trained_models/H_chi_D_models/gp_model_3e3_inducing_variable_sgpr_H_3e4_nodes_standardscale/')
emulator_chi = tf.saved_model.load('../data/trained_models/H_chi_D_models/gp_model_3e3_inducing_variable_sgpr_chi_3e4_nodes_standardscale/')
emulator_D = tf.saved_model.load('../data/trained_models/H_chi_D_models/gp_model_3e3_inducing_variable_sgpr_D_3e4_nodes_standardscale/')

H_mean = np.loadtxt('../data/trained_models/H_chi_D_models/gp_model_3e4_nodes_H_mean.dat')
H_std = np.loadtxt('../data/trained_models/H_chi_D_models/gp_model_3e4_nodes_H_std.dat')
chi_mean = np.loadtxt('../data/trained_models/H_chi_D_models/gp_model_3e4_nodes_chi_mean.dat')
chi_std = np.loadtxt('../data/trained_models/H_chi_D_models/gp_model_3e4_nodes_chi_std.dat')
D_mean = np.loadtxt('../data/trained_models/H_chi_D_models/gp_model_3e4_nodes_D_mean.dat')
D_std = np.loadtxt('../data/trained_models/H_chi_D_models/gp_model_3e4_nodes_D_std.dat')

H_mean = tf.constant(H_mean, dtype=tf.float32)
chi_mean = tf.constant(chi_mean, dtype=tf.float32)
D_mean = tf.constant(D_mean, dtype=tf.float32)

H_std = tf.constant(H_std, dtype=tf.float32)
chi_std = tf.constant(chi_std, dtype=tf.float32)
D_std = tf.constant(D_std, dtype=tf.float32)
#### source redshift distribution ####

n_s_z_BIN_z_tab = np.loadtxt('../data/source_redshift/nofz_DESY3_source_BIN1.tab', usecols=(0))

n_s_z_BIN1_vals = np.loadtxt('../data/source_redshift/nofz_DESY3_source_BIN1.tab', usecols=(1), unpack=True)
n_s_z_BIN1_vals /= np.trapz(n_s_z_BIN1_vals, n_s_z_BIN_z_tab)

n_s_z_BIN2_vals = np.loadtxt('../data/source_redshift/nofz_DESY3_source_BIN2.tab', usecols=(1), unpack=True)
n_s_z_BIN2_vals /= np.trapz(n_s_z_BIN2_vals, n_s_z_BIN_z_tab)

n_s_z_BIN3_vals = np.loadtxt('../data/source_redshift/nofz_DESY3_source_BIN3.tab', usecols=(1), unpack=True)
n_s_z_BIN3_vals /= np.trapz(n_s_z_BIN3_vals, n_s_z_BIN_z_tab)

n_s_z_BIN4_vals = np.loadtxt('../data/source_redshift/nofz_DESY3_source_BIN4.tab', usecols=(1), unpack=True)
n_s_z_BIN4_vals /= np.trapz(n_s_z_BIN4_vals, n_s_z_BIN_z_tab)

n_s_z_BIN1 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN1_vals, fill_value=(0,0), bounds_error=False)
n_s_z_BIN2 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN2_vals, fill_value=(0,0), bounds_error=False)
n_s_z_BIN3 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN3_vals, fill_value=(0,0), bounds_error=False)
n_s_z_BIN4 = interpolate.interp1d(n_s_z_BIN_z_tab, n_s_z_BIN4_vals, fill_value=(0,0), bounds_error=False)

#### A scipy function which compute n(z) w.r.t different photometric redshift uncertainty ####

def n_s_z_compute(theta):
    
    theta = theta.numpy()
    z = np.repeat(z_list_integ.reshape(1,-1), theta.shape[0], axis=0)
    delta_z_1 = theta[:,5]
    delta_z_2 = theta[:,6]
    delta_z_3 = theta[:,7]
    delta_z_4 = theta[:,8]
    z_1 = z - delta_z_1[:,None]
    z_2 = z - delta_z_2[:,None]
    z_3 = z - delta_z_3[:,None]
    z_4 = z - delta_z_4[:,None]
    ns_z_BIN1_los = n_s_z_BIN1(z_1)
    ns_z_BIN2_los = n_s_z_BIN2(z_2)
    ns_z_BIN3_los = n_s_z_BIN3(z_3)
    ns_z_BIN4_los = n_s_z_BIN4(z_4)
    return ns_z_BIN1_los, ns_z_BIN2_los, ns_z_BIN3_los, ns_z_BIN4_los

#### A scipy function which interpolates the projected spectrum to all multipole numbers ###

def interpolation(spectrum):
    
    spectrum = spectrum.numpy()
    interp = interpolate.interp1d(ell, spectrum, kind='cubic', axis=1, fill_value='extrapolate')
    spectrum_new = interp(fullell_kitching)
    return spectrum_new

#### Function carrying out kitching correction and converting the integrated bispectrum to correlation function ####

@tf.function
def kitching_correction_and_FT_plus_zeta(theta, spectrum, FT_plus_kernel, n_angular_bins):
     
    input_spectrum = tf.Variable(spectrum)
    interp_spectrum = tf.py_function(func=interpolation, inp=[input_spectrum], Tout=tf.float32)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell_tf + 2.0)*(fullell_tf + 1.0)*fullell_tf*(fullell_tf - 1.0)*interp_spectrum/(fullell_tf + 0.5)**4
    spectrum_kitching = tf.repeat(tf.reshape(spectrum_kitching, [theta.shape[0], n_fullell, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell_tf[None,:,None] + 1.0)/(2*pi)/(fullell_tf[None,:,None]*(fullell_tf[None,:,None]+1))**2 * FT_plus_kernel[None,:,:] * spectrum_kitching
    CF = tf.reduce_sum(CF, axis=1)
    return CF

@tf.function
def kitching_correction_and_FT_minus_zeta(theta, spectrum, FT_minus_kernel, n_angular_bins):
     
    input_spectrum = tf.Variable(spectrum)
    interp_spectrum = tf.py_function(func=interpolation, inp=[input_spectrum], Tout=tf.float32)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell_tf + 2.0)*(fullell_tf + 1.0)*fullell_tf*(fullell_tf - 1.0)*interp_spectrum/(fullell_tf + 0.5)**4
    spectrum_kitching = tf.repeat(tf.reshape(spectrum_kitching, [theta.shape[0], n_fullell, -1]), n_angular_bins, axis=2)
    
    #Fourier transform
    CF = (2*fullell_tf[None,:,None] + 1.0)/(2*pi)/(fullell_tf[None,:,None]*(fullell_tf[None,:,None]+1))**2 * FT_minus_kernel[None,:,:] * spectrum_kitching
    CF = tf.reduce_sum(CF, axis=1)
    return CF

@tf.function
def kitching_correction_and_FT_plus_xi(theta, spectrum):
     
    input_spectrum = tf.Variable(spectrum)
    interp_spectrum = tf.py_function(func=interpolation, inp=[input_spectrum], Tout=tf.float32)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell_tf + 2.0)*(fullell_tf + 1.0)*fullell_tf*(fullell_tf - 1.0)*interp_spectrum/(fullell_tf + 0.5)**4
    spectrum_kitching = tf.repeat(tf.reshape(spectrum_kitching, [theta.shape[0], n_fullell, -1]), n_angular_bins_xi, axis=2)
    
    #Fourier transform
    CF = (2*fullell_tf[None,:,None] + 1.0)/(2*pi)/(fullell_tf[None,:,None]*(fullell_tf[None,:,None]+1))**2 * FT_plus_binave_xi[None,:,:] * spectrum_kitching
    CF = tf.reduce_sum(CF, axis=1)
    return CF

@tf.function
def kitching_correction_and_FT_minus_xi(theta, spectrum):
     
    input_spectrum = tf.Variable(spectrum)
    interp_spectrum = tf.py_function(func=interpolation, inp=[input_spectrum], Tout=tf.float32)
    #kitching correction: turn the power spectrum into the spherical sky under flat-sky approximation
    spectrum_kitching = (fullell_tf + 2.0)*(fullell_tf + 1.0)*fullell_tf*(fullell_tf - 1.0)*interp_spectrum/(fullell_tf + 0.5)**4
    spectrum_kitching = tf.repeat(tf.reshape(spectrum_kitching, [theta.shape[0], n_fullell, -1]), n_angular_bins_xi, axis=2)
    
    #Fourier transform
    CF = (2*fullell_tf[None,:,None] + 1.0)/(2*pi)/(fullell_tf[None,:,None]*(fullell_tf[None,:,None]+1))**2 * FT_minus_binave_xi[None,:,:] * spectrum_kitching
    CF = tf.reduce_sum(CF, axis=1)
    return CF

#### LOS projection functions ####
@tf.function
def los_proj_power_spectrum(theta, q1, q2, q3, q4, chi, H, P):
    z_los_tf = tf.repeat(tf.reshape(z_list_integ_tf, [1, -1, 1]), n_ell, axis=2)
    z_los_tf = tf.repeat(z_los_tf, theta.shape[0], axis=0)
    
    P_11_integ = (1/H[:,:,None]) * q1[:,:,None]**2/chi[:,:,None]**2 * P
    P_22_integ = (1/H[:,:,None]) * q2[:,:,None]**2/chi[:,:,None]**2 * P
    P_33_integ = (1/H[:,:,None]) * q3[:,:,None]**2/chi[:,:,None]**2 * P
    P_44_integ = (1/H[:,:,None]) * q4[:,:,None]**2/chi[:,:,None]**2 * P
    P_12_integ = (1/H[:,:,None]) * q1[:,:,None]*q2[:,:,None]/chi[:,:,None]**2 * P
    P_13_integ = (1/H[:,:,None]) * q1[:,:,None]*q3[:,:,None]/chi[:,:,None]**2 * P
    P_14_integ = (1/H[:,:,None]) * q1[:,:,None]*q4[:,:,None]/chi[:,:,None]**2 * P
    P_23_integ = (1/H[:,:,None]) * q2[:,:,None]*q3[:,:,None]/chi[:,:,None]**2 * P
    P_24_integ = (1/H[:,:,None]) * q2[:,:,None]*q4[:,:,None]/chi[:,:,None]**2 * P
    P_34_integ = (1/H[:,:,None]) * q3[:,:,None]*q4[:,:,None]/chi[:,:,None]**2 * P
    
    P_proj_11 = tfp.math.trapz(P_11_integ, z_los_tf, axis=1)
    P_proj_22 = tfp.math.trapz(P_22_integ, z_los_tf, axis=1)
    P_proj_33 = tfp.math.trapz(P_33_integ, z_los_tf, axis=1)
    P_proj_44 = tfp.math.trapz(P_44_integ, z_los_tf, axis=1)
    P_proj_12 = tfp.math.trapz(P_12_integ, z_los_tf, axis=1)
    P_proj_13 = tfp.math.trapz(P_13_integ, z_los_tf, axis=1)
    P_proj_14 = tfp.math.trapz(P_14_integ, z_los_tf, axis=1)
    P_proj_23 = tfp.math.trapz(P_23_integ, z_los_tf, axis=1)
    P_proj_24 = tfp.math.trapz(P_24_integ, z_los_tf, axis=1)
    P_proj_34 = tfp.math.trapz(P_34_integ, z_los_tf, axis=1)

    pixwin_func = tf.repeat(pixwin, theta.shape[0], axis=0)
    
    P_proj_11 *= pixwin_func**2 
    P_proj_22 *= pixwin_func**2
    P_proj_33 *= pixwin_func**2
    P_proj_44 *= pixwin_func**2
    P_proj_12 *= pixwin_func**2
    P_proj_13 *= pixwin_func**2
    P_proj_14 *= pixwin_func**2
    P_proj_23 *= pixwin_func**2
    P_proj_24 *= pixwin_func**2
    P_proj_34 *= pixwin_func**2
    
    return P_proj_11, P_proj_22, P_proj_33, P_proj_44, P_proj_12, P_proj_13, P_proj_14, P_proj_23, P_proj_24, P_proj_34

@tf.function
def los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp, iBamm):
    z_los_tf = tf.repeat(tf.reshape(z_list_integ_tf, [1,-1, 1]), n_ell, axis=2)
    z_los_tf = tf.repeat(z_los_tf, theta.shape[0], axis=0)
    
    iBapp_111_integ = (1/H[:,:,None]) * q1[:,:,None]**3/chi[:,:,None]**4 * iBapp
    iBapp_112_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q2[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_113_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_114_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_122_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_123_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q3[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_124_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_133_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_134_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_144_integ = (1/H[:,:,None]) * q1[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_222_integ = (1/H[:,:,None]) * q2[:,:,None]**3/chi[:,:,None]**4 * iBapp
    iBapp_223_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_224_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_233_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_234_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_244_integ = (1/H[:,:,None]) * q2[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_333_integ = (1/H[:,:,None]) * q3[:,:,None]**3/chi[:,:,None]**4 * iBapp
    iBapp_334_integ = (1/H[:,:,None]) * q3[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBapp
    iBapp_344_integ = (1/H[:,:,None]) * q3[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBapp
    iBapp_444_integ = (1/H[:,:,None]) * q4[:,:,None]**3/chi[:,:,None]**4 * iBapp
    
    
    iBamm_111_integ = (1/H[:,:,None]) * q1[:,:,None]**3/chi[:,:,None]**4 * iBamm
    iBamm_112_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q2[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_113_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_114_integ = (1/H[:,:,None]) * q1[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_122_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_123_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q3[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_124_integ = (1/H[:,:,None]) * q1[:,:,None] * q2[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_133_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_134_integ = (1/H[:,:,None]) * q1[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_144_integ = (1/H[:,:,None]) * q1[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_222_integ = (1/H[:,:,None]) * q2[:,:,None]**3/chi[:,:,None]**4 * iBamm
    iBamm_223_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q3[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_224_integ = (1/H[:,:,None]) * q2[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_233_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_234_integ = (1/H[:,:,None]) * q2[:,:,None] * q3[:,:,None] * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_244_integ = (1/H[:,:,None]) * q2[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_333_integ = (1/H[:,:,None]) * q3[:,:,None]**3/chi[:,:,None]**4 * iBamm
    iBamm_334_integ = (1/H[:,:,None]) * q3[:,:,None]**2 * q4[:,:,None]/chi[:,:,None]**4 * iBamm
    iBamm_344_integ = (1/H[:,:,None]) * q3[:,:,None] * q4[:,:,None]**2/chi[:,:,None]**4 * iBamm
    iBamm_444_integ = (1/H[:,:,None]) * q4[:,:,None]**3/chi[:,:,None]**4 * iBamm
    
    iBapp_proj_111 = tfp.math.trapz(iBapp_111_integ, z_los_tf, axis=1)
    iBapp_proj_112 = tfp.math.trapz(iBapp_112_integ, z_los_tf, axis=1)
    iBapp_proj_113 = tfp.math.trapz(iBapp_113_integ, z_los_tf, axis=1)
    iBapp_proj_114 = tfp.math.trapz(iBapp_114_integ, z_los_tf, axis=1)
    iBapp_proj_122 = tfp.math.trapz(iBapp_122_integ, z_los_tf, axis=1)
    iBapp_proj_123 = tfp.math.trapz(iBapp_123_integ, z_los_tf, axis=1)
    iBapp_proj_124 = tfp.math.trapz(iBapp_124_integ, z_los_tf, axis=1)
    iBapp_proj_133 = tfp.math.trapz(iBapp_133_integ, z_los_tf, axis=1)
    iBapp_proj_134 = tfp.math.trapz(iBapp_134_integ, z_los_tf, axis=1)
    iBapp_proj_144 = tfp.math.trapz(iBapp_144_integ, z_los_tf, axis=1)
    iBapp_proj_222 = tfp.math.trapz(iBapp_222_integ, z_los_tf, axis=1)
    iBapp_proj_223 = tfp.math.trapz(iBapp_223_integ, z_los_tf, axis=1)
    iBapp_proj_224 = tfp.math.trapz(iBapp_224_integ, z_los_tf, axis=1)
    iBapp_proj_233 = tfp.math.trapz(iBapp_233_integ, z_los_tf, axis=1)
    iBapp_proj_234 = tfp.math.trapz(iBapp_234_integ, z_los_tf, axis=1)
    iBapp_proj_244 = tfp.math.trapz(iBapp_244_integ, z_los_tf, axis=1)
    iBapp_proj_333 = tfp.math.trapz(iBapp_333_integ, z_los_tf, axis=1)
    iBapp_proj_334 = tfp.math.trapz(iBapp_334_integ, z_los_tf, axis=1)
    iBapp_proj_344 = tfp.math.trapz(iBapp_344_integ, z_los_tf, axis=1)
    iBapp_proj_444 = tfp.math.trapz(iBapp_444_integ, z_los_tf, axis=1)
    
    iBamm_proj_111 = tfp.math.trapz(iBamm_111_integ, z_los_tf, axis=1)
    iBamm_proj_112 = tfp.math.trapz(iBamm_112_integ, z_los_tf, axis=1)
    iBamm_proj_113 = tfp.math.trapz(iBamm_113_integ, z_los_tf, axis=1)
    iBamm_proj_114 = tfp.math.trapz(iBamm_114_integ, z_los_tf, axis=1)
    iBamm_proj_122 = tfp.math.trapz(iBamm_122_integ, z_los_tf, axis=1)
    iBamm_proj_123 = tfp.math.trapz(iBamm_123_integ, z_los_tf, axis=1)
    iBamm_proj_124 = tfp.math.trapz(iBamm_124_integ, z_los_tf, axis=1)
    iBamm_proj_133 = tfp.math.trapz(iBamm_133_integ, z_los_tf, axis=1)
    iBamm_proj_134 = tfp.math.trapz(iBamm_134_integ, z_los_tf, axis=1)
    iBamm_proj_144 = tfp.math.trapz(iBamm_144_integ, z_los_tf, axis=1)
    iBamm_proj_222 = tfp.math.trapz(iBamm_222_integ, z_los_tf, axis=1)
    iBamm_proj_223 = tfp.math.trapz(iBamm_223_integ, z_los_tf, axis=1)
    iBamm_proj_224 = tfp.math.trapz(iBamm_224_integ, z_los_tf, axis=1)
    iBamm_proj_233 = tfp.math.trapz(iBamm_233_integ, z_los_tf, axis=1)
    iBamm_proj_234 = tfp.math.trapz(iBamm_234_integ, z_los_tf, axis=1)
    iBamm_proj_244 = tfp.math.trapz(iBamm_244_integ, z_los_tf, axis=1)
    iBamm_proj_333 = tfp.math.trapz(iBamm_333_integ, z_los_tf, axis=1)
    iBamm_proj_334 = tfp.math.trapz(iBamm_334_integ, z_los_tf, axis=1)
    iBamm_proj_344 = tfp.math.trapz(iBamm_344_integ, z_los_tf, axis=1)
    iBamm_proj_444 = tfp.math.trapz(iBamm_444_integ, z_los_tf, axis=1)

    pixwin_func = tf.repeat(pixwin, theta.shape[0], axis=0)

    iBapp_proj_111 *= pixwin_func**2
    iBapp_proj_112 *= pixwin_func**2
    iBapp_proj_113 *= pixwin_func**2
    iBapp_proj_114 *= pixwin_func**2
    iBapp_proj_122 *= pixwin_func**2
    iBapp_proj_123 *= pixwin_func**2
    iBapp_proj_124 *= pixwin_func**2
    iBapp_proj_133 *= pixwin_func**2
    iBapp_proj_134 *= pixwin_func**2
    iBapp_proj_144 *= pixwin_func**2
    iBapp_proj_222 *= pixwin_func**2
    iBapp_proj_223 *= pixwin_func**2
    iBapp_proj_224 *= pixwin_func**2
    iBapp_proj_233 *= pixwin_func**2
    iBapp_proj_234 *= pixwin_func**2
    iBapp_proj_244 *= pixwin_func**2
    iBapp_proj_333 *= pixwin_func**2
    iBapp_proj_334 *= pixwin_func**2
    iBapp_proj_344 *= pixwin_func**2
    iBapp_proj_444 *= pixwin_func**2

    iBamm_proj_111 *= pixwin_func**2
    iBamm_proj_112 *= pixwin_func**2
    iBamm_proj_113 *= pixwin_func**2
    iBamm_proj_114 *= pixwin_func**2
    iBamm_proj_122 *= pixwin_func**2
    iBamm_proj_123 *= pixwin_func**2
    iBamm_proj_124 *= pixwin_func**2
    iBamm_proj_133 *= pixwin_func**2
    iBamm_proj_134 *= pixwin_func**2
    iBamm_proj_144 *= pixwin_func**2
    iBamm_proj_222 *= pixwin_func**2
    iBamm_proj_223 *= pixwin_func**2
    iBamm_proj_224 *= pixwin_func**2
    iBamm_proj_233 *= pixwin_func**2
    iBamm_proj_234 *= pixwin_func**2
    iBamm_proj_244 *= pixwin_func**2
    iBamm_proj_333 *= pixwin_func**2
    iBamm_proj_334 *= pixwin_func**2
    iBamm_proj_344 *= pixwin_func**2
    iBamm_proj_444 *= pixwin_func**2
    
    return iBapp_proj_111, iBapp_proj_112, iBapp_proj_113, iBapp_proj_114, iBapp_proj_122, iBapp_proj_123, iBapp_proj_124, iBapp_proj_133, iBapp_proj_134, iBapp_proj_144, iBapp_proj_222, iBapp_proj_223, iBapp_proj_224, iBapp_proj_233, iBapp_proj_234, iBapp_proj_244, iBapp_proj_333, iBapp_proj_334, iBapp_proj_344, iBapp_proj_444, iBamm_proj_111, iBamm_proj_112, iBamm_proj_113, iBamm_proj_114, iBamm_proj_122, iBamm_proj_123, iBamm_proj_124, iBamm_proj_133, iBamm_proj_134, iBamm_proj_144, iBamm_proj_222, iBamm_proj_223, iBamm_proj_224, iBamm_proj_233, iBamm_proj_234, iBamm_proj_244, iBamm_proj_333, iBamm_proj_334, iBamm_proj_344, iBamm_proj_444

#### Initialize the starting points of the MCMC chains ####
def start_position_lh_and_step_size(seed):
    random.seed(seed)
    p0 = []
    for i in range(nwalkers):
        Om_rand = random.uniform(Omega_min, Omega_max)
        As_rand = random.uniform(As_min, As_max)
        w0_rand = random.uniform(w0_min, w0_max)
        h_rand = random.uniform(h_min, h_max)
        cmin_rand = random.uniform(cmin_min, cmin_max)
        delta_z_bin1_rand = random.normalvariate(mu_delta_z1, std_delta_z1)
        delta_z_bin2_rand = random.normalvariate(mu_delta_z2, std_delta_z2)
        delta_z_bin3_rand = random.normalvariate(mu_delta_z3, std_delta_z3)
        delta_z_bin4_rand = random.normalvariate(mu_delta_z4, std_delta_z4)
        m1_rand = random.normalvariate(mu_m1, std_m1)
        m2_rand = random.normalvariate(mu_m2, std_m2)
        m3_rand = random.normalvariate(mu_m3, std_m3)
        m4_rand = random.normalvariate(mu_m4, std_m4)
        A_IA_NLA_rand = random.uniform(A_IA_NLA_min, A_IA_NLA_max)
    
        p0.append([Om_rand, As_rand, w0_rand, h_rand, cmin_rand, delta_z_bin1_rand, delta_z_bin2_rand, delta_z_bin3_rand, delta_z_bin4_rand, m1_rand, m2_rand, m3_rand, m4_rand, A_IA_NLA_rand])
    
    p0 = np.array(p0).astype(np.float32)
    return tf.convert_to_tensor(p0)

#### convert the sampled parameter tensor into an accessible table ####
def from_parameters_tensor_to_table(theta_tensor):
    
    training_parameters_names = ['Omega_m', 'As', 'w0', 'h', 'c_min', 'delta_z_bin1', 'delta_z_bin2', 'delta_z_bin3', 'delta_z_bin4', 'm_bin1', 'm_bin2', 'm_bin3', 'm_bin4', 'A_IA_NLA']
    
    parameters_values = tf.transpose(theta_tensor)
    parameters_table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string, 
                                                                value_dtype=tf.float32, 
                                                                empty_key="<EMPTY_SENTINEL>", 
                                                                deleted_key="<DELETE_SENTINEL>", 
                                                                #initial_num_buckets = 4,
                                                                experimental_is_anonymous = True,
                                                                default_value=tf.zeros([theta_tensor.shape[0]], dtype=tf.float32))
    parameters_table.insert(training_parameters_names, parameters_values)
    return parameters_table

#### Function computing power spectrum from CP_NN model ####
@tf.function
def get_P_from_NN(theta):
    
    # Use emulators to compute power spectrum corresponding to multiple cosmologies at a sequence of z along LOS
    theta = theta[:,:5]
    z_expansion = tf.reshape(tf.tile(z_list_integ_tf, [theta.shape[0]]), [-1,1])
    theta_expansion = tf.repeat(theta, tf.size(z_list_integ_tf), axis=0)
    theta_expansion = tf.concat([theta_expansion, z_expansion], axis=1)
    
    P = cp_nn_P.ten_to_predictions_tf(theta_expansion)
    
    return P

#### Function computing integrated bispectrum from CP_NN model ####
@tf.function
def get_iB_from_NN(theta, iBapp_emulator, iBamm_emulator, shift_constant):
    
    # Use emulators to compute integrated bispectrum corresponding to multiple cosmologies at a sequence of z along LOS
    theta = theta[:,:5]
    z_expansion = tf.reshape(tf.tile(z_list_integ_tf, [theta.shape[0]]), [-1,1])
    theta_expansion = tf.repeat(theta, tf.size(z_list_integ_tf), axis=0)
    theta_expansion = tf.concat([theta_expansion, z_expansion], axis=1)
    
    iBapp = iBapp_emulator.ten_to_predictions_tf(theta_expansion)
    iBamm = iBamm_emulator.ten_to_predictions_tf(theta_expansion) - shift_constant
    
    return iBapp, iBamm

#### Function computing H, chi and q from GP model ####
@tf.function
def get_H_chi_q_from_GP(theta):
    
    theta_H_chi_q = tf.concat([tf.reshape(theta[:,0],[-1,1]), tf.reshape(theta[:,2],[-1,1])], axis=1) # our emulators here only require Om and w0
    z_los_Wk_tf = tf.repeat(tf.reshape(z_list_integ_tf, [1,1,-1]), theta.shape[0], axis=0)
    z_los_Wk_tf = tf.repeat(z_los_Wk_tf, n_z_los, axis=1)
    z_expansion = tf.reshape(tf.tile(z_list_integ_tf, [theta.shape[0]]), [-1,1])
    theta_expansion = tf.repeat(theta_H_chi_q, tf.size(z_list_integ_tf), axis=0)
    theta_expansion = tf.concat([theta_expansion, z_expansion], axis=1)
    theta_expansion = tf.cast(theta_expansion, dtype=tf.float64)
    
    H = tf.cast(emulator_H.predict_f_compiled(theta_expansion)[0], dtype=tf.float32)
    H = tf.reshape(H * H_std + H_mean, [theta.shape[0], n_z_los])
    h_0 = tf.cast(tf.repeat(tf.reshape(theta[:,3], [-1,1]), n_z_los, axis=1), dtype=tf.float32)
    h_cosmogrid = 0.673 #the fiducial current expansion rate of Cosmogrid
    H = H*h_0/h_cosmogrid
    
    chi = tf.cast(emulator_chi.predict_f_compiled(theta_expansion)[0], dtype=tf.float32)
    chi = tf.reshape(chi * chi_std + chi_mean, [theta.shape[0], n_z_los])
    chi = chi*h_cosmogrid/h_0
    
    D = tf.cast(emulator_D.predict_f_compiled(theta_expansion)[0], dtype=tf.float32)
    D = tf.reshape(D * D_std + D_mean, [theta.shape[0], n_z_los])

    H_0 = theta[:,3]*100/speed_c 
    Omega_m0 = theta[:,0]
    A_IA_0_NLA = theta[:,13]
    #alpha_IA_0_NLA = theta[:,10]
    alpha_IA_0_NLA = tf.zeros(theta.shape[0])
    
    H_0 = tf.repeat(tf.reshape(H_0, [-1,1]), n_z_los, axis=1)
    Omega_m0 = tf.repeat(tf.reshape(Omega_m0, [-1,1]), n_z_los, axis=1)
    A_IA_0_NLA = tf.repeat(tf.reshape(A_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    alpha_IA_0_NLA = tf.repeat(tf.reshape(alpha_IA_0_NLA, [-1,1]), n_z_los, axis=1)
    
    # make two copies of chi as the chi_z and chi_zs components in the q kernel computation
    chi_z = tf.identity(chi)
    chi_zs = tf.identity(chi)
    
    chi_z = tf.reshape(chi_z, [theta.shape[0], n_z_los, -1])
    chi_zs = tf.reshape(chi_zs, [theta.shape[0], -1, n_z_los])
    chi_z = tf.repeat(chi_z, n_z_los, axis=2)
    chi_zs = tf.repeat(chi_zs, n_z_los, axis=1)
    
    theta_ns = tf.Variable(theta)
    n_zs_los_BIN1, n_zs_los_BIN2, n_zs_los_BIN3, n_zs_los_BIN4 = tf.py_function(func=n_s_z_compute, inp=[theta_ns], Tout=[tf.float32, tf.float32, tf.float32, tf.float32])
    n_zs_los_BIN1_prime = tf.repeat(tf.reshape(n_zs_los_BIN1, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN2_prime = tf.repeat(tf.reshape(n_zs_los_BIN2, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN3_prime = tf.repeat(tf.reshape(n_zs_los_BIN3, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    n_zs_los_BIN4_prime = tf.repeat(tf.reshape(n_zs_los_BIN4, [theta.shape[0], -1, n_z_los]), n_z_los, axis=1)
    
    indicator = tf.ones((n_z_los, n_z_los))
    indicator = tf.linalg.band_part(indicator, 0, -1)
    indicator = tf.repeat(tf.reshape(indicator, [-1, n_z_los, n_z_los]), theta.shape[0], axis=0)
    
    W1_integrand = n_zs_los_BIN1_prime * (chi_zs - chi_z)/chi_zs * indicator
    W2_integrand = n_zs_los_BIN2_prime * (chi_zs - chi_z)/chi_zs * indicator
    W3_integrand = n_zs_los_BIN3_prime * (chi_zs - chi_z)/chi_zs * indicator
    W4_integrand = n_zs_los_BIN4_prime * (chi_zs - chi_z)/chi_zs * indicator
    W1 = tfp.math.trapz(W1_integrand, z_los_Wk_tf, axis=2)
    W2 = tfp.math.trapz(W2_integrand, z_los_Wk_tf, axis=2)
    W3 = tfp.math.trapz(W3_integrand, z_los_Wk_tf, axis=2)
    W4 = tfp.math.trapz(W4_integrand, z_los_Wk_tf, axis=2)
    
    f_IA_NLA_z = -A_IA_0_NLA * 0.0134 * Omega_m0 * ((1.0 + z_list_integ_tf[None,:])/1.62)**alpha_IA_0_NLA/D
    
    q1 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ_tf) * chi * W1 + f_IA_NLA_z * H * n_zs_los_BIN1
    q2 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ_tf) * chi * W2 + f_IA_NLA_z * H * n_zs_los_BIN2
    q3 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ_tf) * chi * W3 + f_IA_NLA_z * H * n_zs_los_BIN3
    q4 = 3./2. * H_0**2 * Omega_m0 * (1 + z_list_integ_tf) * chi * W4 + f_IA_NLA_z * H * n_zs_los_BIN4
    
    return H, chi, q1, q2, q3, q4

#### function computing LOS projection of integrated bispectrum ####
@tf.function
def correlation_compute(theta):
    
    P = get_P_from_NN(theta)
    #iBapp_U50W50, iBamm_U50W50 = get_iB_from_NN(theta, cp_nn_iBapp_U50W50, cp_nn_iBamm_U50W50, iBamm_shift_constant_U50W50)
    #iBapp_U70W70, iBamm_U70W70 = get_iB_from_NN(theta, cp_nn_iBapp_U70W70, cp_nn_iBamm_U70W70, iBamm_shift_constant_U70W70)
    #iBapp_U90W90, iBamm_U90W90 = get_iB_from_NN(theta, cp_nn_iBapp_U90W90, cp_nn_iBamm_U90W90, iBamm_shift_constant_U90W90)
    #iBapp_U110W110, iBamm_U110W110 = get_iB_from_NN(theta, cp_nn_iBapp_U110W110, cp_nn_iBamm_U110W110, iBamm_shift_constant_U110W110)
    #iBapp_U130W130, iBamm_U130W130 = get_iB_from_NN(theta, cp_nn_iBapp_U130W130, cp_nn_iBamm_U130W130, iBamm_shift_constant_U130W130)

    H, chi, q1, q2, q3, q4 = get_H_chi_q_from_GP(theta)
    
    P = tf.reshape(P, [theta.shape[0], n_z_los, n_ell])
    #iBapp_U50W50 = tf.reshape(iBapp_U50W50, [theta.shape[0], n_z_los, n_ell])
    #iBapp_U70W70 = tf.reshape(iBapp_U70W70, [theta.shape[0], n_z_los, n_ell])
    #iBapp_U90W90 = tf.reshape(iBapp_U90W90, [theta.shape[0], n_z_los, n_ell])
    #iBapp_U110W110 = tf.reshape(iBapp_U110W110, [theta.shape[0], n_z_los, n_ell])
    #iBapp_U130W130 = tf.reshape(iBapp_U130W130, [theta.shape[0], n_z_los, n_ell])

    #iBamm_U50W50 = tf.reshape(iBamm_U50W50, [theta.shape[0], n_z_los, n_ell])
    #iBamm_U70W70 = tf.reshape(iBamm_U70W70, [theta.shape[0], n_z_los, n_ell])
    #iBamm_U90W90 = tf.reshape(iBamm_U90W90, [theta.shape[0], n_z_los, n_ell])
    #iBamm_U110W110 = tf.reshape(iBamm_U110W110, [theta.shape[0], n_z_los, n_ell])
    #iBamm_U130W130 = tf.reshape(iBamm_U130W130, [theta.shape[0], n_z_los, n_ell])

    P_proj_11, P_proj_22, P_proj_33, P_proj_44, P_proj_12, P_proj_13, P_proj_14, P_proj_23, P_proj_24, P_proj_34 = los_proj_power_spectrum(theta, q1, q2, q3, q4, chi, H, P)
    #iBapp_U50W50_proj_111, iBapp_U50W50_proj_112, iBapp_U50W50_proj_113, iBapp_U50W50_proj_114, iBapp_U50W50_proj_122, iBapp_U50W50_proj_123, iBapp_U50W50_proj_124, iBapp_U50W50_proj_133, iBapp_U50W50_proj_134, iBapp_U50W50_proj_144, iBapp_U50W50_proj_222, iBapp_U50W50_proj_223, iBapp_U50W50_proj_224, iBapp_U50W50_proj_233, iBapp_U50W50_proj_234, iBapp_U50W50_proj_244, iBapp_U50W50_proj_333, iBapp_U50W50_proj_334, iBapp_U50W50_proj_344, iBapp_U50W50_proj_444, iBamm_U50W50_proj_111, iBamm_U50W50_proj_112, iBamm_U50W50_proj_113, iBamm_U50W50_proj_114, iBamm_U50W50_proj_122, iBamm_U50W50_proj_123, iBamm_U50W50_proj_124, iBamm_U50W50_proj_133, iBamm_U50W50_proj_134, iBamm_U50W50_proj_144, iBamm_U50W50_proj_222, iBamm_U50W50_proj_223, iBamm_U50W50_proj_224, iBamm_U50W50_proj_233, iBamm_U50W50_proj_234, iBamm_U50W50_proj_244, iBamm_U50W50_proj_333, iBamm_U50W50_proj_334, iBamm_U50W50_proj_344, iBamm_U50W50_proj_444 = los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp_U50W50, iBamm_U50W50)
    #iBapp_U70W70_proj_111, iBapp_U70W70_proj_112, iBapp_U70W70_proj_113, iBapp_U70W70_proj_114, iBapp_U70W70_proj_122, iBapp_U70W70_proj_123, iBapp_U70W70_proj_124, iBapp_U70W70_proj_133, iBapp_U70W70_proj_134, iBapp_U70W70_proj_144, iBapp_U70W70_proj_222, iBapp_U70W70_proj_223, iBapp_U70W70_proj_224, iBapp_U70W70_proj_233, iBapp_U70W70_proj_234, iBapp_U70W70_proj_244, iBapp_U70W70_proj_333, iBapp_U70W70_proj_334, iBapp_U70W70_proj_344, iBapp_U70W70_proj_444, iBamm_U70W70_proj_111, iBamm_U70W70_proj_112, iBamm_U70W70_proj_113, iBamm_U70W70_proj_114, iBamm_U70W70_proj_122, iBamm_U70W70_proj_123, iBamm_U70W70_proj_124, iBamm_U70W70_proj_133, iBamm_U70W70_proj_134, iBamm_U70W70_proj_144, iBamm_U70W70_proj_222, iBamm_U70W70_proj_223, iBamm_U70W70_proj_224, iBamm_U70W70_proj_233, iBamm_U70W70_proj_234, iBamm_U70W70_proj_244, iBamm_U70W70_proj_333, iBamm_U70W70_proj_334, iBamm_U70W70_proj_344, iBamm_U70W70_proj_444 = los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp_U70W70, iBamm_U70W70)
    #iBapp_U90W90_proj_111, iBapp_U90W90_proj_112, iBapp_U90W90_proj_113, iBapp_U90W90_proj_114, iBapp_U90W90_proj_122, iBapp_U90W90_proj_123, iBapp_U90W90_proj_124, iBapp_U90W90_proj_133, iBapp_U90W90_proj_134, iBapp_U90W90_proj_144, iBapp_U90W90_proj_222, iBapp_U90W90_proj_223, iBapp_U90W90_proj_224, iBapp_U90W90_proj_233, iBapp_U90W90_proj_234, iBapp_U90W90_proj_244, iBapp_U90W90_proj_333, iBapp_U90W90_proj_334, iBapp_U90W90_proj_344, iBapp_U90W90_proj_444, iBamm_U90W90_proj_111, iBamm_U90W90_proj_112, iBamm_U90W90_proj_113, iBamm_U90W90_proj_114, iBamm_U90W90_proj_122, iBamm_U90W90_proj_123, iBamm_U90W90_proj_124, iBamm_U90W90_proj_133, iBamm_U90W90_proj_134, iBamm_U90W90_proj_144, iBamm_U90W90_proj_222, iBamm_U90W90_proj_223, iBamm_U90W90_proj_224, iBamm_U90W90_proj_233, iBamm_U90W90_proj_234, iBamm_U90W90_proj_244, iBamm_U90W90_proj_333, iBamm_U90W90_proj_334, iBamm_U90W90_proj_344, iBamm_U90W90_proj_444 = los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp_U90W90, iBamm_U90W90)
    #iBapp_U110W110_proj_111, iBapp_U110W110_proj_112, iBapp_U110W110_proj_113, iBapp_U110W110_proj_114, iBapp_U110W110_proj_122, iBapp_U110W110_proj_123, iBapp_U110W110_proj_124, iBapp_U110W110_proj_133, iBapp_U110W110_proj_134, iBapp_U110W110_proj_144, iBapp_U110W110_proj_222, iBapp_U110W110_proj_223, iBapp_U110W110_proj_224, iBapp_U110W110_proj_233, iBapp_U110W110_proj_234, iBapp_U110W110_proj_244, iBapp_U110W110_proj_333, iBapp_U110W110_proj_334, iBapp_U110W110_proj_344, iBapp_U110W110_proj_444, iBamm_U110W110_proj_111, iBamm_U110W110_proj_112, iBamm_U110W110_proj_113, iBamm_U110W110_proj_114, iBamm_U110W110_proj_122, iBamm_U110W110_proj_123, iBamm_U110W110_proj_124, iBamm_U110W110_proj_133, iBamm_U110W110_proj_134, iBamm_U110W110_proj_144, iBamm_U110W110_proj_222, iBamm_U110W110_proj_223, iBamm_U110W110_proj_224, iBamm_U110W110_proj_233, iBamm_U110W110_proj_234, iBamm_U110W110_proj_244, iBamm_U110W110_proj_333, iBamm_U110W110_proj_334, iBamm_U110W110_proj_344, iBamm_U110W110_proj_444 = los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp_U110W110, iBamm_U110W110)
    #iBapp_U130W130_proj_111, iBapp_U130W130_proj_112, iBapp_U130W130_proj_113, iBapp_U130W130_proj_114, iBapp_U130W130_proj_122, iBapp_U130W130_proj_123, iBapp_U130W130_proj_124, iBapp_U130W130_proj_133, iBapp_U130W130_proj_134, iBapp_U130W130_proj_144, iBapp_U130W130_proj_222, iBapp_U130W130_proj_223, iBapp_U130W130_proj_224, iBapp_U130W130_proj_233, iBapp_U130W130_proj_234, iBapp_U130W130_proj_244, iBapp_U130W130_proj_333, iBapp_U130W130_proj_334, iBapp_U130W130_proj_344, iBapp_U130W130_proj_444, iBamm_U130W130_proj_111, iBamm_U130W130_proj_112, iBamm_U130W130_proj_113, iBamm_U130W130_proj_114, iBamm_U130W130_proj_122, iBamm_U130W130_proj_123, iBamm_U130W130_proj_124, iBamm_U130W130_proj_133, iBamm_U130W130_proj_134, iBamm_U130W130_proj_144, iBamm_U130W130_proj_222, iBamm_U130W130_proj_223, iBamm_U130W130_proj_224, iBamm_U130W130_proj_233, iBamm_U130W130_proj_234, iBamm_U130W130_proj_244, iBamm_U130W130_proj_333, iBamm_U130W130_proj_334, iBamm_U130W130_proj_344, iBamm_U130W130_proj_444 = los_proj_bispectrum(theta, q1, q2, q3, q4, chi, H, iBapp_U130W130, iBamm_U130W130)

    m_1 = theta[:,9]
    m_2 = theta[:,10]
    m_3 = theta[:,11]
    m_4 = theta[:,12]
    
    xip_11 = kitching_correction_and_FT_plus_xi(theta, P_proj_11)*(1.0+m_1[:,None])*(1.0+m_1[:,None])
    xip_22 = kitching_correction_and_FT_plus_xi(theta, P_proj_22)*(1.0+m_2[:,None])*(1.0+m_2[:,None])
    xip_33 = kitching_correction_and_FT_plus_xi(theta, P_proj_33)*(1.0+m_3[:,None])*(1.0+m_3[:,None])
    xip_44 = kitching_correction_and_FT_plus_xi(theta, P_proj_44)*(1.0+m_4[:,None])*(1.0+m_4[:,None])
    xip_12 = kitching_correction_and_FT_plus_xi(theta, P_proj_12)*(1.0+m_1[:,None])*(1.0+m_2[:,None])
    xip_13 = kitching_correction_and_FT_plus_xi(theta, P_proj_13)*(1.0+m_1[:,None])*(1.0+m_3[:,None])
    xip_14 = kitching_correction_and_FT_plus_xi(theta, P_proj_14)*(1.0+m_1[:,None])*(1.0+m_4[:,None])
    xip_23 = kitching_correction_and_FT_plus_xi(theta, P_proj_23)*(1.0+m_2[:,None])*(1.0+m_3[:,None])
    xip_24 = kitching_correction_and_FT_plus_xi(theta, P_proj_24)*(1.0+m_2[:,None])*(1.0+m_4[:,None])
    xip_34 = kitching_correction_and_FT_plus_xi(theta, P_proj_34)*(1.0+m_3[:,None])*(1.0+m_4[:,None])
    
    xim_11 = kitching_correction_and_FT_minus_xi(theta, P_proj_11)*(1.0+m_1[:,None])*(1.0+m_1[:,None])
    xim_22 = kitching_correction_and_FT_minus_xi(theta, P_proj_22)*(1.0+m_2[:,None])*(1.0+m_2[:,None])
    xim_33 = kitching_correction_and_FT_minus_xi(theta, P_proj_33)*(1.0+m_3[:,None])*(1.0+m_3[:,None])
    xim_44 = kitching_correction_and_FT_minus_xi(theta, P_proj_44)*(1.0+m_4[:,None])*(1.0+m_4[:,None])
    xim_12 = kitching_correction_and_FT_minus_xi(theta, P_proj_12)*(1.0+m_1[:,None])*(1.0+m_2[:,None])
    xim_13 = kitching_correction_and_FT_minus_xi(theta, P_proj_13)*(1.0+m_1[:,None])*(1.0+m_3[:,None])
    xim_14 = kitching_correction_and_FT_minus_xi(theta, P_proj_14)*(1.0+m_1[:,None])*(1.0+m_4[:,None])
    xim_23 = kitching_correction_and_FT_minus_xi(theta, P_proj_23)*(1.0+m_2[:,None])*(1.0+m_3[:,None])
    xim_24 = kitching_correction_and_FT_minus_xi(theta, P_proj_24)*(1.0+m_2[:,None])*(1.0+m_4[:,None])
    xim_34 = kitching_correction_and_FT_minus_xi(theta, P_proj_34)*(1.0+m_3[:,None])*(1.0+m_4[:,None])

    #iZp_U50W50_111 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_111, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_112 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_112, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_113 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_113, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_114 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_114, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_122 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_122, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_123 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_123, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_124 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_124, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_133 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_133, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_134 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_134, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_144 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_144, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_222 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_222, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_223 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_223, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_224 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_224, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_233 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_233, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_234 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_234, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_244 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_244, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_333 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_333, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_334 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_334, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_344 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_344, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZp_U50W50_444 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U50W50_proj_444, FT_plus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_111 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_111, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_112 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_112, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_113 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_113, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_114 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_114, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_122 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_122, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_123 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_123, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_124 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_124, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_133 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_133, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_134 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_134, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_144 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_144, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_222 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_222, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_223 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_223, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_224 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_224, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_233 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_233, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_234 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_234, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_244 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_244, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_333 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_333, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_334 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_334, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_344 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_344, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]
    #iZm_U50W50_444 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U50W50_proj_444, FT_minus_binave_zeta_U50W50, n_angular_bins_iZ50)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U50W50[None,:]

    #iZp_U70W70_111 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_111, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_112 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_112, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_113 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_113, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_114 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_114, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_122 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_122, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_123 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_123, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_124 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_124, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_133 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_133, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_134 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_134, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_144 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_144, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_222 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_222, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_223 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_223, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_224 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_224, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_233 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_233, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_234 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_234, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_244 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_244, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_333 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_333, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_334 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_334, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_344 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_344, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZp_U70W70_444 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U70W70_proj_444, FT_plus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_111 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_111, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_112 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_112, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_113 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_113, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_114 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_114, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_122 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_122, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_123 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_123, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_124 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_124, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_133 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_133, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_134 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_134, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_144 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_144, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_222 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_222, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_223 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_223, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_224 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_224, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_233 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_233, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_234 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_234, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_244 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_244, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_333 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_333, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_334 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_334, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_344 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_344, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    #iZm_U70W70_444 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U70W70_proj_444, FT_minus_binave_zeta_U70W70, n_angular_bins_iZ70)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U70W70[None,:]
    
    #iZp_U90W90_111 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_111, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_112 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_112, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_113 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_113, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_114 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_114, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_122 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_122, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_123 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_123, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_124 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_124, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_133 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_133, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_134 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_134, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_144 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_144, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_222 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_222, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_223 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_223, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_224 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_224, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_233 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_233, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_234 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_234, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_244 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_244, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_333 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_333, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_334 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_334, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_344 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_344, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZp_U90W90_444 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U90W90_proj_444, FT_plus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_111 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_111, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_112 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_112, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_113 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_113, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_114 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_114, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_122 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_122, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_123 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_123, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_124 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_124, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_133 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_133, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_134 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_134, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_144 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_144, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_222 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_222, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_223 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_223, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_224 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_224, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_233 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_233, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_234 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_234, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_244 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_244, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_333 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_333, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_334 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_334, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_344 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_344, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]
    #iZm_U90W90_444 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U90W90_proj_444, FT_minus_binave_zeta_U90W90, n_angular_bins_iZ90)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U90W90[None,:]

    #iZp_U110W110_111 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_111, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_112 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_112, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_113 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_113, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_114 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_114, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_122 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_122, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_123 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_123, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_124 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_124, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_133 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_133, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_134 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_134, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_144 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_144, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_222 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_222, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_223 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_223, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_224 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_224, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_233 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_233, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_234 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_234, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_244 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_244, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_333 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_333, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_334 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_334, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_344 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_344, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZp_U110W110_444 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U110W110_proj_444, FT_plus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_111 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_111, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_112 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_112, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_113 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_113, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_114 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_114, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_122 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_122, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_123 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_123, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_124 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_124, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_133 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_133, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_134 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_134, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_144 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_144, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_222 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_222, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_223 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_223, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_224 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_224, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_233 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_233, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_234 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_234, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_244 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_244, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_333 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_333, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_334 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_334, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_344 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_344, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]
    #iZm_U110W110_444 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U110W110_proj_444, FT_minus_binave_zeta_U110W110, n_angular_bins_iZ110)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U110W110[None,:]

   #iZp_U130W130_111 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_111, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_112 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_112, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_113 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_113, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_114 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_114, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_122 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_122, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_123 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_123, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_124 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_124, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_133 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_133, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_134 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_134, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_144 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_144, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_222 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_222, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_223 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_223, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_224 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_224, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_233 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_233, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_234 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_234, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_244 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_244, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_333 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_333, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_334 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_334, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_344 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_344, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZp_U130W130_444 = kitching_correction_and_FT_plus_zeta(theta, iBapp_U130W130_proj_444, FT_plus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZm_U130W130_111 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_111, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_1[:,None])/A_2pt_U130W130[None,:]
   #iZm_U130W130_112 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_112, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_2[:,None])/A_2pt_U130W130[None,:]
   #iZm_U130W130_113 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_113, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
   #iZm_U130W130_114 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_114, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_1[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
   #iZm_U130W130_122 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_122, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_123 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_123, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_124 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_124, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_133 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_133, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_134 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_134, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_144 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_144, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_1[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_222 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_222, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_2[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_223 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_223, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_224 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_224, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_2[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_233 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_233, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_234 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_234, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_244 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_244, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_2[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_333 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_333, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_3[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_334 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_334, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_3[:,None])*(1.0+m_3[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_344 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_344, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_3[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    #iZm_U130W130_444 = kitching_correction_and_FT_minus_zeta(theta, iBamm_U130W130_proj_444, FT_minus_binave_zeta_U130W130, n_angular_bins_iZ130)*(1.0+m_4[:,None])*(1.0+m_4[:,None])*(1.0+m_4[:,None])/A_2pt_U130W130[None,:]
    
    #data_vector = tf.concat([xip_11, xip_12, xip_13, xip_14, xip_22, xip_23, xip_24, xip_33, xip_34, xip_44, xim_11, xim_12, xim_13, xim_14, xim_22, xim_23, xim_24, xim_33, xim_34, xim_44, iZp_U50W50_111, iZp_U50W50_112, iZp_U50W50_113, iZp_U50W50_114, iZp_U50W50_122, iZp_U50W50_123, iZp_U50W50_124, iZp_U50W50_133, iZp_U50W50_134, iZp_U50W50_144, iZp_U50W50_222, iZp_U50W50_223, iZp_U50W50_224, iZp_U50W50_233, iZp_U50W50_234, iZp_U50W50_244, iZp_U50W50_333, iZp_U50W50_334, iZp_U50W50_344, iZp_U50W50_444, iZm_U50W50_111, iZm_U50W50_112, iZm_U50W50_113, iZm_U50W50_114, iZm_U50W50_122, iZm_U50W50_123, iZm_U50W50_124, iZm_U50W50_133, iZm_U50W50_134, iZm_U50W50_144, iZm_U50W50_222, iZm_U50W50_223, iZm_U50W50_224, iZm_U50W50_233, iZm_U50W50_234, iZm_U50W50_244, iZm_U50W50_333, iZm_U50W50_334, iZm_U50W50_344, iZm_U50W50_444, iZp_U70W70_111, iZp_U70W70_112, iZp_U70W70_113, iZp_U70W70_114, iZp_U70W70_122, iZp_U70W70_123, iZp_U70W70_124, iZp_U70W70_133, iZp_U70W70_134, iZp_U70W70_144, iZp_U70W70_222, iZp_U70W70_223, iZp_U70W70_224, iZp_U70W70_233, iZp_U70W70_234, iZp_U70W70_244, iZp_U70W70_333, iZp_U70W70_334, iZp_U70W70_344, iZp_U70W70_444, iZm_U70W70_111, iZm_U70W70_112, iZm_U70W70_113, iZm_U70W70_114, iZm_U70W70_122, iZm_U70W70_123, iZm_U70W70_124, iZm_U70W70_133, iZm_U70W70_134, iZm_U70W70_144, iZm_U70W70_222, iZm_U70W70_223, iZm_U70W70_224, iZm_U70W70_233, iZm_U70W70_234, iZm_U70W70_244, iZm_U70W70_333, iZm_U70W70_334, iZm_U70W70_344, iZm_U70W70_444, iZp_U90W90_111, iZp_U90W90_112, iZp_U90W90_113, iZp_U90W90_114, iZp_U90W90_122, iZp_U90W90_123, iZp_U90W90_124, iZp_U90W90_133, iZp_U90W90_134, iZp_U90W90_144, iZp_U90W90_222, iZp_U90W90_223, iZp_U90W90_224, iZp_U90W90_233, iZp_U90W90_234, iZp_U90W90_244, iZp_U90W90_333, iZp_U90W90_334, iZp_U90W90_344, iZp_U90W90_444, iZm_U90W90_111, iZm_U90W90_112, iZm_U90W90_113, iZm_U90W90_114, iZm_U90W90_122, iZm_U90W90_123, iZm_U90W90_124, iZm_U90W90_133, iZm_U90W90_134, iZm_U90W90_144, iZm_U90W90_222, iZm_U90W90_223, iZm_U90W90_224, iZm_U90W90_233, iZm_U90W90_234, iZm_U90W90_244, iZm_U90W90_333, iZm_U90W90_334, iZm_U90W90_344, iZm_U90W90_444, iZp_U110W110_111, iZp_U110W110_112, iZp_U110W110_113, iZp_U110W110_114, iZp_U110W110_122, iZp_U110W110_123, iZp_U110W110_124, iZp_U110W110_133, iZp_U110W110_134, iZp_U110W110_144, iZp_U110W110_222, iZp_U110W110_223, iZp_U110W110_224, iZp_U110W110_233, iZp_U110W110_234, iZp_U110W110_244, iZp_U110W110_333, iZp_U110W110_334, iZp_U110W110_344, iZp_U110W110_444, iZm_U110W110_111, iZm_U110W110_112, iZm_U110W110_113, iZm_U110W110_114, iZm_U110W110_122, iZm_U110W110_123, iZm_U110W110_124, iZm_U110W110_133, iZm_U110W110_134, iZm_U110W110_144, iZm_U110W110_222, iZm_U110W110_223, iZm_U110W110_224, iZm_U110W110_233, iZm_U110W110_234, iZm_U110W110_244, iZm_U110W110_333, iZm_U110W110_334, iZm_U110W110_344, iZm_U110W110_444, iZp_U130W130_111, iZp_U130W130_112, iZp_U130W130_113, iZp_U130W130_114, iZp_U130W130_122, iZp_U130W130_123, iZp_U130W130_124, iZp_U130W130_133, iZp_U130W130_134, iZp_U130W130_144, iZp_U130W130_222, iZp_U130W130_223, iZp_U130W130_224, iZp_U130W130_233, iZp_U130W130_234, iZp_U130W130_244, iZp_U130W130_333, iZp_U130W130_334, iZp_U130W130_344, iZp_U130W130_444, iZm_U130W130_111, iZm_U130W130_112, iZm_U130W130_113, iZm_U130W130_114, iZm_U130W130_122, iZm_U130W130_123, iZm_U130W130_124, iZm_U130W130_133, iZm_U130W130_134, iZm_U130W130_144, iZm_U130W130_222, iZm_U130W130_223, iZm_U130W130_224, iZm_U130W130_233, iZm_U130W130_234, iZm_U130W130_244, iZm_U130W130_333, iZm_U130W130_334, iZm_U130W130_344, iZm_U130W130_444], axis=1)
    data_vector = tf.concat([xip_11, xip_12, xip_13, xip_14, xip_22, xip_23, xip_24, xip_33, xip_34, xip_44, xim_11, xim_12, xim_13, xim_14, xim_22, xim_23, xim_24, xim_33, xim_34, xim_44], axis=1)
    
    return data_vector

#### The functional form of prior ####
@tf.function
def log_gaussian_prior(theta):
    delta_z1 = theta[:,5]
    delta_z2 = theta[:,6]
    delta_z3 = theta[:,7]
    delta_z4 = theta[:,8]
    m1 = theta[:,9]
    m2 = theta[:,10]
    m3 = theta[:,11]
    m4 = theta[:,12]
    log_gaussian = -0.5 * (((delta_z1 - mu_delta_z1_tf)/std_delta_z1_tf)**2 + ((delta_z2 - mu_delta_z2_tf)/std_delta_z2_tf)**2 + ((delta_z3 - mu_delta_z3_tf)/std_delta_z3_tf)**2 + ((delta_z4 - mu_delta_z4_tf)/std_delta_z4_tf)**2 + ((m1 - mu_m1_tf)/std_m1_tf)**2 + ((m2 - mu_m2_tf)/std_m2_tf)**2 + ((m3 - mu_m3_tf)/std_m3_tf)**2 + ((m4 - mu_m4_tf)/std_m4_tf)**2)
    return log_gaussian

#### The functional form of prior ####
@tf.function
def log_prior(theta, theta_table):

    lnprior = log_gaussian_prior(theta)
    #lnprior = tf.zeros(size)
    
    #cosmological priors
    lnprior=tf.where(theta_table.lookup('Omega_m') < Omega_min, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('Omega_m') > Omega_max, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('As') < As_min, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('As') > As_max, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('w0') < w0_min, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('w0') > w0_max, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('h') < h_min, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('h') > h_max, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('c_min') < cmin_min, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('c_min') > cmin_max, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('A_IA_NLA') < A_IA_NLA_min, -np.inf, lnprior)
    lnprior=tf.where(theta_table.lookup('A_IA_NLA') > A_IA_NLA_max, -np.inf, lnprior)
    #lnprior=tf.where(theta_table.lookup('alpha_IA_NLA') < alpha_IA_NLA_min, -np.inf, lnprior)
    #lnprior=tf.where(theta_table.lookup('alpha_IA_NLA') > alpha_IA_NLA_max, -np.inf, lnprior)
    
    return tf.cast(lnprior, dtype=tf.float32)

#### The functional form of likelihood ####
@tf.function
def log_likelihood(theta, data, inv_cov):
    
    # import pdb
    # pdb.set_trace()
    # Compute the model vector in MCMC
    model = correlation_compute(theta)
    
    diff = data - model
    chi2 = tf.matmul(hartlap_factor * inv_cov, tf.transpose(diff))
    chi2 = tf.matmul(diff, chi2)
    chi2 = tf.linalg.diag_part(chi2)
    # Here we use Gaussian likelihood and emcee only takes in log-likelihood
    gaussian_likelihood = -0.5 * chi2
    
    # Here we use the Percival likelihood which takes care of both Dodelson-Schneider and Percival effect in the parameter covariance contours. See https://arxiv.org/pdf/2108.10402.pdf
    #percival_likelihood = -0.5 * m * tf.math.log(1 + chi2/(N_s_tf - 1))
    
    return gaussian_likelihood

#### The functional form of posterior probability
#@tf.function
def log_posterior(theta, data, inv_cov):
    
    theta_table = from_parameters_tensor_to_table(theta)
    lp = log_prior(theta, theta_table)
    posterior = -np.inf * tf.ones(theta.shape[0])
    
    mask = tf.math.is_finite(lp)
    indices = tf.where(mask)
    likelihood = log_likelihood(theta[mask], data, inv_cov)
    
    posterior = tf.tensor_scatter_nd_update(posterior, indices, tf.add(likelihood, lp[mask]))
    return posterior

p0 = start_position_lh_and_step_size(0)
p1 = start_position_lh_and_step_size(1)

current_state = [p0, p1]

start2 = time.time()

# run the sampler
with tf.device(device):
    #chain, logp_chain = affine.affine_sample(log_posterior, total_steps, current_state, args=[])
    chain = affine_sample(log_posterior, total_steps, current_state, args=[data_vec, inv_covariance])
    
# how many burnin steps to remove
samples = chain.numpy()[burnin_steps:,:,:].reshape((-1, p0.shape[1]))
#logp_samples = logp_chain.numpy()[burnin_steps:,:].reshape((-1, 1)).T
print(samples.shape)

np.save('MCMC_2PCF_cosmogrid_'+str(ndim)+'dim_model_1e3steps_alphaIA_0_w_nside512_updated_H_chi_D',samples)
#np.save('logp_zeta_plus_222_'+str(ndim)+'dim_model',logp_samples)

end = time.time()
multi_time2 = end - start2
multi_time = end - start
print("MCMC of 500 walkers and 5000 steps took {0:.1f} seconds".format(multi_time2))
print("The execution of the whole script took {0:.1f} seconds".format(multi_time))
