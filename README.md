# Emulated inference shear integrated 3-point correlation function
Codes that used in producing all results in the paper of emulation based inference of shear integrated 3-point correlation function (i3PCF) which is based on the works: https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2780H/abstract and https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.4639H/abstract

The code includes two major parts:
1. Data preprocessing and the emulation of shear i3PCF and other cosmological background quantities using NN and GP
2. The likelihood analyses of shear i3PCF and 2PCF using MCMC to investigate the parameter constraints improvement and optimize the filter size parameter within the shear i3PCF model

As for the measurement of data vectors from simulated DES (Dark Energy Survey) Year 3-like footprints with realistic source redshift distribution n(z), masks and galaxy shape noise, please refer to the relevant codes in https://github.com/D-Gebauer/CosmoFuse. This package can also be used to estimate data covariance in the statistical analysis. For the code that computes the integrated bispectrum which is used as the training and testing dataset here, please refer to https://github.com/anikhalder/i3PCF/tree/main

# Notice
1. Both emulation and MCMC are computed on GPU to save time expense
2. The two emulation packages we used are Cosmopower (https://alessiospuriomancini.github.io/cosmopower/) and GPflow (https://github.com/GPflow/GPflow)
3. The simulated shear data used in this case comes from the full-sky Takahashi simulation (T17): https://arxiv.org/abs/1706.01472
4. The posterior sampler in MCMC used in this case is from https://github.com/justinalsing/affine
5. Any questions regarding the codes in this repository, please contact the owner via lgong@usm.lmu
