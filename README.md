# Emulated_inference_shear_i3PCF
Codes that used in producing all results in the paper of emulation based inference of shear integrated 3PCF which is based on the works: https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2780H/abstract and https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.4639H/abstract

The code includes three major parts:
1. The emulation of shear i3PCF and other cosmological background quantities using NN and GP
2. The measurement of data vectors from simulated DES (Dark Energy Survey) Year 3-like footprints with realistic source redshift distribution n(z), masks and galaxy shape noise
3. The likelihood analyses of shear i3PCF and 2PCF using MCMC to investigate the parameter constraints improvement and optimize the filter size parameter within the shear i3PCF model

# Notice
1. Both emulation and MCMC are computed on GPU to save time expense
2. The two emulation packages we used are Cosmopower (https://alessiospuriomancini.github.io/cosmopower/) and GPflow (https://github.com/GPflow/GPflow)
3. The simulated shear data used in this case comes from the full-sky Takahashi simulation (T17): https://arxiv.org/abs/1706.01472
4. The posterior sampler in MCMC used in this case is from https://github.com/justinalsing/affine
5. Any questions regarding the codes in this repository, please contact the owner via lgong@usm.lmu
