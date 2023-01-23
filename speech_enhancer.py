#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
License agreement in LICENSE.txt
"""

import torch
import numpy as np
import soundfile as sf
import librosa
import torch.nn as nn

from SE_utils import SpeechEnhancement

from AV_VAE import myVAE_MIN_VAE, myVAE_MIN_VAEDec
import os


#%% network parameters

input_dim = 513
latent_dim = 32
device = 'cpu' # 'cuda'
hidden_dim_encoder = [128]
activation = torch.tanh
activationv = nn.ReLU()
landmarks_dim = 67*67 # if you use raw video data, this dimension should be 67*67. Otherwise, if you use the
#                       pre-trained ASR feature extractor, this dimension is 1280

#%% MCEM algorithm parameters

niter_MCEM = 100 # number of iterations for the MCEM algorithm
niter_MH = 40 # total number of samples for the Metropolis-Hastings algorithm
burnin = 30 # number of initial samples to be discarded
var_MH = 0.01 # variance of the proposal distribution
tol = 1e-5 # tolerance for stopping the MCEM iterations

#%% STFT parameters

wlen_sec = 64e-3
hop_percent = 0.521


fs = 16000

save_dir = './results'  # directory to save results
saved_models = './models' # where the VAE model is
mix_file = './data/mix.wav'  # input noisy speech
video_file = './data/video.npy' # input video data

K_b = 10 # NMF rank for noise model

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

#%% Here, we test the performance of the MIN-VAE model

saved_model_min_vae = os.path.join(saved_models, 'MIN_VAE_checkpoint.pt')

# Loading the pre-trained model:
vae = myVAE_MIN_VAE(input_dim=input_dim, latent_dim=latent_dim,
        hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size,
        activation=activation, activationv=activationv).to(device)

checkpoint = torch.load(saved_model_min_vae, map_location = 'cpu')
vae.load_state_dict(checkpoint['model_state_dict'], strict = False)
decoder = myDecoder(vae)

# As we do not train the models, we set them to the "eval" mode:
vae.eval()
decoder.eval()


# we will not update the network parameters
for param in decoder.parameters():
    param.requires_grad = False

# Instanciate the SE algo
se_algo = SpeechEnhancement(vae0 = vae, mix_file = mix_file, video_file = video_file, K_b = K_b,
                                fs = fs, hop_percent = hop_percent, wlen_sec = wlen_sec, niter_alg = niter_MCEM, pi = 0.5, verbose = False)

# Run the SE algo
s_hat, b_hat = se_algo.run()


# save the results:
save_vae = os.path.join(save_dir, 'MIN-VAE')
if not os.path.isdir(save_vae):
    os.makedirs(save_vae)

sf.write(os.path.join(save_vae,'est_speech.wav'), s_hat, fs)
sf.write(os.path.join(save_vae,'est_noise.wav'), b_hat, fs)
