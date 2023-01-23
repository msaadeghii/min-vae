#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:19:47 2019

@author: Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
"""

import torch
import numpy as np
import soundfile as sf
import librosa
import sys

from torch import optim

from scipy import stats
import os

class SpeechEnhancement:
    def __init__(self, vae0, mix_file, video_file, K_b, fs, hop_percent, wlen_sec
                 , niter_alg, pi = 0.5, verbose = False):

        self.verbose = verbose
        self.device = 'cpu' #'cpu' cuda
        self.fs= fs
        self.index = 0
        self.pi = pi

        self.vae0 = vae0

        self.vae0.eval()

        # load signals
        self.x, _ = librosa.load(mix_file, sr=None)

        self.wlen = int(wlen_sec*self.fs) # window length of 64 ms
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # next power of 2
        self.nfft = self.wlen
        self.hop = np.int(hop_percent*self.wlen) # hop size
        self.win = np.sin(np.arange(.5,self.wlen-.5+1)/self.wlen*np.pi); # sine analysis window

        self.T_orig = len(self.x)
        self.K_b = K_b
        self.niter_alg = niter_alg


        # normalize input mixture
        self.x = self.x/np.max(np.abs(self.x))

        self.X = librosa.stft(self.x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)


        # Observed power spectrogram of the noisy mixture signal
        self.X_abs_2 = np.abs(self.X)**2
        self.X_abs_2_tensor = torch.from_numpy(self.X_abs_2.astype(np.float32))

        F, N = self.X.shape
        self.N_freq, self.N_frames = F, N

        self.v0 = np.load(video_file).T # for AV

        Nl = np.maximum(N, self.v0.shape[1])

        if self.v0.shape[1] < Nl:
            self.v0 = np.hstack((self.v0, np.tile(self.v0[:, [-1]], Nl-self.v0.shape[1])))

        self.v0 = torch.from_numpy(self.v0.astype(np.float32))

        self.v0.requires_grad = False

        # Random initialization of NMF parameters
        self.eps = np.finfo(float).eps
        np.random.seed(0)
        self.W0 = np.maximum(np.random.rand(F,self.K_b), self.eps)
        self.H0 = np.maximum(np.random.rand(self.K_b,N), self.eps)
        self.XS = self.X

        self.V_b0 = self.W0@self.H0
        self.V_b_tensor0 = torch.from_numpy(self.V_b0.astype(np.float32))

        # All-ones initialization of the gain parameters
        self.g0 = np.ones((1,N))
        self.g_tensor0 = torch.from_numpy(self.g0.astype(np.float32))

        # MCMC Algorithm parameters
        self.niter_MCEM = self.niter_alg
        self.niter_MH = 40
        self.burnin = 30
        self.var_MH = 0.01
        self.tol = 1e-5

        # Initialize the latent variables by encoding the noisy mixture
        with torch.no_grad():
            data_orig = np.abs(self.X)**2
            data = data_orig.T
            data = torch.from_numpy(data.astype(np.float32))
            data = data.to(self.device)

            recon_batchA,mu_a,logvar_a,recon_batchV,mu_v,logvar_v, muz0, logvar0, muz1, logvar1 = self.vae0(data, self.v1)

            self.sig_zp0 = logvar0.detach().numpy() #np.asarray([1.0]) #logvar0.detach().numpy() #torch.exp(logvar0).detach().numpy() #np.ones((1)) #torch.exp(logvar0).detach().numpy()
            self.sig_zp1 = logvar1.detach().numpy() #np.asarray([1.0]) #logvar1.detach().numpy() #torch.exp(logvar1).detach().numpy() #np.ones((1)) #torch.exp(logvar1).detach().numpy()
            self.mu_zp0 = muz0.detach().numpy() #np.zeros((1,32)) #muz0.detach().numpy()
            self.mu_zp1 = muz1.detach().numpy() #np.zeros((1,32)) #muz1.detach().numpy()
            z1, _ = self.vae1.encode(data, self.v1)
            z0 = mu_v

            self.z0_init = torch.t(z0)


    def num2torch(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y

    def torch2num(self, x):
        y = x.detach().numpy()
        return y


    def metropolis_hastings_MIN_VAE(self, decoderA, pi_vals, abs_Sp2, muz = None, varz = None, niter_MH=40, burnin=30, var_MH = 0.01):

        N = self.N_frames
        D = self.Zp_mix.shape[0]

        muz0 = self.mu_zp0.T
        muz1 = self.mu_zp1.T

        varz0 = self.sig_zp0.T
        varz1 = self.sig_zp1.T

        Z_sampled = np.zeros((D, N, niter_MH - burnin))

        cpt = 0

        for n in np.arange(niter_MH):

            Z_prime = self.Zp_mix + np.sqrt(self.var_MH)*np.random.randn(D,N)
            with torch.no_grad():
                Z_prime_mapped_decoderA = self.torch2num(decoderA(self.num2torch(Z_prime.T), self.v0)).T

            # shape (F, N)
            speech_var_primeA = Z_prime_mapped_decoderA* self.g # apply gain

            acc_prob = ( np.sum( (np.log(self.speech_varA))
                        - (np.log(speech_var_primeA))
                        + ( (1./(self.speech_varA))
                        - (1./(speech_var_primeA)) )
                        * abs_Sp2, axis=0)
                        + .5*np.sum( pi_vals*(((self.Zp_mix-muz0)**2 - (Z_prime-muz0)**2)/varz0)+\
                                    (1.-pi_vals)*(((self.Zp_mix-muz1)**2 - (Z_prime-muz1)**2)/varz1) , axis=0) )

            is_acc = np.log(np.random.rand(1,N)) < acc_prob
            is_acc = is_acc.reshape((is_acc.shape[1],))

            self.Zp_mix[:,is_acc] = Z_prime[:,is_acc]

            with torch.no_grad():
                self.Z_mapped_decoderA = self.torch2num(decoderA(self.num2torch(self.Zp_mix.T),self.v0)).T

            self.speech_varA = self.Z_mapped_decoderA*self.g

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Zp_mix
                cpt += 1

        return Z_sampled

    def metropolis_hastings_MIN_VAEDec(self, decoderA, pi_vals, abs_Sp2, muz = None, varz = None, niter_MH=40, burnin=30, var_MH = 0.01):

        N = self.N_frames
        D = self.Zp_mix.shape[0]

        muz0 = self.mu_zp0.T
        muz1 = self.mu_zp1.T

        varz0 = self.sig_zp0.T
        varz1 = self.sig_zp1.T

        Z_sampled = np.zeros((D, N, niter_MH - burnin))

        cpt = 0

        for n in np.arange(niter_MH):

            Z_prime = self.Zp_mix + np.sqrt(self.var_MH)*np.random.randn(D,N)
            with torch.no_grad():
                Z_prime_mapped_decoderA = self.torch2num(decoderA(self.num2torch(Z_prime.T), self.v1)).T

            # shape (F, N)
            speech_var_primeA = Z_prime_mapped_decoderA* self.g # apply gain

            acc_prob = ( np.sum( (np.log(self.speech_varA))
                        - (np.log(speech_var_primeA))
                        + ( (1./(self.speech_varA))
                        - (1./(speech_var_primeA)) )
                        * abs_Sp2, axis=0)
                        + .5*np.sum( pi_vals*(((self.Zp_mix-muz0)**2 - (Z_prime-muz0)**2)/varz0)+\
                                    (1.-pi_vals)*(((self.Zp_mix-muz1)**2 - (Z_prime-muz1)**2)/varz1) , axis=0) )

            is_acc = np.log(np.random.rand(1,N)) < acc_prob
            is_acc = is_acc.reshape((is_acc.shape[1],))

            self.Zp_mix[:,is_acc] = Z_prime[:,is_acc]

            with torch.no_grad():
                self.Z_mapped_decoderA = self.torch2num(decoderA(self.num2torch(self.Zp_mix.T),self.v1)).T

            self.speech_varA = self.Z_mapped_decoderA*self.g

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Zp_mix
                cpt += 1

        return Z_sampled


    def run(self):

        self.g = self.g0.copy().squeeze()

        z0 = self.z0_init

        z0.requires_grad = False

        Z0 = z0.numpy().copy()

        W = self.W0.copy()
        H = self.H0.copy()

        Sig_s = np.zeros((self.N_freq, self.N_frames))

        Mu_s = self.X.copy()

        self.V = W @ H

        pi_vec = 0.5*np.ones(self.N_frames) # self.pi initialize posterior membership propbabilities

        Vm = np.zeros((self.N_freq, self.N_frames))
        S_hat = np.zeros_like(self.X)
        B_hat = np.zeros_like(self.X)

        self.Zp_mix = Z0

        with torch.no_grad():
            Z_mapped_decoder = self.torch2num(self.vae0.decode(self.num2torch(self.Zp_mix.T), self.v0)).T

        speech_varA = Z_mapped_decoder*self.g # apply gain

        self.speech_varA = speech_varA


        #%% Main loop

        cost_after_M_step = np.zeros((self.niter_alg, 1))

        for iter_ in np.arange(self.niter_alg):


            ############ E-z step ############

            z_samples = np.transpose(self.metropolis_hastings_MIN_VAE(self.vae0.decode, pi_vec, np.abs(Mu_s)**2+Sig_s,\
                                                    muz = None, varz = None, niter_MH=40, burnin=30, var_MH = 0.01),(1,2,0))

            with torch.no_grad():

                    Z_decoded0 =np.transpose(self.torch2num(self.vae0.decode(self.num2torch(z_samples),
                                                                                self.v1[:,None,:])),(2,0,1))     # shape (F, N, NumSamples)


            inv_gam0 = np.mean(1./Z_decoded0, axis = -1)


            Z_p0_term = np.mean((z_samples-self.mu_zp0[:,None,:])**2, axis = 1)/(2.*self.sig_zp0) + np.log(self.sig_zp0)
            Z_p0_term = Z_p0_term.T

            Z_p1_term = np.mean((z_samples-self.mu_zp1[:,None,:])**2, axis = 1)/(2.*self.sig_zp1) + np.log(self.sig_zp1)
            Z_p1_term = Z_p1_term.T

            z_p_term = Z_p1_term - Z_p0_term

            beta_diff = np.sum(z_p_term, axis = 0)

            beta_diff[beta_diff>30] = 30
            beta_diff[beta_diff<-30] = -30

            pi_vec = self.pi/(self.pi+(1.-self.pi) * np.exp(-beta_diff))

            self.pi = np.mean(pi_vec)


            gam = 1./inv_gam0

            Sig_s = (self.V*gam)/(self.V + gam)
            Mu_s = (gam/(self.V + gam))*self.X


            ############ M step ############

            Vm = np.abs(self.X-Mu_s)**2+Sig_s

            ############ Update W ############

            W = W*( (((self.V**-2) * Vm) @ H.T)
                    / ((self.V**-1) @ H.T) )

            W = np.maximum(W, self.eps)

            self.V = W @ H

            ############ Update H ############

            H = H*( (W.T @ ( (self.V**-2) * Vm ))
                    / (W.T @ (self.V**-1)))

            H = np.maximum(H, self.eps)

            norm_col_W = np.sum(np.abs(W), axis=0)
            W = W/norm_col_W[np.newaxis,:]
            H = H*norm_col_W[:,np.newaxis]

            self.V = W @ H


            if iter_ % 30 ==0:
                print("Proposed-   iter %d/%d - cost=%.4f -pi=%.4f\n" %
                      (iter_+1, self.niter_alg, cost_after_M_step[iter_], self.pi))


        #%% Compute speech estimate

        S_hat = (gam/(self.V + gam))*self.X
        B_hat = self.X - S_hat


        # Compute inverse STFT
        s_hat = librosa.istft(S_hat, hop_length=self.hop, win_length=self.wlen, window=self.win, length=self.T_orig)
        b_hat = librosa.istft(B_hat, hop_length=self.hop, win_length=self.wlen, window=self.win, length=self.T_orig)

        return s_hat, b_hat
