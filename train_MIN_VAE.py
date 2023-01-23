#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:51:14 2020

@author: Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch import optim
from TCD_TIMIT import TIMIT
import matplotlib.pyplot as plt
from AV_VAE import myVAE_MIN_VAE, myVAE_MIN_VAEDec
import os
from torchvision.utils import save_image

import argparse

def main(args):

    #%% network parameters

    input_dim = 513
    latent_dim = 32
    hidden_dim_encoder = [128]
    activation = torch.tanh # for audio net
    activationv = nn.ReLU() # for visual net


    #%% training parameters

    data_dir_tr = '/pathTo/train_data'
    data_dir_val = '/pathTo/valid_data'

    # We assume that the audio data have already been preprocessed (spectrogram), and saved in .npy format. But you can adapt the code and directly use the TIMIT data loader
    file_list_tr = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir_tr)
             for name in files
             if name.endswith('audio.npy')]

    file_list_val = [os.path.join(root, name)
             for root, dirs, files in os.walk(data_dir_val)
             for name in files
             if name.endswith('audio.npy')]

    print('Number of training samples: ', len(file_list_tr))
    print('Number of validation samples: ', len(file_list_val))


    lr = args.lr
    pi_alph = args.pi_alph

    epoches = 100
    batch_size = 128
    save_dir = '/pathTo/saved_dir'

    device = 'cuda'

    # check Pretraining
    vae_mode = args.vae_mode


    #%%

    # use myVAE_MIN_VAEDec when the visual information is used in the decoder as well
    vae = myVAE_MIN_VAE(input_dim=input_dim, latent_dim=latent_dim,
            hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size,
            activation=activation, activationv=activationv).to(device)


    #%% Pretraining:

    vae_A = myVAE_MIN_VAE(input_dim=input_dim, latent_dim=latent_dim,
            hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size,
            activation=activation, activationv=activationv).to(device)

    vae_A.load_state_dict(torch.load('/pathTo/A_VAE.pt', map_location='cpu'), strict=False)


    vae_V = myVAE_MIN_VAE(input_dim=input_dim, latent_dim=latent_dim,
            hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size,
            activation=activation, activationv=activationv).to(device)

    vae_V.load_state_dict(torch.load('/pathTo/V_VAE.pt', map_location='cpu'), strict=False)

    pret_a = vae_A.state_dict()
    pret_v = vae_V.state_dict()

    vae_kvpair = vae.state_dict()

    vae_kvpair['encoder_layerV0.weight'] = pret_v['encoder_layerV0.weight']
    vae_kvpair['encoder_layerV0.bias'] = pret_v['encoder_layerV0.bias']
    vae_kvpair['encoder_layerV.weight'] = pret_v['encoder_layerV.weight']
    vae_kvpair['encoder_layerV.bias'] = pret_v['encoder_layerV.bias']
    vae_kvpair['encoder_layerX.weight'] = pret_a['encoder_layerX.weight']
    vae_kvpair['encoder_layerX.bias'] = pret_a['encoder_layerX.bias']

    vae_kvpair['latent_mean_layer_v.weight'] = pret_v['latent_mean_layer_v.weight']
    vae_kvpair['latent_logvar_layer_v.weight'] = pret_v['latent_logvar_layer_v.weight']

    vae_kvpair['latent_mean_layer_v.bias'] = pret_v['latent_mean_layer_v.bias']
    vae_kvpair['latent_logvar_layer_v.bias'] = pret_v['latent_logvar_layer_v.bias']

    vae_kvpair['decoder_layerZ.weight'] = pret_a['decoder_layerZ.weight']
    vae_kvpair['output_layer.weight'] = pret_a['output_layer.weight']

    vae_kvpair['decoder_layerZ.bias'] = pret_a['decoder_layerZ.bias']
    vae_kvpair['output_layer.bias'] = pret_a['output_layer.bias'] 


    vae_kvpair['latent_mean_layer_a.weight'] = pret_a['latent_mean_layer_a.weight']
    vae_kvpair['latent_logvar_layer_a.weight'] = pret_a['latent_logvar_layer_a.weight']

    vae_kvpair['latent_mean_layer_a.bias'] = pret_a['latent_mean_layer_a.bias']
    vae_kvpair['latent_logvar_layer_a.bias'] = pret_a['latent_logvar_layer_a.bias']

    vae.load_state_dict(vae_kvpair)

    vae.mupA = vae_A.mupA
    vae.sigpA = vae_A.sigpA

    vae.mupV = vae_V.mupV
    vae.sigpV = vae_V.sigpV

    print(vae.mupA, vae.sigpA)
    print(vae.mupV, vae.sigpV)

    #%% Pretraining:

    pi_alph = torch.from_numpy(np.asarray(pi_alph).astype(np.float32)).to(device)
    pi_vecs = 0.5*torch.ones(571720,1).to(device)

    # optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, vae.parameters()), lr=lr)

    def loss_function(xi, recon_xi_a, mui_a, logvari_a, recon_xi_v, mui_v, logvari_v, mupAi, sigpAi, mupVi, sigpVi, pi_s = 0.5):

        pi_si = pi_s.unsqueeze(1)
        recon_a = torch.sum( pi_si * (torch.log(recon_xi_a) + xi/(recon_xi_a) ))
        recon_v = torch.sum( (1.-pi_si) * (torch.log(recon_xi_v) + xi/(recon_xi_v) ))

        KLD_a = -0.5 * torch.sum( pi_si * (logvari_a- sigpAi.log() - ((mui_a-mupAi).pow(2) + logvari_a.exp())/sigpAi))
        KLD_v = -0.5 * torch.sum( (1.-pi_si) * (logvari_v - sigpVi.log() - ((mui_v-mupVi).pow(2) + logvari_v.exp())/sigpVi))

        return recon_a + KLD_a + recon_v + KLD_v

    #%% main loop for training

    save_loss_dir_tr = os.path.join(save_dir, 'Train_loss_'+str(vae_mode))

    # to track the training loss as the model trains
    train_losses = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []

    def num2torch(x):
        y = torch.from_numpy(x.astype(np.float32))
        return y

    epoch0 = 0

    for epoch in range(epoch0, epoches):

        vae.train()

        rand_seeds = torch.zeros(len(file_list_tr),1)

        for idx_, file_i in enumerate(file_list_tr):

            path, file_name = os.path.split(file_i)
            batch_idx = int(file_name[:-10])
            batch_audio = num2torch(np.load(file_i))
            batch_video = num2torch(np.load(os.path.join(path, str(file_name[:-10])+'_video.npy')))

            rand_seed = torch.rand(1)
            rand_seeds[idx_] = rand_seed
            torch.manual_seed(rand_seed)
            inds_noisy = np.random.randint(batch_audio.shape[0], size=int(0.5*batch_audio.shape[0]))

            batch_audio_occ =  batch_audio.clone()
            batch_audio_occ[inds_noisy,:] += 50*torch.rand(batch_audio_occ[inds_noisy,:].shape)

            batch_audio_occ = batch_audio_occ.to(device)

            batch_audio = batch_audio.to(device)
            batch_video = batch_video.to(device)

            recon_batch_a, mu_a, logvar_a, recon_batch_v, mu_v, logvar_v, mupA, sigpA, mupV, sigpV = vae(batch_audio_occ, batch_video)
            loss = loss_function(batch_audio, recon_batch_a, mu_a, logvar_a, recon_batch_v, mu_v, logvar_v, mupA, sigpA, mupV, sigpV, pi_s = pi_vecs[batch_idx*batch_size:(batch_idx+1)*batch_size,0])

            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Update alpha_n:
        vae.eval()

        for idx_, file_i in enumerate(file_list_tr):

            path, file_name = os.path.split(file_i)
            batch_idx = int(file_name[:-10])
            batch_audio = num2torch(np.load(file_i))
            batch_video = num2torch(np.load(os.path.join(path, str(file_name[:-10])+'_video.npy')))

            rand_seed = rand_seeds[idx_]
            torch.manual_seed(rand_seed)
            inds_noisy = np.random.randint(batch_audio.shape[0], size=int(0.5*batch_audio.shape[0]))

            batch_audio_occ =  batch_audio.clone()
            batch_audio_occ[inds_noisy,:] += 50*torch.rand(batch_audio_occ[inds_noisy,:].shape)

            batch_audio_occ = batch_audio_occ.to(device)

            batch_audio = batch_audio.to(device)
            batch_video = batch_video.to(device)
            recon_batchAi, mu_ai, logvar_ai, recon_batchVi, mu_vi, logvar_vi, mupA, sigpA, mupV, sigpV  = vae(batch_audio_occ, batch_video)

            # Update alpha_n:
            with torch.no_grad():
                sum_s_term = torch.log(recon_batchAi) + batch_audio/recon_batchAi - (torch.log(recon_batchVi) + batch_audio/recon_batchVi)
                KL_term = torch.sum(sigpA.log()-logvar_ai + (torch.abs(mu_ai-mupA).pow(2)+logvar_ai.exp())/sigpA, dim = 1) - torch.sum(sigpV.log()-logvar_vi + (torch.abs(mu_vi-mupV).pow(2)+logvar_vi.exp())/sigpV, dim = 1)
                beta_diff = torch.sum(sum_s_term, dim = 1) + 0.5*(KL_term)

                beta_diff[beta_diff>30] = 30
                beta_diff[beta_diff<-30] = -30

                pi_vec = pi_alph/(pi_alph + (1.-pi_alph)*(beta_diff.exp()))
                pi_vec.requires_grad = False

                pi_vecs[batch_idx*batch_size:(batch_idx+1)*batch_size,0] = pi_vec

        pi_vecs.requires_grad = False

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.sum(train_losses) / 571720

        avg_train_losses = np.append(avg_train_losses, train_loss)

        epoch_len = len(str(epoches))

        print_msg = (f'====> Epoch: [{epoch:>{epoch_len}}/{epoches:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'pi: {pi_alph}')

        print(print_msg)

        np.save(save_loss_dir_tr, avg_train_losses)

        # clear lists to track next epoch
        train_losses = []

        # Update pi_alph:
        pi_alph = torch.mean(pi_vecs)

        save_file = os.path.join(save_dir, 'final_model_'+str(vae_mode)+'.pt')
        torch.save(vae.state_dict(), save_file)

        save_pvecs = os.path.join(save_dir, 'pvecs_'+str(vae_mode)+'.npy')
        np.save(save_pvecs, pi_vecs.cpu().numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--pgrad", type=float, default=-0.2)
    parser.add_argument("--vae_mode", type=str, default='a_vae', help='name of the used vae net')
    parser.add_argument("--beta", type=float, default=0.5, help='weight between log p(x|z) and log p(x|zp)')
    parser.add_argument("--pi_alph", type=float, default=0.5, help='weight between log p(x|z) and log p(x|zp)')

    args = parser.parse_args()

    main(args)
