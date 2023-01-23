#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
License agreement in LICENSE.txt
"""

import torch
from torch import nn
import torch.nn.functional as F

#%% VAE with encoder trained using both audio and video. The latent variables are sampled from both "A" and "V"
# The prior for a- and v-encoders is different. The priors' parameters are learned.

class myVAE_MIN_VAE(nn.Module):


    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, batch_size=None, activation=None, activationv=None,
                 blockZ = 0., blockVenc = 1., blockVdec = 1., x_block = 0.0, landmarks_dim = 67*67, pv = 0):

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim #67*67 #256*5 #100 #67*67 #256*5
        self.hidden_dim_encoder = hidden_dim_encoder
        self.batch_size = batch_size
        self.activation = activation
        self.activationv = activationv
        self.model = None
        self.history = None
        self.x_block = x_block

        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)

        self.blockZ = blockZ
        self.blockVenc = blockVenc
        self.blockVdec = blockVdec

        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim)


        #### Define bottleneck layer ####

        self.latent_mean_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        self.latent_mean_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        ### Define prior parameters ###
        self.mupA = nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.zeros((1,self.latent_dim), requires_grad = False, device = 'cpu') #nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1,self.latent_dim)))
        self.mupV = nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.zeros((1,self.latent_dim), requires_grad = False, device = 'cpu') #nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1,self.latent_dim)))

        self.sigpA = nn.Parameter(torch.ones((1))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1)))
        self.sigpV = nn.Parameter(torch.ones((1))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1)))

    def encodeA(self, x, v):
        xv = self.encoder_layerX(x)
        he = self.activation(xv)

        return self.latent_mean_layer_a(he), self.latent_logvar_layer_a(he)

    def encodeV(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        xv = self.encoder_layerV(ve)
        he = self.activation(xv)

        return self.latent_mean_layer_v(he), self.latent_logvar_layer_v(he)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        zv = self.decoder_layerZ(z)
        hd = self.activation(zv)

        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu_a, logvar_a = self.encodeA(x, v)
        z_a = self.reparameterize(mu_a, logvar_a)

        mu_v, logvar_v = self.encodeV(x, v)
        z_v = self.reparameterize(mu_v, logvar_v)

        return self.decode(z_a, v), mu_a, logvar_a, self.decode(z_v, v), mu_v, logvar_v, self.mupA, self.sigpA, self.mupV, self.sigpV

    def print_model(self):

        print('------ Encoder -------')
        for layer in self.encoder_layers:
            print(layer)
            print(self.activation)

        print('------ Bottleneck -------')
        print(self.latent_mean_layer)
        print(self.latent_logvar_layer)

        print('------ Decoder -------')
        for layer in self.decoder_layers:
            print(layer)
            print(self.activation)
        print(self.output_layer)


#%% VAE with encoder trained using both audio and video. The latent variables are sampled from both "A" and "V".
#   the decoder is also trained conditioned on visual data


class myVAE_MIN_VAEDec(nn.Module):


    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, batch_size=None, activation=None, activationv=None,
                 blockZ = 0., blockVenc = 1., blockVdec = 1., aux_video = False, vae_joint = False, x_block = 0.0, landmarks_dim = 67*67):

        super(myVAE_AVEDec, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.landmarks_dim = landmarks_dim #67*67 #256*5 #100 #67*67 #256*5
        self.hidden_dim_encoder = hidden_dim_encoder
        self.batch_size = batch_size
        self.activation = activation
        self.activationv = activationv
        self.model = None
        self.history = None
        self.aux_video = aux_video
        self.vae_joint = vae_joint
        self.x_block = x_block

        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])

        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])
        self.encoder_layerV0 = nn.Linear(self.landmarks_dim, 512)
        self.decoder_layerV = nn.Linear(512, self.hidden_dim_encoder[0])

        if self.aux_video:
           self.aux_layerV0 = nn.Linear(self.latent_dim, 512)
           self.aux_layerV = nn.Linear(512, self.landmarks_dim)

        if self.vae_joint:
           self.joint_layerV = nn.Linear(self.latent_dim, 512)
           self.joint_mean = nn.Linear(512, self.landmarks_dim)
           self.joint_logvar = nn.Linear(512, 1)

        self.blockZ = blockZ
        self.blockVenc = blockVenc
        self.blockVdec = blockVdec

        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim)


        #### Define bottleneck layer ####

        self.latent_mean_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_a = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        self.latent_mean_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer_v = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        ### Define prior parameters ###
        self.mupA = nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.zeros((1,self.latent_dim), requires_grad = False, device = 'cpu') #nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1,self.latent_dim)))
        self.mupV = nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.zeros((1,self.latent_dim), requires_grad = False, device = 'cpu') #nn.Parameter(torch.zeros((1,self.latent_dim))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1,self.latent_dim)))

        self.sigpA = nn.Parameter(torch.ones((1))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1)))
        self.sigpV = nn.Parameter(torch.ones((1))) #torch.tensor(0.0, device = 'cuda') #nn.Parameter(torch.rand((1)))

    def encodeA(self, x, v):
        xv = self.encoder_layerX(x)
        he = self.activation(xv)

        return self.latent_mean_layer_a(he), self.latent_logvar_layer_a(he)

    def encodeV(self, x, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        xv = self.encoder_layerV(ve)
        he = self.activation(xv)

        return self.latent_mean_layer_v(he), self.latent_logvar_layer_v(he)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, v):
        ve = self.encoder_layerV0(v)
        ve = self.activationv(ve)
        zv = self.decoder_layerZ(z) + self.decoder_layerV(ve)
        hd = self.activation(zv)

        return torch.exp(self.output_layer(hd))

    def forward(self, x, v):
        mu_a, logvar_a = self.encodeA(x, v)
        z_a = self.reparameterize(mu_a, logvar_a)

        mu_v, logvar_v = self.encodeV(x, v)
        z_v = self.reparameterize(mu_v, logvar_v)

        return self.decode(z_a, v), mu_a, logvar_a, self.decode(z_v, v), mu_v, logvar_v, self.mupA, self.sigpA, self.mupV, self.sigpV

    def print_model(self):

        print('------ Encoder -------')
        for layer in self.encoder_layers:
            print(layer)
            print(self.activation)

        print('------ Bottleneck -------')
        print(self.latent_mean_layer)
        print(self.latent_logvar_layer)

        print('------ Decoder -------')
        for layer in self.decoder_layers:
            print(layer)
            print(self.activation)
        print(self.output_layer)
