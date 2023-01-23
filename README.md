## Mixture of Inference Networks for Audio-visual Speech Enhancement

This repository contains PyTorch implementations of the audio-visual speech enhancement method based on the mixture of inference networks variational autoencoder (MIN-VAE) framework presented in [1].

The VAE architectures are defined in `AV_VAE.py`.

To train the MIN-VAE model using clean audio and video data, use `train_MIN_VAE.py`.

`TCD_TIMIT.py` contains a custom dataset loader for the TCD-TIMIT dataset.

The MCEM algorithm for speech enhancement is impelemented in `SE_utils.py`.

To enhacne a given speech, use `speech_enhancer.py`.

## Reference:

[1] M. Sadeghi and X. Alameda-Pineda, “Mixture of Inference Networks for VAE-based Audio-visual Speech Enhancement,” IEEE Transactions on Signal Processing, vol. 69, pp. 1899-1909, March 2021.

## Contact:

If you have any questions, please do not hesitate to contact me: mostafa[dot]sadeghi@inria[dot]fr
