# -*- coding: utf-8 -*-
#

import sys, os

import numpy as np
import time

import torch
import librosa

#from .model import Tacotron2
#from .layers import TacotronSTFT, STFT
#from .audio_processing import griffin_lim

from hparams_tts import create_hparams
from train_tts import load_model
from text import text_to_sequence
from melgan.model.generator import Generator
from melgan.utils.hparams import load_hparam

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def load_tts_model(checkpoint_path=None, melgan_path=None):

    # set-up params
    hparams = create_hparams()

    # load model from checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
    _ = model.eval()

    # Load MelGAN for mel2audio synthesis and denoiser
    checkpoint = torch.load(melgan_path, map_location='cpu')
    #TODO base path = path of the file
    hp_melgan = load_hparam("./melgan/config/default.yaml")
    vocoder_model = Generator(80)  # Number of mel channels
    vocoder_model.load_state_dict(checkpoint['model_g'])
    vocoder_model.eval(inference=False)

    return model, vocoder_model, hparams

def speechGeneration(model, vocoder_model, hparams, text,
                     outAudioPath, removeBias=False, lang='ca'):

    if lang == 'en':
        cleaner = ['english_cleaners']
    elif lang == 'ca':
        cleaner = ['catalan_cleaners']
    else:
        raise ValueError('Unknown language %s'%lang)

    # text pre-processing
    text = text.replace('\n\n', '')
    text = text.replace('\n', '')

    # Prepare text input
    sequence = np.array(text_to_sequence(text, cleaner))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

    # decode text input
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    # Synthesize audio from spectrogram using WaveGlow
    with torch.no_grad():
        audio = vocoder_model.inference(mel_outputs_postnet)
        #audio_numpy = audio.data.cpu().detach().numpy()
        audio_numpy = audio.data.cpu().numpy()
    

    # (Optional) Remove WaveGlow bias
    #if removeBias:
    #    audio = denoiser(audio, strength=0.01)[:, 0]

    # save
    #audio_numpy = np.array(audio_numpy, dtype=np.float64)
    #audio_numpy /= np.max(np.abs(audio_numpy))
    #write('/content/drive/My Drive/test_wavs/pau_test02_melgan.wav', rate, audio_numpy)

    librosa.output.write_wav(outAudioPath, audio_numpy.astype(np.float64), 22050, norm=True)

    return


if __name__ == "__main__":

    # load model
    start = time.time()
    model, melgan, hparams = load_tts_model(checkpoint_path="models/upc_ona_tacotron2.pt", melgan_path="models/melgan_LJ11_epoch.pt")
    print('model loaded in: ', time.time() - start, 'seconds')

    # generate speech and save audio
    inputText = "Les formes innovadores de la cooperació social desborden i enriqueixen el bagatge de l’economia social i solidària catalana."
    outAudioPath = "test_ca.wav"

    start = time.time()
    speechGeneration(model, melgan, hparams, inputText, outAudioPath, lang='ca')
    print('inference done in: ', time.time() - start, 'seconds')
