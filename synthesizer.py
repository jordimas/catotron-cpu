import os
import io
import torch
import scipy
import numpy as np

from hparams_tts import create_hparams
from train_tts import load_model
from text import text_to_sequence
from melgan.model.generator import Generator
from melgan.utils.hparams import load_hparam

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

class Synthesizer:
  def load(self, t_checkpoint_path, v_checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)

    # set-up params
    hparams = create_hparams()

    # load model from checkpoint
    self.model = load_model(hparams)
    self.model.load_state_dict(torch.load(t_checkpoint_path,
                                          map_location='cpu')['state_dict'])
    _ = self.model.eval()

    # Load MelGAN for mel2audio synthesis
    checkpoint = torch.load(v_checkpoint_path, map_location='cpu')
    hp_melgan = load_hparam(os.path.join(PROJECT_PATH,
                                         "melgan/config/default.yaml"))
    self.vocoder_model = Generator(80)
    self.vocoder_model.load_state_dict(checkpoint['model_g'])
    self.vocoder_model.eval(inference=False)

  def synthesize(self, text):
    # TODO choose language?
    cleaner = ['catalan_cleaners']

    # Prepare text input
    sequence = np.array(text_to_sequence(text, cleaner))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()

    # TODO run within the queue
    # decode text input
    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)

    # TODO run within the queue
    # Synthesize audio from spectrogram using Melgan
    with torch.no_grad():
        audio = self.vocoder_model.inference(mel_outputs_postnet)
        audio_numpy = audio.data.cpu().numpy()

    # out
    out = io.BytesIO()

    # save
    scipy.io.wavfile.write(out, 22050, audio_numpy.astype(np.int16))

    return out.getvalue()
