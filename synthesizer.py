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

    # Load neurips MelGAN for mel2audio synthesis
    self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    melgan_ckpt = torch.load(v_checkpoint_path, map_location='cpu')
    self.vocoder.mel2wav.load_state_dict(melgan_ckpt)


  def synthesize(self, text):
    # TODO choose language?
    cleaner = ['catalan_cleaners']

    # Prepare text input
    sequence = np.array(text_to_sequence(text, cleaner))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cpu', dtype=torch.int64)

    # TODO run within the queue
    # decode text input
    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)

    # TODO run within the queue
    # Synthesize using neurips Melgan
    with torch.no_grad():
        audio = self.vocoder.inverse(mel_outputs_postnet.float())
    audio_numpy = audio[0].data.cpu().numpy()

    # normalize and convert from float32 to int16 pcm
    audio_numpy /= np.max(np.abs(audio_numpy))
    audio_numpy *= 32768

    # out
    out = io.BytesIO()

    # save
    scipy.io.wavfile.write(out, 22050, audio_numpy.astype(np.int16))

    return out.getvalue()
