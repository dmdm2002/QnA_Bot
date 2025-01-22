import torch
import torch.nn as nn
import numpy as np
import utils.hparams as hp
import os
import argparse
import re
from string import punctuation

from fastspeech2 import FastSpeech2
from vocoder import vocgan_generator

from text import text_to_sequence, sequence_to_text
import utils
import audio as Audio

import codecs
from g2pk import G2p
from jamo import h2j


def kor_preprocess(text):
    g2p=G2p()
    phone = g2p(text)
    print('after g2p: ',phone)
    phone = h2j(phone)
    print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sil}', phone)
    print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone,hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(hp.device)


def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()

    return model


def synthesize(model, vocoder, text, sentence, dur_pitch_energy_aug, prefix=''):
    sentence = sentence[:10]  # long filename will result in OS Error

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(hp.device)
    mean_f0, std_f0 = f0_stat = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")),
                                             dtype=torch.float).to(hp.device)
    mean_energy, std_energy = energy_stat = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")),
                                                         dtype=torch.float).to(hp.device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(hp.device)

    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len,
                                                                                           dur_pitch_energy_aug=dur_pitch_energy_aug,
                                                                                           f0_stat=f0_stat,
                                                                                           energy_stat=energy_stat)

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    mel_torch = utils.de_norm(mel_torch.transpose(1, 2), mean_mel, std_mel)
    mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), mean_mel, std_mel).transpose(1, 2)
    f0_output = utils.de_norm(f0_output, mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(energy_output, mean_energy, std_energy).squeeze().detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    Audio.tools.inv_mel_spec(mel_postnet_torch[0],
                             os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))

    if vocoder is not None:
        if hp.vocoder.lower() == "vocgan":
            utils.vocgan_infer(mel_postnet_torch, vocoder,
                               path=os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))

    utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)],
                    ['Synthesized Spectrogram'],
                    filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))
