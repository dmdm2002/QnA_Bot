import torch
import torch.nn as nn
import numpy as np
import fastspeech2.utils.hparams as hp
import os
import re

from fastspeech2.fastspeech2 import FastSpeech2
from fastspeech2.text import text_to_sequence, sequence_to_text
import fastspeech2.utils.utils as utils

from g2pk import G2p
from jamo import h2j

from kss import split_sentences
import sounddevice as sd


def kor_preprocess(text):
    g2p=G2p()
    phone = g2p(text)
    # print('after g2p: ',phone)
    phone = h2j(phone)
    # print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    # print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sil}', phone)
    # print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone,hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(hp.device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = FastSpeech2()
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()

    return model


def synthesize(model, vocoder, text, sentence, dur_pitch_energy_aug, prefix=''):
    sentence = sentence[:10]

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(
        hp.device)
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

    if vocoder is not None:
        if hp.vocoder.lower() == "vocgan":
           audio = utils.vocgan_infer(mel_postnet_torch, vocoder)

    # utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)],
    #                 ['Synthesized Spectrogram'],
    #                 filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))

    return mel_postnet_torch, audio

def inference(answer):
    dur_pitch_energy_aug = [1.0, 1.0, 1.0]

    answer_list = answer.split('\n')

    sentences = []
    for split_answer in answer_list:
        split_answer = split_sentences(split_answer)
        sentences += split_answer

    model = get_FastSpeech2(num=290000).to(hp.device)
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    else:
        vocoder = None
        print('vocoder가 존재하지 않습니다.')

    # text to audio
    combined_audio = []
    for sentence in sentences:
        text = kor_preprocess(sentence)
        mel_postnet_torch, audio = synthesize(model, vocoder, text, sentence, dur_pitch_energy_aug)
        combined_audio.append(audio)

    # Combine all audio arrays
    combined_audio = np.concatenate(combined_audio, axis=0)

    # Normalize the combined audio for playback
    combined_audio = combined_audio / np.max(np.abs(combined_audio))

    sd.play(combined_audio, samplerate=hp.sampling_rate)
    sd.wait()
