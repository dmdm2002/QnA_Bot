import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os
import argparse
import re

from fastspeech2 import FastSpeech2
from utils.loss import FastSpeech2Loss

from utils.dataset import Dataset
from text import text_to_sequence, sequence_to_text

import utils.hparams as hp
import utils.utils as utils
import audio as Audio


def get_checkpoint(num):
    ckp_path = f'{hp.checkpoint_path}/ckp_{num}.pth.tar'
    model = FastSpeech2()

    model.load_state_dict(torch.load(ckp_path)['state_dict_model'])
    model.requires_grad = False
    model.eval()

    return model

def evaluate(model, step, vocoder=None, device='cuda'):
    """
    FastSpeech2 모델의 예측결과를 평가하는 함수
    """
    model.eval()
    torch.manual_seed(hp.seed)

    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(device)

    eval_path = hp.eval_path
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    dataset = Dataset(filename="test.txt", sort=False)
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=False, collate_fn=dataset.collate_fn, drop_last=False, num_workers=0)

    loss_fn = FastSpeech2Loss()

    # Evaluation
    # evaluation을 위한 loss들의 list
    d_l = []
    f_l = []
    e_l = []
    mel_l = []
    mel_p_l = []
    current_step = 0
    idx = 0
    
    # batch size 내부의 Iteration
    for i, batchs in enumerate(loader):
        for j, data_of_batch in enumerate(batchs):
            # Get Data
            id_ = data_of_batch["id"]
            text = torch.from_numpy(data_of_batch["text"]).long().to(device)
            mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            log_D = torch.from_numpy(data_of_batch["log_D"]).int().to(device)
            f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
            energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
            src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
            mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
            max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
            max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

            # Forward
            with torch.no_grad():
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, out_mel_len = model(
                    text, src_len, mel_len, log_D, f0, energy, max_src_len, max_mel_len
                )

                # Calculate Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = loss_fn(
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask
                )

                d_l.append(d_loss.item())
                f_l.append(f_loss.item())
                e_l.append(e_loss.item())
                mel_l.append(mel_loss.item())
                mel_p_l.append(mel_postnet_loss.item())

                if idx == 0 and vocoder is not None:
                    # Vocoder 실행 및 plotting spectrogram
                    for k in range(1):
                        basename = id_[k]  # 현재 처리중인 음성 파일의 기본 이름
                        gt_length = mel_len[k]  # 실제(ground truth) 멜 스펙트로그램의 길이
                        out_length = out_mel_len[k]  # 모델이 출력한 멜 스펙트로그램의 길이

                        # 타겟(실제) 멜 스펙트로그램 데이터 추출
                        mel_target_torch = mel_target[k:k+1, :out_length]  # 배치에서 하나의 샘플 선택
                        mel_target_ = mel_target[k, :gt_length]  # ground truth 길이만큼 선택

                        # 모델이 생성한 멜 스펙트로그램 데이터 추출
                        mel_postnet_torch = mel_postnet_output[k:k+1, :out_length]
                        mel_postnet = mel_postnet_output[k, :out_length]
                        
                        # 타겟 데이터의 정규화 해제 및 차원 변환
                        mel_target_torch = utils.de_norm(mel_target_torch, mean_mel, std_mel).transpose(1, 2).detach()
                        mel_target_ = utils.de_norm(mel_target_, mean_mel, std_mel).cpu().transpose(0, 1).detach()

                        # 생성된 데이터의 정규화 해제 및 차원 변환
                        mel_postnet_torch = utils.de_norm(mel_postnet_torch, mean_mel, std_mel).transpose(1, 2).detach()
                        mel_postnet = utils.de_norm(mel_postnet, mean_mel, std_mel).cpu().transpose(0, 1).detach()

                        if hp.vocoder == "vocgan":
                            utils.vocgan_infer(mel_target_torch, vocoder, path=os.path.join(hp.eval_path, 'eval_groundtruth_{}_{}.wav'.format(basename, hp.vocoder)))   
                            utils.vocgan_infer(mel_postnet_torch, vocoder, path=os.path.join(hp.eval_path, 'eval_step_{}_{}_{}.wav'.format(step, basename, hp.vocoder)))  
                        np.save(os.path.join(hp.eval_path, 'eval_step_{}_{}_mel.npy'.format(step, basename)), mel_postnet.numpy())
                        
                        f0_ = f0[k, :gt_length]
                        energy_ = energy[k, :gt_length]
                        f0_output_ = f0_output[k, :out_length]
                        energy_output_ = energy_output[k, :out_length]

                        # 생성된 데이터의 정규화 및 GPU 할당 해제
                        f0_ = utils.de_norm(f0_, mean_f0, std_f0).detach().cpu().numpy()
                        f0_output_ = utils.de_norm(f0_output, mean_f0, std_f0).detach().cpu().numpy()
                        energy_ = utils.de_norm(energy_, mean_energy, std_energy).detach().cpu().numpy()
                        energy_output_ = utils.de_norm(energy_output_, mean_energy, std_energy).detach().cpu().numpy()
 
                        utils.plot_data([(mel_postnet.numpy(), f0_output_, energy_output_), (mel_target_.numpy(), f0_, energy_)], 
                            ['Synthesized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(hp.eval_path, 'eval_step_{}_{}.png'.format(step, basename)))
                        idx += 1
                    print("done")
            current_step += 1

    d_l = sum(d_l) / len(d_l)
    f_l = sum(f_l) / len(f_l)
    e_l = sum(e_l) / len(e_l)
    mel_l = sum(mel_l) / len(mel_l)
    mel_p_l = sum(mel_p_l) / len(mel_p_l) 
                    
    str1 = "FastSpeech2 Step {},".format(step)
    str2 = "Duration Loss: {}".format(d_l)
    str3 = "F0 Loss: {}".format(f_l)
    str4 = "Energy Loss: {}".format(e_l)
    str5 = "Mel Loss: {}".format(mel_l)
    str6 = "Mel Postnet Loss: {}".format(mel_p_l)

    print("\n" + str1)
    print(str2)
    print(str3)
    print(str4)
    print(str5)
    print(str6)

    with open(os.path.join(hp.log_path, "eval.txt"), "a") as f_log:
        f_log.write(str1 + "\n")
        f_log.write(str2 + "\n")
        f_log.write(str3 + "\n")
        f_log.write(str4 + "\n")
        f_log.write(str5 + "\n")
        f_log.write(str6 + "\n")
        f_log.write("\n")
        
    model.train()

    return d_l, f_l, e_l, mel_l, mel_p_l
