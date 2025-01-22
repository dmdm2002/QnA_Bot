import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils.hparams as hp
import os

import numpy as np
import argparse
import time
import tqdm

from fastspeech2 import FastSpeech2
from utils.loss import FastSpeech2Loss
from utils.dataset import Dataset
from utils.optimizer import ScheduledOptim
from evaluate import evaluate
import utils.utils as utils
import audio as Audio


def main(args):
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)

    if hp.device == 'cuda':
        hp.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'-----Device: {hp.device}-----')

    # Get dataset
    dataset = Dataset("train.txt") 
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    model = FastSpeech2().to(hp.device)
    print("-----Model Defined-----")

    num_param = utils.get_param_num(model)
    print(f'-----Number of model parameters: {num_param}-----')

    # Get optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay=hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(hp.device)

    print("-----Optimizer and Loss Function Defined-----")

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(f'{hp.checkpoint_path}/ckp_{args.restore_step}.pth.tar')
        model.load_state_dict(checkpoint['state_dict_model'])
        optimizer.load_state_dict(checkpoint['state_dict_optim'])
        print(f"-----Checkpoint {args.restore_step} loaded-----")
    except:
        print("-----No checkpoint found-----")
        print("-----Training from scratch-----")
        os.makedirs(hp.checkpoint_path, exist_ok=True)

    # read params
    mean_mel, std_mel = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "mel_stat.npy")), dtype=torch.float).to(hp.device)
    mean_f0, std_f0 = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "f0_stat.npy")), dtype=torch.float).to(hp.device)
    mean_energy, std_energy = torch.tensor(np.load(os.path.join(hp.preprocessed_path, "energy_stat.npy")), dtype=torch.float).to(hp.device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)

    # Load vocoder
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
        vocoder.to(hp.device)
        print('-----Vocoder Loaded-----')
    else:
        vocoder = None
        print('-----No vocoder loaded-----')
    
    # Initialize logger
    if not os.path.exists(hp.log_path):
        os.makedirs(hp.log_path, exist_ok=True)
        os.makedirs(f'{hp.log_path}/train', exist_ok=True)
        os.makedirs(f'{hp.log_path}/val', exist_ok=True)
    train_logger = SummaryWriter(f'{hp.log_path}/train')
    val_logger = SummaryWriter(f'{hp.log_path}/val')

        # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    
    # Training
    model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        for i, batch in enumerate(tqdm.tqdm(loader, desc=f"Epcoh : [{epoch}/{hp.epochs}]")):
            for j, data_of_batch in enumerate(batch):
                start_time = time.perf_counter()

                current_step = i*hp.batch_size + j + args.restore_step + epoch*len(loader)*hp.batch_size + 1
                
                # Get Data
                text = torch.from_numpy(data_of_batch["text"]).long().to(hp.device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(hp.device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(hp.device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(hp.device)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(hp.device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(hp.device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(hp.device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(hp.device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                
                # Forward
                mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(
                    text, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)
                
                # Cal Loss
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask)
                total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss
                 
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
                with open(os.path.join(hp.log_path, "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")
                with open(os.path.join(hp.log_path, "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")
                with open(os.path.join(hp.log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")
                with open(os.path.join(hp.log_path, "duration_loss.txt"), "a") as f_d_loss:
                    f_d_loss.write(str(d_l)+"\n")
                with open(os.path.join(hp.log_path, "f0_loss.txt"), "a") as f_f_loss:
                    f_f_loss.write(str(f_l)+"\n")
                with open(os.path.join(hp.log_path, "energy_loss.txt"), "a") as f_e_loss:
                    f_e_loss.write(str(e_l)+"\n")
                 
                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)
                    
                    with open(os.path.join(hp.log_path, "log.txt"), "a") as f_log:
                        f_log.write(str1 + "\n")
                        f_log.write(str2 + "\n")
                        f_log.write(str3 + "\n")
                        f_log.write("\n")

                train_logger.add_scalar('Loss/total_loss', t_l, current_step)
                train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                train_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                train_logger.add_scalar('Loss/energy_loss', e_l, current_step)
                
                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))

                if current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, f_l, e_l, m_l, m_p_l = evaluate(model, current_step, vocoder)
                        t_l = d_l + f_l + e_l + m_l + m_p_l
                        
                        val_logger.add_scalar('Loss/total_loss', t_l, current_step)
                        val_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                        val_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                        val_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                        val_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                        val_logger.add_scalar('Loss/energy_loss', e_l, current_step)

                    model.train()

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    args = parser.parse_args()

    main(args)
