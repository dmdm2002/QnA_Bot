import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import utils.hparams as hparams
import audio as Audio
from utils.utils import pad_1D, pad_2D, process_meta, standard_norm
from text import text_to_sequence, sequence_to_text
import time


class Dataset(Dataset):
    """
    FastSpeech2 학습을 위한 데이터셋 클래스
    텍스트, 멜 스펙트로그램, F0, 에너지, 지속시간 등의 특성을 처리하고
    효율적인 배치 학습을 위한 전처리를 수행합니다.
    """
    def __init__(self, filename="train.txt", sort=True):
        """
        데이터셋 초기화
        Args:
            filename (str): 학습 데이터 메타 파일 이름
            sort (bool): 배치 내 데이터 정렬 여부
        """
        # 메타 파일에서 기본 정보(파일명, 텍스트) 로드
        meta_file = os.path.join(hparams.preprocessed_path, filename)
        self.basename, self.text = process_meta(meta_file)

        # 특성 정규화를 위한 통계 데이터 로드
        self.mean_mel, self.std_mel = np.load(os.path.join(hparams.preprocessed_path, "mel_stat.npy"))  # 멜 스펙트로그램 평균/표준편차
        self.mean_f0, self.std_f0 = np.load(os.path.join(hparams.preprocessed_path, "f0_stat.npy"))    # 기본 주파수(F0) 평균/표준편차
        self.mean_energy, self.std_energy = np.load(os.path.join(hparams.preprocessed_path, "energy_stat.npy"))  # 에너지 평균/표준편차

        self.sort = sort

    def __len__(self):
        """데이터셋의 총 샘플 수 반환"""
        return len(self.text)

    def __getitem__(self, index):
        """
        개별 학습 샘플 로드
        Args:
            index (int): 데이터 인덱스
        Returns:
            dict: 학습에 필요한 모든 특성이 포함된 딕셔너리
        """
        t = self.text[index]
        basename = self.basename[index]
        # 텍스트를 음소 시퀀스로 변환
        phone = np.array(text_to_sequence(t, []))

        # 멜 스펙트로그램 로드
        mel_path = os.path.join(hparams.preprocessed_path, "mel", "{}-mel-{}.npy".format(hparams.dataset, basename))
        mel_target = np.load(mel_path)

        # 정렬 정보(duration) 로드
        D_path = os.path.join(hparams.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hparams.dataset, basename))
        D_target = np.load(D_path)

        # 피치 정보(F0) 로드
        f0_path = os.path.join(hparams.preprocessed_path, "f0", "{}-f0-{}.npy".format(hparams.dataset, basename))
        f0_target = np.load(f0_path)

        # 에너지 정보 로드
        energy_path = os.path.join(hparams.preprocessed_path, "energy", "{}-energy-{}.npy".format(hparams.dataset, basename))
        energy_target = np.load(energy_path)

        # 모든 특성을 딕셔너리로 반환
        sample = {
            "id": basename,
            "text": phone,
            "mel_target": mel_target,
            "D": D_target,
            "f0": f0_target,
            "energy": energy_target
        }

        return sample

    def reprocess(self, batch, cut_list):
        """
        배치 데이터 전처리
        Args:
            batch (list): 배치 데이터 리스트
            cut_list (list): 배치 내 인덱스 리스트
        Returns:
            dict: 전처리된 배치 데이터
        """
        # 배치 내 각 샘플의 ID와 텍스트 추출
        ids = [batch[ind]["id"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        
        # 특성들의 정규화 적용
        mel_targets = [standard_norm(batch[ind]["mel_target"], self.mean_mel, self.std_mel, is_mel=True) for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [standard_norm(batch[ind]["f0"], self.mean_f0, self.std_f0, is_mel=False) for ind in cut_list]
        energies = [standard_norm(batch[ind]["energy"], self.mean_energy, self.std_energy, is_mel=False) for ind in cut_list]

        # 텍스트와 지속시간의 길이 일치 여부 확인
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print('The dimension of text and duration sould be same')
                print('text: ', sequence_to_text(text))
                print(text, text.shape, D, D.shape)
                
        # 텍스트와 멜 스펙트로그램 길이 계산
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        # 패딩 적용하여 배치 내 길이 통일
        texts = pad_1D(texts)          # 텍스트 패딩
        Ds = pad_1D(Ds)               # 지속시간 패딩
        mel_targets = pad_2D(mel_targets)  # 멜 스펙트로그램 패딩
        f0s = pad_1D(f0s)             # F0 패딩
        energies = pad_1D(energies)    # 에너지 패딩
        log_Ds = np.log(Ds + hparams.log_offset)  # 지속시간 로그 변환

        # 전처리된 데이터를 딕셔너리로 반환
        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel}

        return out

    def collate_fn(self, batch):
        """
        배치 생성을 위한 콜레이트 함수
        Defulat 인 경우 배치를 생성하는 역할을 한다.
        Dataset을 호출하였을 때, 배치가 나누어지는 이유는 이 함수가 있기 때문이다.
        Args:
            batch (list): 배치 데이터 리스트
        Returns:
            list: 전처리된 서브 배치 리스트
        """
        # 텍스트 길이에 따라 정렬
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        # 서브 배치로 분할
        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                # 길이순 정렬된 인덱스로 서브 배치 구성
                cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                # 순차적으로 서브 배치 구성
                cut_list.append(np.arange(i*real_batchsize, (i+1)*real_batchsize))
        
        # 각 서브 배치 전처리
        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output
