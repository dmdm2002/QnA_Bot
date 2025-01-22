import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from audio.audio_processing import dynamic_range_compression
from audio.audio_processing import dynamic_range_decompression
from audio.audio_processing import window_sumsquare


class STFT(torch.nn.Module):
    """
    Short-Time Fourier Transform (STFT)를 PyTorch로 구현한 클래스
    오디오 신호를 시간-주파수 도메인으로 변환하고 다시 복원하는 기능 제공
    adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft
    """

    def __init__(self, filter_length, hop_length, win_length, window='hann'):
        """
        STFT 클래스 초기화
        Args:
            filter_length (int): FFT 크기
            hop_length (int): 프레임 간 이동 거리
            win_length (int): 윈도우 함수의 길이
            window (str): 윈도우 함수 종류 (예: 'hann')
        """
        super(STFT, self).__init__()
        # 기본 파라미터 설정
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        
        # FFT 기저 행렬 생성을 위한 스케일링
        scale = self.filter_length / self.hop_length
        # 단위 행렬에 FFT 적용하여 푸리에 기저 생성
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        # Nyquist frequency 까지만 사용 (대칭성)
        cutoff = int((self.filter_length / 2 + 1))
        # 실수부와 허수부를 분리하여 쌓기
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        # 순방향과 역방향 변환을 위한 기저 행렬 생성
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        # 윈도우 함수 적용
        if window is not None:
            assert(filter_length >= win_length)
            # 윈도우 함수 생성 및 중앙 패딩
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # 기저 행렬에 윈도우 적용
            forward_basis *= fft_window
            inverse_basis *= fft_window

        # 모델 버퍼에 기저 행렬 등록
        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """
        시간 도메인 신호를 주파수 도메인으로 변환
        Args:
            input_data: 입력 오디오 신호 [배치, 샘플]
        Returns:
            magnitude: 스펙트로그램 크기
            phase: 위상 정보
        """
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        self.num_samples = num_samples

        # librosa 스타일의 reflect 패딩 적용
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        # 컨볼루션으로 STFT 수행
        forward_transform = F.conv1d(
            input_data.cuda(),
            Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0).cpu()

        # 실수부와 허수부 분리
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # 크기와 위상 계산
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        """
        주파수 도메인 신호를 시간 도메인으로 역변환
        Args:
            magnitude: 스펙트로그램 크기
            phase: 위상 정보
        Returns:
            inverse_transform: 복원된 시간 도메인 신호
        """
        # 크기와 위상을 실수부와 허수부로 재결합
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        # 역컨볼루션으로 시간 도메인 신호 복원
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        # 윈도우 함수 효과 보정
        if self.window is not None:
            # 윈도우 함수의 제곱합 계산
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # 0에 가까운 값 처리를 위한 인덱스 찾기
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            
            # 윈도우 효과 제거
            inverse_transform[:, :,
                              approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # 홉 길이에 따른 스케일링
            inverse_transform *= float(self.filter_length) / self.hop_length

        # 패딩 제거
        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:,
                                              :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        """
        PyTorch 모듈의 forward 메서드
        STFT 변환 후 즉시 역변환하여 신호 재구성
        Args:
            input_data: 입력 오디오 신호
        Returns:
            reconstruction: 재구성된 오디오 신호
        """
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length,
                 n_mel_channels, sampling_rate, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)
         
        return mel_output, energy
