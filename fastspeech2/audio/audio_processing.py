import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import utils.hparams as hp


def window_sumsquare(window, n_frames, hop_length=hp.hop_length, win_length=hp.win_length, n_fft=hp.filter_length, dtype=np.float32, norm=None):
    """
    STFT에서 윈도우 함수에 의해 발생하는 변조 효과를 추정하는데 사용됩니다.

    # from librosa >= 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """

    if win_length is None:
        win_length = n_fft

    # 전체 신호 길이 계산
    n = n_fft + hop_length * (n_frames - 1)
    # 결과를 저장할 배열 초기화
    x = np.zeros(n, dtype=dtype)

    # 윈도우 함수 생성 및 처리
    win_sq = get_window(window, win_length, fftbins=False)  # 윈도우 함수 생성
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2   # 정규화 후 제곱
    win_sq = librosa_util.pad_center(win_sq, n_fft)        # n_fft 길이에 맞게 패딩

    # 각 프레임에 대해 윈도우 적용
    for i in range(n_frames):
        sample = hop_length * i  # 현재 프레임의 시작 위치

        # 윈도우 함수를 현재 프레임의 시작 위치에 더함
        x[sample:min(n, sample + n_fft)] += win_sq[:min(n_fft, n - sample)]

    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C