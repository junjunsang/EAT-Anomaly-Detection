import torchaudio.transforms as T
import torchaudio.functional as F

def mel_spectrogram(waveform, sample_rate, n_fft=1024, hop_length=512, n_mels=80):
    """
    입력 오디오(waveform)를 멜스펙트로그램 dB 스케일로 변환
    Args:
        waveform (Tensor): (channel, time)
        sample_rate (int): 샘플링 레이트
        n_fft (int): FFT 윈도우 크기 (default=1024)
        hop_length (int): hop 크기 (default=512)
        n_mels (int): 멜 밴드 개수 (default=80)
    Returns:
        mel_db (Tensor): (n_mels, time)
    """
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    mel_db = F.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0)
    return mel_db
