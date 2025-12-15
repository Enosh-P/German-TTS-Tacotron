"""
Mel Spectrogram Processing for German TTS
"""
import librosa
import numpy as np
import torch

class MelProcessor:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
        ref_level_db=20,
        min_level_db=-100
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
    
    def load_wav(self, path):
        """Load and preprocess audio file"""
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav
    
    def wav_to_mel(self, wav):
        """Convert waveform to mel spectrogram"""
        #1. get mel spectogram, 2. convert to log scale, 3 . normalize it

        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        mel = librosa.power_to_db(mel, ref=1.0)
        mel = self._normalize(mel)
        
        return mel.T
    
    def _normalize(self, mel):
        """Normalize mel spectrogram to [-1, 1]"""
        mel = np.clip(mel, self.min_level_db, self.ref_level_db)
        mel = (mel - self.min_level_db) / (self.ref_level_db - self.min_level_db)
        mel = 2 * mel - 1
        return mel
    
    def _denormalize(self, mel):
        """Denormalize mel spectrogram"""
        mel = (mel + 1) / 2
        mel = mel * (self.ref_level_db - self.min_level_db) + self.min_level_db
        return mel
    
    def mel_to_wav(self, mel):
        """Convert mel spectrogram back to waveform (using Griffin-Lim)"""
        mel = self._denormalize(mel.T)
        mel = librosa.db_to_power(mel)
        
        # Griffin-Lim algorithm to convert mel to waveform
        wav = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        return wav


