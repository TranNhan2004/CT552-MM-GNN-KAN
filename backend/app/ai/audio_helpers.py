from pathlib import Path
from typing import List
import librosa
import numpy as np

class AudioHelpers:
    @staticmethod
    def extract_mfcc_3d(file_path: Path, sr=22050, n_mfcc=13) -> np.ndarray:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.stack([mfcc, delta, delta2], axis=0)  
        return features
    
    @staticmethod
    def pad_to_max_shape(arrays: List[np.ndarray]) -> np.ndarray:
        max_shape = np.max([a.shape for a in arrays], axis=0)
        padded = np.zeros((len(arrays),) + tuple(max_shape), dtype=np.float32)
        for i, a in enumerate(arrays):
            slices = tuple(slice(0, s) for s in a.shape)
            padded[i][slices] = a
        return padded