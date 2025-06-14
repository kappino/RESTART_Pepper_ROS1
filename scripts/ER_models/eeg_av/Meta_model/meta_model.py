import random
import numpy as np
import torch 
from EmotionStackingClassifier import EmotionStackingClassifier
from audio_video_model import audio_video_model
from eeg_model import eeg_model



class meta_model():
    def __init__(self):
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        

