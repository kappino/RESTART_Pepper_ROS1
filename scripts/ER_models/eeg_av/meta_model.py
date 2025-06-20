import random
import numpy as np
import torch 
import os
from EmotionStackingClassifier import EmotionStackingClassifier
from audio_video_model import audio_video_model
from eeg_model import eeg_model
from audio_video_preprocessing import preprocessing_audio_video
from eeg_input_preprocess import preprocess_data

PATH_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/Complete_model.pth")
VIDEO_NORM_VALUE = 255

LOGITS_TO_LABEL = {
    0: "Neutral",
    1: "Happy",
    2: "Angry",
    3: "Sad"
}
class meta_model():
    def __init__(self):
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"

        self.av = audio_video_model()
        self.eeg = eeg_model()

        self.meta_model = EmotionStackingClassifier(
            model1=self.av.model,
            model2=self.eeg.model,
            batch_size= 8,
            max_iter=1000
        )

        weights = torch.load(PATH_WEIGHTS)
        self.meta_model.meta_model = weights['meta_model_state_dict']
        self.meta_model.eval()

    def predict(self, video_path, eeg_path, audio_path = None):
        video_path = video_path
        if audio_path:
            audio_path = audio_path
        audio_data, video_data = preprocessing_audio_video(video_path,audio_path,video_norm_value=VIDEO_NORM_VALUE, batch_size=1)
        if video_data is None:
            return None
        eeg_data = preprocess_data(eeg_path).to(self.device)
        eeg_data = eeg_data.unsqueeze(0)

        emotion = self.meta_model.forward(audio_data=audio_data, video_data=video_data, eeg_data=eeg_data)
        emotion = LOGITS_TO_LABEL[emotion[0]]
        return emotion

