import os
import torch
from torch import nn
from Multimodal_transformer.MultimodalTransformer import MultimodalTransformer
from audio_video_preprocessing import preprocessing_audio_video

N_CLASSES = 4
SAMPLE_DURATION = 15
NUM_HEADS = 1
BEST_MODEL_CPU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/RAVDESS_multimodalcnn_15_best_cpu.pth")
BEST_MODEL_GPU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/RAVDESS_multimodalcnn_15_best.pth")
VIDEO_NORM_VALUE = 255

LOGITS_TO_LABEL = {
    0: "Neutral",
    1: "Happy",
    2: "Angry",
    3: "Sad"
}

class audio_video_model():
    def __init__(self):
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.model = MultimodalTransformer(N_CLASSES, seq_length = SAMPLE_DURATION, pretr_ef=None, num_heads=NUM_HEADS)


        if self.device != 'cpu':
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model, device_ids=None)

        #load best state, there are two file pth for separating if the machine hase cuda or not
        if(self.device=="cuda"):
            best_state = torch.load(BEST_MODEL_GPU)
        else:
            best_state = torch.load(BEST_MODEL_CPU, map_location="cpu")
            
        #Load the weigths on the model
        self.model.load_state_dict(best_state['state_dict'])
        self.model.eval()

    def predict(self, video_file_path, audio_file_path=None):
        
        video_path = video_file_path
        if audio_file_path:
            audio_path = audio_file_path
        
        
        audio_var, video_var = preprocessing_audio_video(video_path,audio_path,video_norm_value=VIDEO_NORM_VALUE, batch_size=1)

        if video_var is None:
            return None
        
        with torch.no_grad():
            output = self.model(x_audio=audio_var, x_visual=video_var)
        max_value, max_index = torch.max(output, dim=1)
        emotion = LOGITS_TO_LABEL[max_index.item()]
        return(emotion)