import torch
import os
from torcheeg.models import FBCCNN
from eeg_input_preprocess import preprocess_data

NUM_CLASSES = 4
IN_CHANNELS=4
GRID_SIZE=(9,9)
BEST_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/eeg_best_state.pth")

LOGITS_TO_LABEL = {
    0: "Neutral",
    1: "Happy",
    2: "Angry",
    3: "Sad"
}

class eeg_model():
    def __init__(self):
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.model = FBCCNN(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS, grid_size=GRID_SIZE).to(self.device)
        best_state = torch.load(BEST_MODEL, map_location=torch.device(self.device))
        self.model.load_state_dict(best_state['state_dict'])
        self.model.eval()

    def predict(self, eeg_data_path):    
        input_data = preprocess_data(eeg_data_path).to(self.device)
        input_data = input_data.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_data)
        max_value, max_index = torch.max(output, dim=1)
        emotion = LOGITS_TO_LABEL[max_index.item()]
        return emotion