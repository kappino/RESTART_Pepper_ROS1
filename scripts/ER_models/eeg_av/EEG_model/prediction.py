import torch
import os
from data_preprocessing.eeg_input_preprocess import preprocess_data
logits_to_label = {
    0: "Neutral",
    1: "Happy",
    2: "Angry",
    3: "Sad"
}
def predict(opt, model):
    model.eval()
    best_state = torch.load('results/best_state.pth', map_location=torch.device(opt.device))
    model.load_state_dict(best_state['state_dict'])
    input_data = preprocess_data(opt.eeg_data).to(opt.device)
    input_data = input_data.unsqueeze(0)
    with torch.no_grad():
        output = model(input_data)
    max_value, max_index = torch.max(output, dim=1)
    emotion = logits_to_label[max_index.item()]
    print(f"Emotion: {emotion}\n")
    
    
