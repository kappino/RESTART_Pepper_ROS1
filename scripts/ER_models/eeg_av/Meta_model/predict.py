import sys
sys.path.append('../EEG_model')
sys.path.append('../audio_video_emotion_recognition_model')

from datasets.seediv_dataset import generate_dataset_SEEDIV
from Data_preprocessing import input_preprocessing_predict
import torch
from torch.utils.data import DataLoader
from data_preprocessing.eeg_input_preprocess import preprocess_data


dict_label = {
    0: "Neutral", 
    1: "Happy", 
    2: "Angry",
    3: "Sad"
}
def predict_testing(opts, stacking_classifier):
    
    #eeg_dataset = generate_dataset_SEEDIV(opts.path_eeg, opts.path_cached)

    # Load weights
    weights = torch.load('results/Complete_model.pth')
    stacking_classifier.meta_model = weights['meta_model_state_dict']
    stacking_classifier.eval()

    # Input preprocessing
    video_file_path = "../audio_video_emotion_recognition_model/raw_video_files/happy_wrong_1.mp4"
    video_file_path = opts.video_file_path
    audio_file_path = None
    if opts.audio_file_path:
        audio_file_path = opts.audio_file_path
        audio_var, video_var = input_preprocessing_predict.preprocessing_audio_video(
            video_file_path,
            audio_file_path,
            video_norm_value=opts.video_norm_value,
            batch_size=1
        )
    else:
        audio_var, video_var = input_preprocessing_predict.preprocessing_audio_video(
            video_file_path,
            video_norm_value=opts.video_norm_value,
            batch_size=1
        )

    # Load EEG data
    '''loader_eeg = DataLoader(
        eeg_dataset, 
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )'''

    '''# Organize EEG data
    print("Reached data synchronization...")
    labels = stacking_classifier.organize_by_labels(loader_eeg)

    # Select a random EEG sample deterministically
    sample_data = (((labels[1])[0]).unsqueeze(0))
    sample_data = sample_data.to(opts.device)'''

    input_data_eeg = preprocess_data(opts.eeg_data).to(opts.device)
    input_data_eeg = input_data_eeg.unsqueeze(0)

    # Perform predictions
    print("Initializing prediction step")
    with torch.no_grad():
        eeg_predict = stacking_classifier.model2(input_data_eeg)
        av_predict = stacking_classifier.model1(audio_var, video_var)
        final_prediction = stacking_classifier.forward(audio_var,video_var,input_data_eeg)
    max_value_eeg, max_index_eeg = torch.max(eeg_predict, dim=1)  # Se vuoi il massimo lungo una certa dimensione
    max_value_av, max_index_av = torch.max(av_predict, dim=1)
    emotion = dict_label[final_prediction[0]]
    # Output predictions
    print(f"EEG prediction: {eeg_predict}, Emotion: {dict_label[max_index_eeg.item()]}")
    print(f"AV prediction: {av_predict}, Emotion: {dict_label[max_index_av.item()]}")
    print(f"Final prediction: {final_prediction}: {dict_label[final_prediction[0]]}")
    print(f"Emotion: {emotion}\n")
    
    
