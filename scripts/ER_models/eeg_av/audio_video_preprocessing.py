import torch
import cv2
import numpy as np
import functools
from PIL import Image
import librosa
from torch.autograd import Variable
import librosa
import os
import numpy as np
from moviepy.editor import VideoFileClip
import torch
from facenet_pytorch import MTCNN
from utils import transforms
import random

video_extension = ["mp4", "avi", "mkv"]

class Audio_preprocessing():
    def __init__ (self, data_path):
        self.data_path = data_path
        self.target_time = 3
    
    def process(self):
        filename, file_extension = os.path.splitext(self.data_path)
        if file_extension in video_extension:
            video = VideoFileClip(self.video_path)
            audio = video.audio
            audio.write_audiofile("prova.wav")
            audios = librosa.core.load("./prova.wav", sr=22050, res_type="kaiser_fast")
            os.remove("prova.wav")
        else:
            audios = librosa.core.load(self.data_path, sr=22050, res_type="kaiser_fast")
        y = audios[0]
        sr = audios[1]
        target_length = int(sr * self.target_time)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        else:
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]

        return y,sr
    
select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
class Video_preprocessing():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    def __init__(self, video_path):
        self.video_path = video_path
        self.MTCNN = MTCNN(image_size=(720,1280), device=self.device)
        self.save_frames = 15
        self.input_fps = 30
        self.save_length = 3.6 #seconds
    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        #calculate length in frames
        framen = 0
        while True:
            i, _ = cap.read()
            if not i:
                break
            framen += 1
        cap = cv2.VideoCapture(self.video_path)

        if self.save_length*self.input_fps > framen:                    
            skip_begin = int((framen - (self.save_length*self.input_fps)) // 2)
            for i in range(skip_begin):
                _, im = cap.read() 
                    
        framen = int(self.save_length*self.input_fps)    
        frames_to_select = select_distributed(self.save_frames,framen)

        numpy_video = []
        frame_ctr = 0
            
        while True: 
            ret, im = cap.read()
            if not ret:
                break
            if frame_ctr not in frames_to_select:
                frame_ctr += 1
                continue
            else:
                frames_to_select.remove(frame_ctr)
                frame_ctr += 1

            temp = im[:,:,-1]
            im_rgb = im.copy()
            im_rgb[:,:,-1] = im_rgb[:,:,0]
            im_rgb[:,:,0] = temp
            im_rgb = torch.tensor(im_rgb)
            im_rgb = im_rgb.to(self.device)

            bbox = self.MTCNN.detect(im_rgb)
            if bbox[0] is not None:
                bbox = bbox[0][0]
                bbox = [round(x) for x in bbox]
                x1, y1, x2, y2 = bbox
            im = im[y1:y2, x1:x2, :]
            im = cv2.resize(im, (224,224))
            numpy_video.append(im)
        if len(frames_to_select) > 0:
            for i in range(len(frames_to_select)):
                numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
        return np.array(numpy_video)
    

def video_loader(video_dir_path):
    video = video_dir_path   
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def preprocess_frame(frame, input_size=(224, 224), video_norm_value=None):
    '''# Convert BGR (OpenCV) to RGB
    # Ridimensiona l'immagine
    # Trasformazioni (puoi aggiungere altre se necessarie)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte in tensor [C, H, W]
    ])'''
    frame = cv2.resize(frame, input_size)
    video_transform = transforms.Compose([
                transforms.ToTensor(video_norm_value)])
    
    return video_transform(frame)

def preprocessing_audio_video(video_path, audio_path=None, video_norm_value=None, batch_size=1):
    video_npy = Video_preprocessing(video_path).process()
    if audio_path is None:
        audio_npy,sr = Audio_preprocessing(video_path).process()
    else:
        audio_npy,sr = Audio_preprocessing(audio_path).process()
    loader = get_default_video_loader()
    visual_input_batch = loader(video_npy)
    #VIDEO
    video_transform = transforms.Compose([
                transforms.ToTensor(video_norm_value)])
    clip = [video_transform(img) for img in visual_input_batch]            
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    #AUDIO
    mfcc = get_mfccs(audio_npy, sr=sr)   
    audio_features = mfcc
    audio_var = torch.from_numpy(audio_features).float()
    
    
    clip = clip.unsqueeze(0).expand(batch_size, -1, -1,-1, -1)
    audio_var = audio_var.unsqueeze(0).expand(batch_size, -1, -1)
    with torch.no_grad():
        video_var = Variable(clip)
        audio_var = Variable(audio_var)
        video_var = video_var.permute(0,2,1,3,4)
        video_var = video_var.reshape(video_var.shape[0]*video_var.shape[1], video_var.shape[2], video_var.shape[3], video_var.shape[4])
        return audio_var,video_var
    
