import warnings
warnings.filterwarnings("ignore")

import mediapipe as mp
import cv2
import numpy as np
from fer import FER
import random




class pose_image_er():
    def __init__(self):
        #Initialization FER for Emotion recognition
        self.detector = FER()
        #Initialization for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose()

    def read_image(self, image):
        self.img = cv2.imread(image, cv2.IMREAD_COLOR)
        #print(self.img.shape)

    def read_video(self, video_path, num_frame=3):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Errore nell'apertura del video.")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = sorted(random.sample(range(1, total_frames - 1), num_frame))
        
        frame_list = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame)
            else:
                print(f"Errore nel leggere il frame {frame_idx}")
                frame_list.append(None)

        cap.release()
        return frame_list



    def classify_emotion(self, img=None):
        if img is not None:
            self.img = img
        #print("DETECT EMOTION")
        emotion, emotion_score = self.detection_emotion()
        if emotion is None:
            emotion = "Unknown"
            emotion_score = 0.0
        #print("DETECT POSTURE")
        posture_landmarks = self.detect_posture_landmarks()
        posture_type = self.classify_posture(posture_landmarks)
        # Calcoliamo il punteggio finale dell'emozione con la logica di fusione
        final_emotion_score = self.adjust_emotion_based_on_posture(emotion, emotion_score, posture_type)
        sentiment_score = self.analyze_sentiment(emotion)  # Analizza il sentiment dell'emozione
        # Applica il sentiment al punteggio finale
        final_emotion_score_with_sentiment = self.apply_sentiment_to_emotion(final_emotion_score, sentiment_score)
        return emotion, final_emotion_score, posture_type, final_emotion_score_with_sentiment
    
    def detection_emotion(self):
        try:
            if self.img is None:
                raise ValueError("There is no image")
            emotion, score = self.detector.top_emotion(self.img)
            return emotion, score
        except ValueError as e:
            return str(e)

    def detect_posture_landmarks(self):
        try:
            if self.img is None:
                raise ValueError("There is no image")
            # Usa MediaPipe per rilevare i landmarks della postura
            frame_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(frame_rgb)
            return results.pose_landmarks
        except ValueError as e:
            return str(e)

    def classify_posture(self, landmarks):
        try:
            if self.img is None:
                raise ValueError("There is no image")
            if landmarks:
                return "open"
            return "neutral"
        except ValueError as e:
            return str(e)
    
    def adjust_emotion_based_on_posture(self, emotion, emotion_score, posture_type):
        # Logica di fusione dell'emozione basata sulla postura
        if emotion == "happy" and posture_type == "open":
            emotion_score *= 1.1
        elif emotion == "sad" and posture_type == "closed":
            emotion_score *= 1.1
        elif emotion == "sad" and posture_type == "open":
            emotion_score *= 0.7
        elif emotion == "neutral" and posture_type == "closed":
            emotion_score *= 1.2
        return emotion_score

    def analyze_sentiment(self, emotion):
        # Dummy sentiment analysis
        if emotion == "happy":
            sentiment_score = 0.9  # Positivo
        elif emotion == "sad":
            sentiment_score = -0.7  # Negativo
        else:
            sentiment_score = 0.0  # Neutrale
        return sentiment_score

    def apply_sentiment_to_emotion(self, final_emotion_score, sentiment_score):
        # Applica il punteggio di sentiment all'emozione finale
        return final_emotion_score + sentiment_score

    

