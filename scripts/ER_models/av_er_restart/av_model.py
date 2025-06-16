import warnings
warnings.filterwarnings("ignore")

from pose_image_er import pose_image_er
from audio_er import audio_er
from extract_frames import estrai_frame_casuali
from collections import Counter
import os




emotion_to_label = {
  'angry': "negative", 'disgust': "negative", 'fear': "negative", 
  'happy': "positive", 'neutral': "neutral", 'sad': "negative", 'surprise': "positive"}

class av_model():
    def __init__(self):
        self.pose_image = pose_image_er()
        self.audioer = audio_er()

    def emotion_recognition(self, video_path, audio_path):
        list_frames = self.pose_image.read_video(video_path)
        emotions = []
        for frame in list_frames:
            emotion, final_emotion_score, posture_type, final_emotion_score_with_sentiment = self.pose_image.classify_emotion(frame)
            emotions.append(emotion_to_label[emotion.lower()])
        sentiment = self.audioer.analyze_sentiment(audio_path)
        print(sentiment)
        emotions.append(sentiment[0]["label"].lower())
        counts = Counter(emotions)
        return counts.most_common(1)[0][0]

        
