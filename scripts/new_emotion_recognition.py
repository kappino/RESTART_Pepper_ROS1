#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
import sys
import os
META_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ER_models/eeg_av/")
MODEL_AV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ER_models/av_er_restart/")
sys.path.append(META_MODEL_PATH)
sys.path.append(MODEL_AV_PATH)
import rospy
from std_msgs.msg import String
from care_er_ave.msg import audioVideo, audio_video_eeg
from meta_model import meta_model
from av_model import av_model


MAP_EMOTION = {
    "neutral": "neutral",
    "happy" : "positive",
    "angry" : "negative",
    "sad" : "negative"
}


class emotion_recognition():
    def __init__(self, salerno=False):
        rospy.init_node('Emotion_recognition', anonymous=True)
        self.salerno = salerno
        if not self.salerno:
            self.meta_model = meta_model()
            rospy.Subscriber('av', audioVideo, self.run_model_audio_video)
            rospy.Subscriber('eeg', String, self.run_model_eeg)
            rospy.Subscriber("av_eeg", audio_video_eeg, self.run_meta_model)
        if self.salerno:
            self.av_model = av_model()
            rospy.Subscriber('av', audioVideo, self.run_model_audio_video)

        self.pub_emotion = rospy.Publisher('emotion', String, queue_size=10, latch=True)
        rospy.loginfo("Node Emotion recognition running")
        rospy.sleep(2)

    def run_model_audio_video(self, data):
        video_path = data.video_path
        audio_path = data.audio_path
        if self.salerno:
            emotion = self.av_model.emotion_recognition(video_path, audio_path)
        else:
            emotion = self.meta_model.av.predict(video_path, audio_path)
            emotion = MAP_EMOTION[emotion.lower()]
        if emotion:
            print("AV: ", emotion)
            self.pub_emotion.publish(emotion)

    def run_model_eeg(self,data):
         eeg_path = data.data
         emotion = self.meta_model.eeg.predict(eeg_path)
         emotion = MAP_EMOTION[emotion.lower()]
         print("EEG: ", emotion)
         self.pub_emotion.publish(emotion)

    def run_meta_model(self, data):
        eeg_path = data.eeg
        video_path = data.video_path
        audio_path = data.audio_path
        emotion = self.meta_model.predict(video_path, eeg_path, audio_path)
        emotion = MAP_EMOTION[emotion.lower()]
        if emotion:
            print("META_MODEL: ", emotion)
            self.pub_emotion.publish(emotion)


if __name__ == '__main__':
    er_node = emotion_recognition(salerno=False)
    rospy.spin()






