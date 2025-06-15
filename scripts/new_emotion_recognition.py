#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./ER_models/eeg_av") 
import rospy
from std_msgs.msg import String
from care_er_ave.msg import audioVideo, audio_video_eeg
import re
import subprocess
import os
from meta_model import meta_model

class emotion_recognition():
    def __init__(self):
        self.meta_model = meta_model()

        rospy.init_node('Emotion_recognition', anonymous=True)
        rospy.Subscriber('av', audioVideo, self.run_model_audio_video)
        rospy.Subscriber('eeg', String, self.run_model_eeg)
        rospy.Subscriber("av_eeg", audio_video_eeg, self.run_meta_model)
        self.pub_emotion = rospy.Publisher('emotion', String, queue_size=10, latch=True)

        rospy.loginfo("Node Emotion recognition running")
        rospy.sleep(2)

    def run_model_audio_video(self, data):
        video_path = data.video_path
        audio_path = data.audio_path
        emotion = self.meta_model.av.predict(video_path, audio_path)
        if emotion:
            self.pub_emotion.publish(emotion)

    def run_model_eeg(self,data):
         eeg_path = data.data
         emotion = self.eeg.predict(eeg_path)
         self.pub_emotion.publish(emotion)

    def run_meta_model(self, data):
        eeg_path = data.eeg
        video_path = data.video_path
        audio_path = data.audio_path
        emotion = self.meta_model.predict(video_path, eeg_path, audio_path)
        if emotion:
            self.pub_emotion.publish(emotion)


if __name__ == '__main__':
    er_node = emotion_recognition()
    rospy.spin()






