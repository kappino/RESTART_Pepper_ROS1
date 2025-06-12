#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
from care_er_ave.msg import audioVideo, audio_video_eeg
import re
import subprocess
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_AV_MODEL_DIR = os.path.join(CURRENT_DIR, "ER_models/eeg_av/")
PATH_MODEL_AUDIO_VIDEO = os.path.join(EEG_AV_MODEL_DIR, "audio_video_emotion_recognition_model/")
PATH_MODEL_EEG = os.path.join(EEG_AV_MODEL_DIR, "EEG_model/")
PATH_META_MODEL = os.path.join(EEG_AV_MODEL_DIR, "Meta_model/")

'''PATH_MODEL_AUDIO_VIDEO = "/home/CARE-ER-AVE/audio_video_emotion_recognition_model/"
PATH_MODEL_EEG = "/home/CARE-ER-AVE/EEG_model/"
PATH_META_MODEL = "/home/CARE-ER-AVE/Meta_model/"'''

class emotion_recognition():
    def __init__(self):
        rospy.init_node('Emotion_recognition', anonymous=True)
        rospy.Subscriber('av', audioVideo, self.run_model_audio_video)
        rospy.Subscriber('eeg', String, self.run_model_eeg)
        rospy.Subscriber("av_eeg", audio_video_eeg, self.run_meta_model)
        self.pub_emotion = rospy.Publisher('emotion', String, queue_size=10, latch=True)

        rospy.loginfo("Node Emotion recognition running")
        rospy.sleep(2)
    
    def run_model(self, path_model, cwd):
        try:
            process = subprocess.Popen(
            args= path_model,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd = cwd
            )
            stdout, stderr = process.communicate()
            match = re.search(r'Emotion:\s*(\w+)', stdout)
            if match:
                emotion = match.group(1)
                return emotion
        except subprocess.CalledProcessError as e:
            print("Errore nell'esecuzione del modello:")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            return None


    def run_model_audio_video(self, data):
        video_path = data.video_path
        audio_path = data.audio_path
        cwd = PATH_MODEL_AUDIO_VIDEO
        path_model = ['conda', 'run', '-n', 'myenv', 'python', "main.py", "--no_train", "--no_val", "--predict", "--video_file_path", video_path, "--audio_file_path", audio_path]
        emotion = self.run_model(path_model=path_model, cwd=cwd)
        if emotion:
            print(emotion)
            self.pub_emotion.publish(emotion)

    def run_model_eeg(self, data):
        eeg_path = data.data
        cwd = PATH_MODEL_EEG
        path_model = ['conda', 'run', '-n', 'myenv', 'python', "main.py", "--no_train", "--no_val", "--predict", "--eeg_data", eeg_path]
        emotion = self.run_model(path_model=path_model, cwd=cwd)
        if emotion:
            print(emotion)
            self.pub_emotion.publish(emotion)

    def run_meta_model(self, data):
        eeg_path = data.eeg
        video_path = data.video_path
        audio_path = data.audio_path
        cwd = PATH_META_MODEL
        path_model = ['conda', 'run', '-n', 'myenv', 'python', "main.py", "--no_train", "--no_val", "--predict", "--eeg_data", eeg_path, "--video_file_path", video_path, "--audio_file_path", audio_path]
        emotion = self.run_model(path_model=path_model, cwd=cwd)
        if emotion:
            print(emotion)
            self.pub_emotion.publish(emotion)

if __name__ == '__main__':
    er = emotion_recognition()
    rospy.spin()