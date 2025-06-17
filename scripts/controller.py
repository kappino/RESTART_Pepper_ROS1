#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool
from pepper import Pepper
import yaml
import os

#IP = "169.254.115.62"
IP = "host.docker.internal"
#IP = "localhost"
PORT = 9559
BEHAVIOUR_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behavior_rules.yaml")
map_emotion = {
    "neutral": "neutral",
    "happy" : "positive",
    "angry" : "negative",
    "sad" : "negative"
}

performances = ["HIGH", "MEDIUM", "LOW"]
emotions = ["NEUTRAL", "POSITIVE", "NEGATIVE"]

class controller():
    def __init__(self):
        rospy.init_node('Controller_Pepper', anonymous=True)
        self.session_robot = Pepper.create(IP, PORT)
        with open(BEHAVIOUR_RULES, 'r') as file:
            self.behavior = yaml.safe_load(file)
        if self.session_robot is None:
            exit("Pepper is not online")
            #print("Pepper is not online")
        # ------------------
        # DEBUG: Abilita il test delle gestures decommentando la riga seguente.
        # Questo farà eseguire tutte le combinazioni di emozioni e performance
        # per testare le frasi e i gesti associati.
        # self.try_all_gestures()
        # ------------------
        self.pub_terapia_attiva = rospy.Publisher('terapia_attiva', Bool, queue_size=10, latch=True)
        rospy.sleep(2)
        rospy.Subscriber('performance', String, self.start_action)
        rospy.loginfo("Node controller is online")

    
    def start_action(self, data):
        print("performance ricevuto")
        performance = data.data
        self.pub_terapia_attiva.publish(True)
        #msg = rospy.wait_for_message("emotion", String, timeout=None)
        #emotion = msg.data
        emotion = "NEUTRAL"  
        #print(self.behavior[emotion][performance])
        config = {
            'bodyLanguageMode': 'contextual',
        }
        self.session_robot.pepper_animated_say(self.behavior[emotion][performance]['phrases'][0],config)
    
    def try_all_gestures(self):
        for emotion in emotions:
            for performance in performances:
                print("Trying gesture for emotion: {emotion}, performance: {performance}")
                config = {
                    'bodyLanguageMode': 'contextual',
                }
                for phrase in self.behavior[emotion][performance]['phrases']:
                    print("Phrase: {phrase}")
                    self.session_robot.pepper_animated_say(phrase, config)
        print("All gestures tried successfully.")
    


    



if __name__ == '__main__':
    cont = controller()
    rospy.spin()



        