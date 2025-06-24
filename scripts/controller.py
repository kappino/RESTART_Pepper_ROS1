#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool
from pepper import Pepper
import yaml
import os
import random

#IP = "169.254.115.62"
IP = "host.docker.internal"
#IP = "localhost"
PORT = 9559
BEHAVIOUR_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behavior_rules.yaml")
'''map_emotion = {
    "neutral": "neutral",
    "happy" : "positive",
    "angry" : "negative",
    "sad" : "negative"
}'''
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
        #self.pub_terapia_attiva = rospy.Publisher('terapia_attiva', Bool, queue_size=10, latch=True)
        rospy.sleep(2)
        self.start_welcome_pepper()
        rospy.Subscriber('performance', String, self.start_action)
        rospy.loginfo("Node controller is online")

    def start_welcome_pepper(self):
        while True:
            print("\nPremi 1 per eseguire accoglienza e ROT")
            print("Altrimenti premi 0 per uscire")
            
            scelta = input("La tua scelta: ")
            
            if scelta == '1':
                from pepper_welcome import pepperWelcome
                pepper_welcome = pepperWelcome(self.session_robot)
                pepper_welcome.welcome()
                pepper_welcome.ROT()
            elif scelta == '0':
                break
            else:
                print("Scelta non valida, riprova")

    def start_action(self, data):
        print("performance ricevuto")
        performance = data.data
        #msg = rospy.wait_for_message("emotion", String, timeout=None)
        #emotion = msg.data
        emotion = random.choice(emotions) #ONLY FOR TESTING, IF YOU HAVE PEPPER PLS UNCOMMENT THE TWO LINES ABOVE
        #print(self.behavior[emotion][performance])
        config = {
            'bodyLanguageMode': 'contextual',
        }
        number = random.randint(0, 2)
        self.session_robot.set_eye_color(self.behavior[emotion][performance]['eye_color'])
        self.session_robot.pepper_animated_say(self.behavior[emotion][performance]['phrases'][number],config)
    
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



        