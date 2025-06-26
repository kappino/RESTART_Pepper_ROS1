#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import json
from std_msgs.msg import String, Bool
from restart.msg import Event
from pepper_copy import Pepper
import yaml
import os
import random

IP = "169.254.250.162"
#IP = "host.docker.internal"
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
            exit("Robot is not online")
        # ------------------
        # DEBUG: Abilita il test delle gestures decommentando la riga seguente.
        # Questo farà eseguire tutte le combinazioni di emozioni e performance
        # per testare le frasi e i gesti associati.
        # self.try_all_gestures()
        # ------------------
        #self.pub_terapia_attiva = rospy.Publisher('terapia_attiva', Bool, queue_size=10, latch=True)
        rospy.sleep(2)
        self.session_robot.start_welcome()

        rospy.Subscriber('event', Event, self.handle_event)
        rospy.loginfo("Node controller is online")

    def handle_event(self, msg):
        if msg.type == "say":
            rospy.loginfo("Received say event: %s", msg.args)
            self.session_robot.pepper_say(msg.args)

        elif msg.type == "asr":
            try: 
                args = json.loads(msg.args)  # Converte la stringa in dizionario
                self.session_robot.asr_subscribe(
                    args.get('GameName'),
                    args.get('Language'),
                    args.get('Vocabulary'),
                    args.get('Flag', True)
                )
                #self.session_robot.asr_subscribe(args.get('GameName'),args.get('Language'),args.get('Vocabulary'),args.get('Flag', True))
                quantity = len(args.get('Vocabulary'))
                if quantity <= 0:
                    rospy.logwarn("Invalid quantity for ASR event: %s", quantity)
                    return
                repeated_words = self.session_robot.start_say_words(quantity=quantity)
                publisher = rospy.Publisher('asr', String, queue_size=10)
                rospy.sleep(1)
                publisher.publish(json.dumps(repeated_words))
                self.session_robot.asr_unsubscribe(args.get('GameName'))

            except json.JSONDecodeError:
                rospy.logerr("Failed to decode JSON from ASR event args: %s", msg.args)
                return
            
            rospy.loginfo("Received ASR event: %s", msg.args)

        elif msg.type == "performance":
            rospy.loginfo("Received performance event: %s", msg.args)
            performance = msg.args
            if performance not in performances:
                rospy.logwarn(f"Unknown performance level: {performance}")
                return
            # Start the action based on the received performance
            self.start_action(performance)
        else:
            rospy.logwarn(f"Unknown event type: {msg.type}")

    def start_action(self, data):
        print("performance ricevuto")
        performance = data
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



        