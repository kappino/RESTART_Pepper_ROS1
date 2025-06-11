#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool
from pepper import Pepper
import yaml
import os

IP = "169.254.115.62"
PORT = 9559
BEHAVIOUR_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behavior_rules.yaml")
map_emotion = {
    "neutral": "neutral",
    "happy" : "positive",
    "angry" : "negative",
    "sad" : "negative"
}

class controller():
    def __init__(self):
        rospy.init_node('Controller_Pepper', anonymous=True)
        self.session_robot = Pepper.create(IP, PORT)
        with open(BEHAVIOUR_RULES, 'r') as file:
            self.behavior = yaml.safe_load(file)
        if self.session_robot is None:
            #exit("Pepper is not online")
            print("Pepper is not online")
        self.pub_terapia_attiva = rospy.Publisher('terapia_attiva', Bool, queue_size=10, latch=True)
        rospy.sleep(2)
        rospy.Subscriber('performance', String, self.start_action)
        rospy.loginfo("Node controller is online")

    
    def start_action(self, data):
        print("performance ricevuto")
        performance = data.data
        self.pub_terapia_attiva.publish(True)
        msg = rospy.wait_for_message("emotion", String, timeout=None)
        emotion = msg.data
        print(self.behavior[emotion][performance])

    


    



if __name__ == '__main__':
    cont = controller()
    rospy.spin()



        