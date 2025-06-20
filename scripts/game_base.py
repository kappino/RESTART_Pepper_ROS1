#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from pepper import Pepper


class BaseGame:
    def __init__(self, name_game):
        self.name_game = name_game
        self.level = 1  # livello iniziale
        self.success = 0  # percentuale
        self.last_level = 1  # ultimo livello raggiunto
        self.max_level = 6
        self.min_level = 1
        self.performance = "medio"
        self.behaviour_pub = None
        self.game_pub = None
        self.pepper = Pepper.create("DA INSERIRE", 9559)
        self.pub_performance = rospy.Publisher('performance', String, queue_size=10)

    def update_level(self):
        if self.success >= 80:
            self.level = min(self.level + 1, self.max_level)
        elif 20 < self.success < 80:
            pass  # livello invariato
        else:
            self.level = max(self.level - 1, self.min_level)
        self.last_level = self.level

    def reset(self):
        self.level = 1
        self.success = 0
        self.last_level = 1
        self.performance = "MEDIUM"

    def calculate_performance(self, success):
        if success >= 80:
            performance = "HIGH"
        elif 20 < self.success < 80:
            performance = "MEDIUM"
        else:
            performance = "LOW"
        self.pub_performance(performance)
        return performance


    def end(self):
        rospy.loginfo(f"[{self.name_game}] Fine del gioco.")
