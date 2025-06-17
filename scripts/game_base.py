#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from pepper_speech.msg import ActionModeInt


class BaseGame:
    def __init__(self, nome):
        self.nome = nome
        self.livello = 1  # livello iniziale
        self.successo = 0  # percentuale
        self.ultimo_livello = 1  # ultimo livello raggiunto
        self.max_livello = 6
        self.min_livello = 1
        self.performance = "medio"
        self.behaviour_pub = None
        self.game_pub = None

    def aggiorna_livello(self):
        if self.successo >= 80:
            self.livello = min(self.livello + 1, self.max_livello)
        elif 20 < self.successo < 80:
            pass  # livello invariato
        else:
            self.livello = max(self.livello - 1, self.min_livello)

        self.ultimo_livello = self.livello

    def reset(self):
        self.livello = 1
        self.successo = 0
        self.ultimo_livello = 1
        self.performance = "medio"

    def start(self):
        raise NotImplementedError("Il metodo 'start' deve essere implementato nelle sottoclassi.")

    def calculate_successo(self):
        raise NotImplementedError("Il metodo 'calculate_successo' deve essere implementato nelle sottoclassi.")

    def calculate_performance(self):
        if self.successo >= 80:
            self.performance = "alto"
        elif 20 < self.successo < 80:
            self.performance = "medio"
        else:
            self.performance = "basso"

    def get_performance(self):
        return self.performance

    def publish_robot_behaviour(self, command, modality):
        msg = ActionModeInt()
        msg.command = int(command)
        msg.modality = modality

        rospy.loginfo("[{self.nome}] Invio comportamento robot - Modality: {modality} - Command: {command.name} ({int(command)})")
        self.behaviour_pub.publish(msg)

    def publish_game_instruction(self, phrase):
        rospy.loginfo("[{self.nome}] Invio istruzione di gioco: {phrase}")
        self.game_pub.publish(phrase)

    def end(self):
        rospy.loginfo("[{self.nome}] Fine del gioco.")
