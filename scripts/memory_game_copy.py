#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import rospy
from game_base_copy import BaseGame
from std_msgs.msg import String, Bool
from RESTART_Pepper_ROS1.msg import Event
import time

class MemoryGame(BaseGame):
    def __init__(self, name_game):
        super().__init__(name_game)

        self.words_list = [
            "cane", "luna", "telefono", "bicicletta", "montagna",
            "finestra", "mare", "computer", "fiore", "tavolo",
            "libro", "scarpa", "palla", "albero", "forchetta"
        ]
        rospy.loginfo("Initialized MemoryGame.")

    def calculate_success(self, repeated_words, number_words):
        correct = sum(1 for word in repeated_words)
        success = (float(correct) / float(number_words)) * 100
        return success
    
    def start(self):
        self.pub_event.publish(Event(type="say", args="Benvenuto al gioco di memoria!"))
        #self.pepper.pepper_say("Benvenuto al gioco di memoria!")
        self.pub_start_emotion.publish(True)
        rospy.sleep(1)

        while True:
            try:
                level = int(input("Seleziona il livello di difficoltà (1-6): "))
                if 1 <= level <= 6:
                    break
                else:
                    print("Inserisci un numero tra 1 e 6.")
            except ValueError:
                print("Input non valido. Inserisci un numero.")

        self.level = level
        number_words = 1 + level - 1

        instructions = (
            "Il test inizierà ora. Dirò alcune parole. "
            "Cerca di ricordarne il più possibile. "
            "Poi ti chiederò di ripeterle, una alla volta."
        )

        print(instructions)
        self.pub_event.publish(Event(type="say", args=instructions))
        #self.pepper.pepper_say(instructions)
        rospy.sleep(10)

        words_to_say = random.sample(self.words_list, number_words)
        for word in words_to_say:
            print(word)
            self.pub_event.publish(Event(type="say", args=word))
            #self.pepper.pepper_say(word)
            rospy.sleep(1.5)

        print("\nRipetimi una parola alla volta:")
        #Start short term memory
        msg = Event()
        msg.type = "asr"
        msg.args = json.dumps({
            "GameName": "MemoriaGioco",
            "Language": "Italian",
            "Vocabulary": words_to_say,
            "Flag": True
        })
        self.pub_event.publish(msg)
        #self.pepper.asr_subscribe("MemoriaGioco", "Italian", words_to_say, True)
        #repeated_words_short_term_memory = self.start_say_words(quantity=number_words)
        wait_msg = rospy.wait_for_message("asr", String, timeout=None)
        repeated_words_short_term_memory = json.loads(wait_msg.data)

        self.success_short_term_memory = self.calculate_success(repeated_words=repeated_words_short_term_memory, number_words=number_words)
        print(self.success_short_term_memory)
        self.performance_short_term_memory = self.calculate_performance(self.success_short_term_memory)
        
        #Start drawing
        #self.pepper.asr_unsubscribe("MemoriaGioco")
        self.pub_event.publish(Event(type="say", args="Ora disegna le parole che ti ho detto."))
        #self.pepper.pepper_say("Adesso prenditi una pausa e divertiti disegnando")
        rospy.sleep(3)
        
        #self.pepper.pepper_say("Adesso ripetimi le parole che ti ho elencato prima")
        self.pub_event.publish(Event(type="say", args="Adesso ripetimi le parole che ti ho elencato prima"))
        rospy.sleep(2)

        #Start long term memory
        msg = Event()
        msg.type = "asr"
        msg.args = json.dumps({
            "GameName": "MemoriaGioco",
            "Language": "Italian",
            "Vocabulary": words_to_say,
            "Flag": True
        })
        self.pub_event.publish(msg)
        
        #self.pepper.asr_subscribe("MemoriaGioco", "Italian", words_to_say, True)
        #repeated_words_short_term_memory = self.start_say_words(quantity=number_words)
        wait_msg = rospy.wait_for_message("asr", String, timeout=None)
        repeated_words_long_term_memory = json.loads(wait_msg.data)


        #self.pepper.asr_subscribe("MemoriaGioco", "Italian", words_to_say, True)
        #repeated_words_long_term_memory = self.start_say_words(quantity=number_words)
        self.success_long_term_memory = self.calculate_success(repeated_words=repeated_words_long_term_memory, number_words=number_words)
        self.performance_long_term_memory = self.calculate_performance(self.success_long_term_memory)
        #self.pepper.asr_unsubscribe("MemoriaGioco")

        #Print results
        print("\n--- Risultati ---")
        print(f"Parole mostrate: {words_to_say}")

        print("Memoria Breve Termine: ")
        print(f"Parole ripetute: {repeated_words_short_term_memory}")
        print(f"Punteggio di successo: {self.success_short_term_memory:.2f}%")
        print(f"Performance: {self.performance_short_term_memory}")

        print("Memoria Lungo Termine: ")
        print(f"Parole ripetute: {repeated_words_long_term_memory}")
        print(f"Punteggio di successo: {self.success_long_term_memory:.2f}%")
        print(f"Performance: {self.performance_long_term_memory}")
        #self.pepper.pepper_say("Hai completato il gioco!")
        self.pub_event.publish(Event(type="say", args="Hai completato il gioco!"))

        self.end()
