#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import rospy
from game_base import BaseGame
from pepper import Pepper
import time

class MemoryGame(BaseGame):
    def __init__(self, name_game, pepper):
        super().__init__(name_game)
        self.pepper = pepper  # istanza Pepper già connessa
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
    
    def start_say_words(self):
        word = None
        timeout = 60  # secondi massimi per parola
        repeated_words = []
        start_time = time.time()
        while time.time() - start_time < timeout:
            data = self.pepper.memory.getData("WordRecognized")
            if data and isinstance(data, list) and len(data) >= 2:
                recognized_word = data[0]
                confidence = data[1]
                if confidence > 0.4:
                    word = recognized_word.lower()
                    if word not in repeated_words:
                        repeated_words.append(recognized_word)
                        print(f"Riconosciuto: {word} (confidenza: {confidence:.2f})")
                    else:
                        self.pepper.say(f"Questa parola {word} già l'hai detta")
                else:
                    self.pepper.say(f"Questa parola {word} non è corretta")
        return repeated_words


    def start(self):
        self.pepper.pepper_say("Benvenuto al gioco di memoria!")
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
        number_words = 5 + level - 1

        instructions = (
            "Il test inizierà ora. Dirò alcune parole. "
            "Cerca di ricordarne il più possibile. "
            "Poi ti chiederò di ripeterle, una alla volta."
        )

        print(instructions)
        self.pepper.pepper_say(instructions)
        rospy.sleep(2)

        words_to_say = random.sample(self.words_list, number_words)
        for word in words_to_say:
            print(word)
            self.pepper.pepper_say(word)
            rospy.sleep(1.5)

        print("\nRipetimi una parola alla volta:")
        #Start short term memory
        self.pepper.asr_subscribe("MemoriaGioco", "Italian", words_to_say, True)
        repeated_words_short_term_memory = self.start_say_words()
        self.success_short_term_memory = self.calculate_success(repeated_words=repeated_words_short_term_memory, number_words=number_words)
        self.performance_short_term_memory = self.calculate_performance(self.success_short_term_memory)
        
        #Start drawing
        self.pepper.asr_unsubscribe("MemoriaGioco")
        self.pepper.say("Divertiti disegnando")
        rospy.sleep(300)

        #Start long term memory
        self.pepper.asr_subscribe("MemoriaGioco", "Italian", words_to_say, True)
        repeated_words_long_term_memory = self.start_say_words()
        self.success_long_term_memory = self.calculate_success(repeated_words=repeated_words_long_term_memory, number_words=number_words)
        self.performance_long_term_memory = self.calculate_performance(self.success_long_term_memory)
        self.pepper.asr_unsubscribe("MemoriaGioco")

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
        self.pepper.pepper_say("Hai completato il gioco!")
        self.end()
