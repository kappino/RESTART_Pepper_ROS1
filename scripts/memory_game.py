#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import rospy
from game_base import BaseGame
from pepper import Pepper

class MemoryGame(BaseGame):
    def __init__(self, nome, pepper):
        super().__init__(nome)
        self.pepper = pepper  # istanza Pepper già connessa
        self.words_list = [
            "cane", "luna", "telefono", "bicicletta", "montagna",
            "finestra", "mare", "computer", "fiore", "tavolo",
            "libro", "scarpa", "palla", "albero", "forchetta"
        ]
        rospy.loginfo("MemoryGame inizializzato.")

    def calculate_successo(self, parole_ripetute, parole_da_ripetere):
        correct = sum(1 for parola in parole_ripetute if parola in parole_da_ripetere)
        self.successo = (float(correct) / float(len(parole_da_ripetere))) * 100

    def start(self):
        self.pepper.pepper_say("Benvenuto al gioco di memoria!")
        rospy.sleep(1)

        while True:
            try:
                livello = int(input("Seleziona il livello di difficoltà (1-6): "))
                if 1 <= livello <= 6:
                    break
                else:
                    print("Inserisci un numero tra 1 e 6.")
            except ValueError:
                print("Input non valido. Inserisci un numero.")

        self.livello = livello
        numero_parole = 5 + livello - 1

        istruzioni = (
            "Il test inizierà ora. Dirò alcune parole. "
            "Cerca di ricordarne il più possibile. "
            "Poi ti chiederò di ripeterle, una alla volta."
        )

        print(istruzioni)
        self.pepper.pepper_say(istruzioni)
        rospy.sleep(2)

        words_to_say = random.sample(self.words_list, numero_parole)
        for word in words_to_say:
            print(word)
            self.pepper.pepper_say(word)
            rospy.sleep(1.5)

        parole_ripetute = []
        print("\nRipetimi una parola alla volta:")
        # Avvia il riconoscimento
        asr.subscribe("MemoriaGioco")
        asr.setLanguage("Italian")
        asr.setVocabulary(words_to_say, True)  # solo parole previste, no parola nuova
        parola = None
        timeout = 10  # secondi massimi per parola

        start_time = time.time()
        while time.time() - start_time < timeout:
            data = memory.getData("WordRecognized")
            if data and isinstance(data, list) and len(data) >= 2:
                recognized_word = data[0]
                confidence = data[1]
                if confidence > 0.4:
                    parola = recognized_word.lower()
                    print(f"Riconosciuto: {parola} (confidenza: {confidence:.2f})")
                    break
            rospy.sleep(0.5)

        asr.unsubscribe("MemoriaGioco")

        if parola:
            parole_ripetute.append(parola)
        else:
            print("Nessuna parola riconosciuta.")
            parole_ripetute.append("")

        for i in range(len(words_to_say)):
            parola = input("{i + 1}: ").strip().lower()
            parole_ripetute.append(parola)

        self.calculate_successo(parole_ripetute, words_to_say)
        self.calculate_performance()

        print("\n--- Risultati ---")
        print("Parole mostrate: {words_to_say}")
        print("Parole ripetute: {parole_ripetute}")
        print("Punteggio di successo: {self.successo:.2f}%")
        print("Performance: {self.performance}")
        self.pepper.pepper_say("Hai completato il gioco!")
        self.end()
