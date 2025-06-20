#!/usr/bin/env python                   
# -*- coding: utf-8 -*-                                        
                                                                                                       
import rospy
import csv                                                                                                        
import sys                  
import time                                                        
import random 
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_msgs.msg import Bool
from datetime import date
from datetime import datetime,timedelta

# Globals variables

today = datetime.today()

currrent_day = today.weekday()

current_month = today.month - 1  # Mese corrente (0-11, quindi sottraiamo 1)

day = today.day

# Colors
red = 0xFF0000
green = 0x00FF00
blue = 0x0000FF
white = 0xFFFFFF
light_blue = 0xB0E0E6
grey = 0x444444

port = 9559

#setting vocabulary to pass to every questions

statement_vocables = ["si","no"]
companion_options = [
    "nessuno",
    "moglie",
    "marito",
    "figlio",
    "nipote",
    "un amico",
    "l'autista",
    "l'infermiere",
    "badante",
    "sorella",
    "fratello",
    "mamma",
    "papà",
    "madre",
    "padre"
]

day_vocables = ["lunedi","martedi","mercoledi","giovedi","venerdi","sabato","domenica"]

months_vocables = ["gennaio","febbraio","marzo","aprile","maggio","giugno","luglio","agosto","settembre","ottobre","novembre","dicembre"]

current_month_string = months_vocables[current_month]  # Nome del mese corrente

season_vocables = ["estate","primavera","autunno","inverno"]

holydays_months = {
    "gennaio": ["Capodanno", "Epifania"],
    "febbraio": ["Carnevale"],  # Data mobile
    "marzo": ["Festa della donna", "Festa del papà"],  # Pasqua può cadere a marzo
    "aprile": ["Pasqua", "Lunedì dell'Angelo", "Anniversario della Liberazione"],
    "maggio": ["Festa dei lavoratori", "Festa della mamma"],
    "giugno": ["Festa della Repubblica", "Corpus Domini"],
    "luglio": ["Festa della Madonna del Carmine"],  # Locale ma sentita in molte città
    "agosto": ["Ferragosto"],
    "settembre": ["Festa di San Gennaro", "Inizio scuola"],
    "ottobre": ["Festa dei nonni", "Halloween"],
    "novembre": ["Ognissanti", "Commemorazione dei defunti"],
    "dicembre": ["Immacolata Concezione", "Natale", "Santo Stefano", "San Silvestro"]
}

holydays_months_vocables = holydays_months.get(current_month_string,[])

class pepperWelcome:
    def __init__(self,pepper):
        self.pepper = pepper

    def save_file_csv(self, file_name, question, answer):
        with open("root/catkin_ws/config/"+file_name, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer])

    def get_pazient_name(self):
        name = input("Insert pazient name: ").strip().lower()
        surname = input("Insert pazient surname: ").strip().lower()

        pazient_name = name + "_" + surname
        return pazient_name
    
    import time

    def question_and_answer(self, question, options):
        self.pepper.pepper_say(question)

        attempts = 2
        answer = None

        for attempt in range(attempts):
            if attempt == 1:
                self.pepper.pepper_say("Non ho capito. Puoi ripetere?")

            # Configura il riconoscimento vocale
            self.pepper.asr.pause(True)
            self.pepper.asr.setLanguage("Italian")
            self.pepper.asr.setVocabulary(options, False)  # False = permette parole non nel vocabolario
            self.pepper.asr.pause(False)

            self.pepper.asr.subscribe("SessioneCognitiva")
            self.pepper.memory.raiseEvent("WordRecognized", [])

            print(f"[INFO] In ascolto per: {options} (tentativo {attempt + 1})")

            start_time = time.time()
            while time.time() - start_time < 10:
                data = self.pepper.memory.getData("WordRecognized")
                if data and isinstance(data, list) and len(data) >= 2 and data[1] > 0.5:
                    risposta = data[0]
                    print(f"[UTENTE] {risposta}")
                    break
                time.sleep(0.2)

            self.pepper.asr.unsubscribe("SessioneCognitiva")

            if answer:
                self.pepper.pepper_say("Hai detto " + answer)
                break

        if not answer:
            self.pepper.pepper_say("Non ho capito neanche stavolta.")
            answer = "nessuna"

        self.pepper.memory.raiseEvent("WordRecognized", [])  # reset dell'evento
        return answer

    def current_season(self):

        if (current_month == 12 and day >= 21) or current_month in [1, 2] or (current_month == 3 and day < 21):
            return "inverno"
        elif (current_month == 3 and day >= 21) or current_month in [4, 5] or (current_month == 6 and day < 21):
            return "primavera"
        elif (current_month == 6 and day >= 21) or current_month in [7, 8] or (current_month == 9 and day < 23):
            return "estate"
        elif (current_month == 9 and day >= 23) or current_month in [10, 11] or (current_month == 12 and day < 21):
            return "autunno"

    def welcome(self):

        self.pazient_name = self.get_pazient_name()

        self.save_file_csv(self.pazient_name, "Accoglienza", "Inizio: " + time.strftime("%H:%M:%S"))

        self.pepper.pepper_say("Ciao io sono Pepper")
        self.pepper.set_eye_color(green)
        time.sleep(2)

        start_time = time.time()
        answer1 = self.question_and_answer("Ti senti stanco?", statement_vocables)
        self.save_file_csv(self.pazient_name, "Ti senti stanco?", answer1)
        time.sleep(2)
        
        answer2 = self.question_and_answer("Hai dormito stanotte?", statement_vocables)
        self.save_file_csv(self.pazient_name, "Hai dormito stanotte?", answer2)
        time.sleep(2)

        answer3 = self.question_and_answer("Chi ti ha accompagnato oggi?", companion_options)
        self.save_file_csv(self.pazient_name, "Chi ti ha accompagnato oggi?", answer3)
        time.sleep(2)

        self.set_eye_color(white)
        self.save_file_csv(self.pazient_name, "Accoglienza", "Fine: " + time.strftime("%H:%M:%S"))

    def ROT(self):

        self.save_file_csv(self.pazient_name, "ROT", "Inizio: " + time.strftime("%H:%M:%S"))
        self.pepper.set_eye_color(blue)

        answer1 = self.question_and_answer("Che giorno e oggi?",day_vocables)
        self.save_file_csv(self.pazient_name, "Che giorno e oggi?", answer1)
        time.sleep(2)

        if answer1 in day_vocables and answer1 != day_vocables[currrent_day]:
            yesterday = (today - timedelta(days=1)).weekday()
            answer1 = self.question_and_answer("non e' esattamente " +answer1+ ", ieri era  "+ day_vocables[yesterday] + " quindi oggi che giorno e'",day_vocables)
            self.save_file_csv(self.pazient_name, "Che giorno e oggi?", answer1)
            time.sleep(2)

        answer2 = self.question_and_answer("In che mese siamo?", months_vocables)
        self.save_file_csv(self.pazient_name, "In che mese siamo?", answer2)
        time.sleep(2)

        if answer2 in months_vocables and answer2 != months_vocables[current_month]:
            last_month = current_month - 1
            answer2 = self.question_and_answer("non e' esattamente " +answer2+ "il mese precedente era " + months_vocables[last_month] + "quindi oggi in che mese siamo?",months_vocables)
            self.save_file_csv(self.pazient_name, "In che mese siamo?", answer2)
            time.sleep(2)

        answer3 = self.question_and_answer("In che stagione ci troviamo?", season_vocables)
        self.save_file_csv(self.pazient_name, "In che stagione ci troviamo?", answer3)
        time.sleep(2)

        season = self.current_season()
        if answer3 in season_vocables and answer3 != season:
            if season == "inverno":
                answer3 = self.question_and_answer("rifletti, fa molto freddo quindi in che stagione siamo?",season_vocables)
                self.save_file_csv(self.pazient_name, "In che stagione ci troviamo?", answer3)
                time.sleep(2)

            elif season == "primavera":
                answer3 = self.question_and_answer("rifletti, cominciano a sbocciare i fiori quindi in che stagione siamo?",season_vocables)
                self.save_file_csv(self.pazient_name, "In che stagione ci troviamo?", answer3)
                time.sleep(2)

            elif season == "estate":
                answer3 = self.question_and_answer("rifletti, si inizia ad andare a mare, quindi in che stagione siamo?",season_vocables)
                self.save_file_csv(self.pazient_name, "In che stagione ci troviamo?", answer3)
                time.sleep(2)

            else:
                answer3 = self.question_and_answer("rifletti, cominciano a cadere le foglie dagli alberi, quindi in che stagione siamo?",vocaboli_stagioni)
                self.save_file_csv(self.pazient_name, "In che stagione ci troviamo?", answer3)
                time.sleep(2)

        answer4 = self.question_and_answer("Hai fatto colazione questa mattina?", statement_vocables)
        self.save_file_csv(self.pazient_name, "Hai fatto colazione questa mattina?", answer4)
        time.sleep(2)

        answer5 = self.question_and_answer("Cosa si festeggia questo mese?",holydays_months_vocables)
        self.save_file_csv(self.pazient_name, "Cosa si festeggia questo mese?", answer5)
        time.sleep(2)

        self.pepper.set_eye_color(white)
        self.save_file_csv(self.pazient_name, "ROT", "Fine: " + time.strftime("%H:%M:%S"))