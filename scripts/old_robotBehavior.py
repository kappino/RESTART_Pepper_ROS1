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
from pepper_speech.msg import ActionModeInt
from datetime import date
from naoqi import ALProxy
from datetime import datetime,timedelta
from emotions_class import PepperEmotionSpeaker
#----------------------------------------------------------VARIABILI GLOBALI--------------------------------------------------------
# Variabili globali

oggi = datetime.today()

giorno_corrente = oggi.weekday()

mese_corrente = oggi.month - 1  # Mese corrente (0-11, quindi sottraiamo 1)

terapy = False
robot_behaviour_sub = None
game_instruction_sub = None
nome_file = None
phrase_to_say = None

# Colori
rosso = 0xFF0000
verde = 0x00FF00
blu = 0x0000FF
bianco = 0xFFFFFF
azzurro = 0xB0E0E6
grigio = 0x444444

port = 9559

#settiamo i vocabolari da passare ad ogni domanda delle varie fasi

vocaboli_affermazione = ["si","no"]
opzioni_accompagnatore = [
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

vocaboli_giorni = ["lunedi","martedi","mercoledi","giovedi","venerdi","sabato","domenica"]

vocaboli_mesi = ["gennaio","febbraio","marzo","aprile","maggio","giugno","luglio","agosto","settembre","ottobre","novembre","dicembre"]

mese_corrente_stringa = vocaboli_mesi[mese_corrente]  # Nome del mese corrente

vocaboli_stagioni = ["estate","primavera","autunno","inverno"]

festivita_per_mese = {
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

vocaboli_festivita_mese = festivita_per_mese.get(mese_corrente_stringa,[])

#----------------------------------------------------------CONNESSIONE ROBOT--------------------------------------------------------
#Connessione con pepper e inizializzazione proxy tts
def connetti_pepper():
    while True:
        try:
            ip = raw_input("Inserisci l'indirizzo IP del robot Pepper: ").strip()
            print("Tentativo di connessione a Pepper su IP: {}".format(ip))
            tts = ALProxy("ALTextToSpeech", ip, port)
            tts.setLanguage("Italian")
            print("Connessione riuscita.")
            return ip
        except Exception as e:
            print("Connessione fallita. Errore: {}".format(e))
            print("Riprova.\n")
#----------------------------------------------------------FUNZIONE PER FARE DOMANDE E OTTENERE RISPOSTE----------------------------------------------------
def chiedi_e_rispondi(domanda, opzioni):
    tts.say(domanda)

    tentativi = 2
    risposta = None

    for tentativo in range(tentativi):
        if tentativo == 1:
            tts.say("Non ho capito. Puoi ripetere?")

        # Permetti qualsiasi risposta, non solo quelle nel vocabolario
        sr.pause(True)
        sr.setLanguage("Italian")
        sr.setVocabulary(opzioni, False)  # <--- False permette qualsiasi parola
        sr.pause(False)
        sr.subscribe("SessioneCognitiva")
        memory.raiseEvent("WordRecognized", [])
        print("[INFO] In ascolto per: {} (tentativo {})".format(opzioni, tentativo + 1))

        start_time = time.time()
        while time.time() - start_time < 10:
            data = memory.getData("WordRecognized")
            if data and data[1] > 0.5:
                risposta = data[0]
                print("[UTENTE] {}".format(risposta))
                break
            time.sleep(0.2)

        sr.unsubscribe("SessioneCognitiva")

        if risposta:
            tts.say("Hai detto " + risposta)
            break

    if not risposta:
        tts.say("Non ho capito neanche stavolta.")
        risposta = "nessuna"
    
    memory.raiseEvent("WordRecognized", [])
    return risposta

#----------------------------------------------------------FUNZIONE PER SALVARE LE RISPOSTE IN UN CSV--------------------------------------------------------
def salva_risposta_csv(nome_file, domanda, risposta):
    with open("root/catkin_ws/config/"+nome_file, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([domanda, risposta])
#----------------------------------------------------------FUNZIONE DETERMINA STAGIONE--------------------------------------------------------

def determina_stagione():
    oggi = datetime.today()
    giorno = oggi.day
    mese = oggi.month

    if (mese == 12 and giorno >= 21) or mese in [1, 2] or (mese == 3 and giorno < 21):
        return "inverno"
    elif (mese == 3 and giorno >= 21) or mese in [4, 5] or (mese == 6 and giorno < 21):
        return "primavera"
    elif (mese == 6 and giorno >= 21) or mese in [7, 8] or (mese == 9 and giorno < 23):
        return "estate"
    elif (mese == 9 and giorno >= 23) or mese in [10, 11] or (mese == 12 and giorno < 21):
        return "autunno"

#----------------------------------------------------------FUNZIONE COLORE OCCHI--------------------------------------------------------

# Funzione per cambiare il colore degli occhi
def cambia_colore_occhi(colore):
    nome_led_occhi = "FaceLeds"
    leds.fadeRGB(nome_led_occhi, colore, 1.0)

#----------------------------------------------------------FUNZIONE ACCOGLIENZA PAZIENTE--------------------------------------------------------

def accoglienza():

    salva_risposta_csv(nome_file, "Accoglienza", "Inizio: " + time.strftime("%H:%M:%S"))

    tts.say("Ciao io sono Pepper")
    cambia_colore_occhi(verde)
    time.sleep(2)

    start_time = time.time()
    risposta1 = chiedi_e_rispondi("Ti senti stanco?", vocaboli_affermazione)
    salva_risposta_csv(nome_file, "Ti senti stanco?", risposta1)
    time.sleep(2)
    
    risposta2 = chiedi_e_rispondi("Hai dormito stanotte?", vocaboli_affermazione)
    salva_risposta_csv(nome_file, "Hai dormito stanotte?", risposta2)
    time.sleep(2)

    risposta3 = chiedi_e_rispondi("Chi ti ha accompagnato oggi?", opzioni_accompagnatore)
    salva_risposta_csv(nome_file, "Chi ti ha accompagnato oggi?", risposta3)
    time.sleep(2)

    cambia_colore_occhi(bianco)
    salva_risposta_csv(nome_file, "Accoglienza", "Fine: " + time.strftime("%H:%M:%S"))

#----------------------------------------------------------FUNZIONE TERAPIA ORIENTATA ALLA REALTA'--------------------------------------------------------

def ROT():
    salva_risposta_csv(nome_file, "ROT", "Inizio: " + time.strftime("%H:%M:%S"))
    cambia_colore_occhi(blu)

    risposta1 = chiedi_e_rispondi("Che giorno e oggi?",vocaboli_giorni)
    salva_risposta_csv(nome_file, "Che giorno e oggi?", risposta1)
    time.sleep(2)

    if risposta1 in vocaboli_giorni and risposta1 != vocaboli_giorni[giorno_corrente]:
        ieri = oggi - timedelta(days=1)
        giorno_ieri = ieri.weekday()
        risposta1 = chiedi_e_rispondi("non e' esattamente " +risposta1+ ", ieri era  "+ vocaboli_giorni[giorno_ieri] + " quindi oggi che giorno e'",vocaboli_giorni)
        salva_risposta_csv(nome_file, "Che giorno e oggi?", risposta1)
        time.sleep(2)

    risposta2 = chiedi_e_rispondi("In che mese siamo?", vocaboli_mesi)
    salva_risposta_csv(nome_file, "In che mese siamo?", risposta2)
    time.sleep(2)

    if risposta2 in vocaboli_mesi and risposta2 != vocaboli_mesi[mese_corrente]:
        mese_precedente = mese_corrente - 1
        risposta2 = chiedi_e_rispondi("non e' esattamente " +risposta2+ "il mese precedente era " + vocaboli_mesi[mese_precedente] + "quindi oggi in che mese siamo?",vocaboli_mesi)
        salva_risposta_csv(nome_file, "In che mese siamo?", risposta2)
        time.sleep(2)

    risposta3 = chiedi_e_rispondi("In che stagione ci troviamo?", vocaboli_stagioni)
    salva_risposta_csv(nome_file, "In che stagione ci troviamo?", risposta3)
    time.sleep(2)

    stagione_corrente = determina_stagione()
    if risposta3 in vocaboli_stagioni and risposta3 != stagione_corrente:
        if stagione_corrente == "inverno":
            risposta3 = chiedi_e_rispondi("rifletti, fa molto freddo quindi in che stagione siamo?",vocaboli_stagioni)
            salva_risposta_csv(nome_file, "In che stagione ci troviamo?", risposta3)
            time.sleep(2)

        elif stagione_corrente == "primavera":
            risposta3 = chiedi_e_rispondi("rifletti, cominciano a sbocciare i fiori quindi in che stagione siamo?",vocaboli_stagioni)
            salva_risposta_csv(nome_file, "In che stagione ci troviamo?", risposta3)
            time.sleep(2)

        elif stagione_corrente == "estate":
            risposta3 = chiedi_e_rispondi("rifletti, si inizia ad andare a mare, quindi in che stagione siamo?",vocaboli_stagioni)
            salva_risposta_csv(nome_file, "In che stagione ci troviamo?", risposta3)
            time.sleep(2)

        else:
            risposta3 = chiedi_e_rispondi("rifletti, cominciano a cadere le foglie dagli alberi, quindi in che stagione siamo?",vocaboli_stagioni)
            salva_risposta_csv(nome_file, "In che stagione ci troviamo?", risposta3)
            time.sleep(2)

    risposta4 = chiedi_e_rispondi("Hai fatto colazione questa mattina?", vocaboli_affermazione)
    salva_risposta_csv(nome_file, "Hai fatto colazione questa mattina?", risposta4)
    time.sleep(2)

    risposta5 = chiedi_e_rispondi("Cosa si festeggia questo mese?",vocaboli_festivita_mese)
    salva_risposta_csv(nome_file, "Cosa si festeggia questo mese?", risposta5)
    time.sleep(2)

    cambia_colore_occhi(bianco)
    salva_risposta_csv(nome_file, "ROT", "Fine: " + time.strftime("%H:%M:%S"))
#----------------------------------------------------------FUNZIONE FRASE ISTRUZIONI--------------------------------------------------------

def phrase_to_say_command_callback(data):
    print ("Ho ricevuto la frase da dire: " + str(data.data))
    global phrase_to_say
    phrase_to_say = data.data

#----------------------------------------------------------FUNZIONE SPIEGAZIONE ISTRUZIONI--------------------------------------------------------

def game_instruction(phrase):
    tts.say("adesso ti spiegherò il funzionamento del test: ")
    config = {"bodyLanguageMode": "contextual"}
    text = "^start(animations/Stand/Gestures/Explain_1) {} ^wait(animations/Stand/Gestures/Explain_1)".format(phrase)
    animated_speech.say(text, config)

#----------------------------------------------------------FUNZIONE TERAPIA ATTIVA--------------------------------------------------------
def terapy_command_callback(data):
    global terapy, robot_behaviour_sub, game_instruction_sub
    new_terapy = data.data
    print ("terapia attiva: " + str(new_terapy))
    if new_terapy and not terapy:
        # Attiva i subscriber
        robot_behaviour_sub = rospy.Subscriber('/robot_behaviour', ActionModeInt, modality_command_callback)
        game_instruction_sub = rospy.Subscriber('/game_instruction', String, phrase_to_say_command_callback)
        print ("Subscriber attivi")
    elif not new_terapy and terapy:
        # Disattiva i subscriber
        if robot_behaviour_sub is not None:
            robot_behaviour_sub.unregister()
            robot_behaviour_sub = None
        if game_instruction_sub is not None:
            game_instruction_sub.unregister()
            game_instruction_sub = None
        print ("Subscriber disattivi")
    terapy = new_terapy
#----------------------------------------------------------FUNZIONE MODALITA' E COMANDI PEPPER--------------------------------------------------------
def modality_command_callback(data):
    command = data.command.data
    modality = data.modality.data

    print ("sono dentro " + str(type(command)) + str(type(modality))) 
    command_completed_pub = rospy.Publisher('/command_completed', Int32, queue_size=10)

    if modality == 1:
        if command == 1:
    	    #greetings
            cambia_colore_occhi(verde)
            config = {"bodyLanguageMode": "contextual"}
            rospy.loginfo("Sono in com 1")
            text = "^start(animations/Stand/Gestures/Salute) Ciao, Sono Pepper! Tieniti pronto per iniziare!^wait(animations/Stand/Gestures/Salute)"           
            animated_speech.say(text, config)
            command_completed_pub.publish(1)

        elif command == 2:
            #Instructing
            max_ripetizioni = 3
            tentativo = 0
            risposta = "no"

            while tentativo < max_ripetizioni and risposta == "no":
                cambia_colore_occhi(verde)
                game_instruction(phrase_to_say)
                tts.say("Hai capito le istruzioni del test?")

            	#Qui chiami la tua funzione di ascolto per ottenere la risposta
                risposta = chiedi_e_rispondi("Hai capito le istruzioni del test?", vocaboli_affermazione)
                tentativo += 1

            if (risposta == "si"):
            	tts.say("Perfetto! Procediamo con il test.")
            else:
                tts.say("Va bene, passiamo comunque alla fase successiva.")
            phrase_to_say = None
            command_completed_pub.publish(2)

        elif command == 3:
            #suggesting
	        command_completed_pub.publish(3)

        elif command == 4:
            #feedbacking
      	    command_completed_pub.publish(4)

        elif command == 5:
	        #closing
	        cambia_colore_occhi(rosso)
            tts.say("Il test è concluso. Grazie per la tua partecipazione!")
            command_completed_pub.publish(5)

        elif command == 6:
	        #accoglienza
	        accoglienza()

        elif command == 7:
	        #ROT
	        ROT()

#----------------------------------------------------------FUNZIONE RICAVA NOME DEL PAZIENTE---------------------------------
def get_pazient_name():
    name = raw_input("Insert pazient name: ").strip()
    surname = raw_input("Insert pazient surname: ").strip()

    pazient_name = name + "_" + surname
    return pazient_name

#----------------------------------------------------------FUNZIONE CALLBACK PER LA MODALITA' DI COMPORTAMENTO--------------------------------------------------------
def behaviour_mode_callback(data):
    global behaviour_mode
    behaviour_mode = data.data
    print ("Modalità di comportamento ricevuta: " + str(behaviour_mode))
    speaker.say("Modalità di comportamento impostata a: " + str(behaviour_mode), behaviour_mode)
#----------------------------------------------------------FUNZIONE SUBSCRIBER DEL NODO ROS--------------------------------------------------------
def robot_behaviour_subscriber():
    # Inizializza il nodo Subscriber
    rospy.init_node('robot_behaviour_subscriber', anonymous=True)
    # Inizializza il subscriber per il topic /terapia_attiva
    rospy.Subscriber('/terapia_attiva', Bool, terapy_command_callback)
    rospy.Subscriber('/behaviour_mode', String, behaviour_mode_callback)
    rospy.spin()
#----------------------------------------------------------MAIN--------------------------------------------------------

if __name__ == "__main__":
    print (mese_corrente)
    pepper_ip = connetti_pepper()
    # Inizializza nome paziente
    pazient_name = get_pazient_name()
    # Inizializza le interfacce del robot
    motion = ALProxy("ALMotion", pepper_ip, port)
    leds = ALProxy("ALLeds", pepper_ip, port)
    animated_speech = ALProxy("ALAnimatedSpeech", pepper_ip, port)
    sr = ALProxy("ALSpeechRecognition", pepper_ip, port)
    memory = ALProxy("ALMemory",pepper_ip,port)
    tts = ALProxy("ALTextToSpeech", pepper_ip, port)
    sr.pause(True)
    sr.setLanguage("Italian")
    sr.pause(False)
    # Inizializza il file CSV
    nome_file = "risposte_" + str(pazient_name) + "_" + str(oggi) + ".csv"
    speaker = PepperEmotionSpeaker(tts, animated_speech, leds)
    # Inizio Programma
    robot_behaviour_subscriber()
