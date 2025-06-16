import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
import speech_recognition as sr
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"





class audio_er():
    def __init__(self):
        # Carica il modello di analisi del sentiment
        self.sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Funzione di analisi sentiment
    def analyze_sentiment(self, audio_file):
        # Trascrizione dell'audio in testo
        transcript = self.transcribe_audio(audio_file)  # Funzione che converte l'audio in testo
        
        # Analisi sentimentale
        sentiment = self.sentiment_pipeline(transcript)
        return sentiment

    def transcribe_audio(self, audio_file):
        # Crea un oggetto Recognizer
        recognizer = sr.Recognizer()
        
        # Carica il file audio
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)  # Legge tutto l'audio del file
        
        try:
            # Usa Google Web Speech API per trascrivere l'audio, specificando la lingua italiana
            #print("Inizio trascrizione...")
            transcription = recognizer.recognize_google(audio, language="it-IT")  # Specifica lingua italiana
            #print("Trascrizione completata: ", transcription)
            return transcription
        
        except sr.UnknownValueError:
            #print("Google Speech Recognition non ha capito l'audio")
            return ""
        except sr.RequestError as e:
            #print("Errore nel servizio di Google Speech Recognition; {0}".format(e))
            return ""
