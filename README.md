
# RESTART PROJECT

Sistema per l'interazione adattiva con pazienti anziani basato su performance cognitive e stato emotivo.

---

## Indice

* [Obiettivo](#-obiettivo)
* [Architettura](#-architettura)
* [Specifiche Comportamentali](#-specifiche-comportamentali)
* [Flusso Operativo](#-flusso-operativo)
* [Interfacce ROS](#-interfacce-ros)
* [Implementazione](#-implementazione)
* [Testing](#-testing)
* [Roadmap](#-roadmap)

---

## Obiettivo

Implementare un controller ROS che:
1. Riceve in input:
   - Livello di performance (`BASSO/MEDIO/ALTO`) dal modulo giochi
   - Stato emotivo (`NEUTRO/POSITIVO/NEGATIVO`) dal modulo AI
2. Genera output comportamentali per Pepper secondo le specifiche dell'Allegato 1:
   - Frasi di rinforzo (pag. 11-13)
   - Parametri fisici (colore occhi, gesti, tono voce - pag. 14)
Implementare un game_manager che:
   - Permette di eseguire le terapie contenute direttamente sul robot
Implementare una classe di terapie
---

## Architettura

![Image](images/diagram_restart.drawio.svg)

---

## Specifiche Comportamentali

### Mappatura Input-Output

| Emozione  | Performance | Azione Robot                                                                 | Parametri Fisici                     |
|-----------|-------------|------------------------------------------------------------------------------|--------------------------------------|
| NEUTRO    | ALTO        | "Hai ottenuto un'ottima performance. Proviamo qualcosa di più sfidante?"     | Occhi azzurri, gesto approvante, tono 0.85-0.9 |
| POSITIVO  | BASSO       | "Il tuo atteggiamento è incoraggiante, i risultati verranno col tempo"       | Occhi verdi, braccia aperte, tono 0.85 |
| NEGATIVO  | MEDIO       | "Capisco che oggi non ti senti al massimo, ma stai lavorando bene"           | Occhi grigi, postura non minacciosa, tono 0.7-0.8 |

*Tabella completa disponibile nell'Allegato 1*

---

## Nodi ROS

Il sistema ROS contiene i seguenti nodi:
   -   Controller: è il nodo principale che permette di avere un sistema coeso tra ROS e ambiente esterno, e inoltre permette al robot di eseguire azioni specifiche.
   -   Pepper_av_get: nodo adibito alla registrazione audio e video del paziente e alla comunicazione del path sul quale è salvata la registrazione
   -   EEG_get: nodo adibito all'acquisizione di dati EEG tramite caschetto e alla comunicazione del path sul quale sono stati salvati questi dati
   -   Broker_synchronizer: nodo che sincronizza i due nodi prima descritti (Pepper_av_get e EEG_get) in modo che registrino nella stessa finestra temporale. Oltre a sincronizzare tramite topics specifici permette di capire quali dei due nodi è attivo (/av, /eeg e /av_eeg)
   -   Emotion_recognition: nodo che ha il compito di rilevare emozioni mappandole in tre possibili stati(NEUTRAL, POSITIVE e NEGATIVE)
  
### Messaggi Custom

**audioVideo.msg**
```
string video_path
string audio_path
```

**audio_video_eeg.msg**
```
string video_path
string audio_path
string eeg
```

**eeg.msg**
```
string eeg
```

---

## Flusso Operativo Controller

1. **Acquisizione Input**:
   - Ricezione performance via stringa sul topic "/perfomance" (valori: LOW, MEDIUM, HIGH)
   - Ricezione emozione via stringa sul topic "/emotion" (valore che corrisponde all'emozione dominante rilevata: NEUTRAL, POSITIVE, NEGATIVE)

2. **Valutazione del livello di performance**:
   ```python
   # Esempio di classificazione performance
   # Analizziamo le percentuali di successo per ottenere il livello di performance
   if success >= 80:
       level = "HIGH"
   elif success >= 20:
       level = "MEDIUM"
   else:
       level = "LOW"
   ```

3. **Generazione Comando**:
   La generazione del comando da far eseguire al robot avviene nel seguente modo:
      -   Il controller è un nodo ROS che è sempre attivo che ha il compito di elaborare i dati ricevuti.
      -   Il nodo di emotion recognition è anch’esso attivo continuamente e pubblica le emozioni riconosciute sul topic /emotion.
      -   Durante l’esecuzione della terapia, vengono pubblicati sul topic /performance dei valori che rappresentano la performance dell’utente.

      -   Quando il controller riceve un dato di performance, recupera l’ultima emozione disponibile sul topic /emotion.
Dall’incrocio di questi due parametri (emozione e performance), viene determinato un comando da inviare al robot.

   Il comando è selezionato da un file YAML che funge da mappa di comportamento e contiene le possibili combinazioni di emozione e performance. Ogni combinazione definisce:
   
      -   Una lista di frasi motivazionali o di supporto da pronunciare (con eventuali animazioni)
      -   Il colore degli occhi del robot
      -   Il tipo di gesto corporeo da eseguire
      -   L’intonazione della voce da utilizzare

---

## Implementazione

### Struttura Package
```
├── CMakeLists.txt
├── images
│   ├── ros_nodes.drawio
│   └── ros_nodes.pdf
├── __init__.py
├── msg
│   ├── audio_video_eeg.msg
│   ├── audioVideo.msg
│   └── eeg.msg
├── package.xml
├── README.md
├── scripts
│   ├── behavior_rules.yaml
│   ├── broker_av_eeg.py
│   ├── controller.py
│   ├── eeg_get.py
│   ├── EEG_processing
│   │   ├── eeg_process.py
│   │   ├── emotiv_streamer.py
│   │   ├── kalman_filter.py
│   │   └── shared_dir
│   ├── emotion_recognition.py
│   ├── ER_models
│   │   ├── av_er_restart
│   │   │   ├── audio_er.py
│   │   │   ├── av_model.py
│   │   │   ├── example.jpg
│   │   │   ├── extract_frames.py
│   │   │   ├── main.py
│   │   │   ├── pose_image_er.py
│   │   │   └── test.wav
│   │   └── eeg_av
│   │       ├── audio_video_model.py
│   │       ├── audio_video_preprocessing.py
│   │       ├── dummy_eeg.csv
│   │       ├── eeg_input_preprocess.py
│   │       ├── eeg_model.py
│   │       ├── EmotionStackingClassifier.py
│   │       ├── envs
│   │       │   ├── environment_linux.yml
│   │       │   └── environment_windows.yml
│   │       ├── images
│   │       │   └── architecture_overview.jpg
│   │       ├── meta_model.py
│   │       ├── Multimodal_transformer
│   │       │   ├── MultimodalTransformer.py
│   │       │   ├── Preprocessing_CNN
│   │       │   │   ├── Audio_preprocessing.py
│   │       │   │   ├── Preprocessing_utils
│   │       │   │   │   ├── efficientface.py
│   │       │   │   │   └── modulator.py
│   │       │   │   └── Video_preprocessing.py
│   │       │   └── Transformers
│   │       │       └── Transformer_funcs.py
│   │       ├── utils_av
│   │       │   ├── average_meter.py
│   │       │   ├── logger.py
│   │       │   ├── precision.py
│   │       │   └── transforms.py
│   │       └── weights
│   │           ├── Complete_model.pth
│   │           ├── eeg_best_state.pth
│   │           ├── RAVDESS_multimodalcnn_15_best_cpu.pth
│   │           └── RAVDESS_multimodalcnn_15_best.pth
│   ├── game_base.py
│   ├── game_manager.py
│   ├── memory_game.py
│   ├── pepper_av_get.py
│   ├── pepper.py
│   ├── pepper.pyc
│   ├── pepper_welcome.py
│   └── test_av_eeg.py
├── srv
│   └── Check.srv
└── utils
    └── write_csv.py
```

---

## Note Tecniche

1. ### Requisiti

- ROS Noetic su Ubuntu 20.04
- Python 3.8
- Emotiv Epocx (EEG)
- Pepper Robot con NAOqi SDK
- [Anaconda](https://www.anaconda.com/) (per ambienti ML)

