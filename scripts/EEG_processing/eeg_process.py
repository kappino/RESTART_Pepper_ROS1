
import argparse
import os
import time
from datetime import datetime
import pandas as pd
from kalman_filter import KalmanFilter
from emotiv_streamer import EmotivStreamer

SHARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_dir/")
EMOTIV_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                   'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

class EEG_process():
    def __init__(self, fs=256, window=3, debug=True, orientation="horizontal"):
        self.fs = fs
        self.window = window
        self.debug = debug
        self.orientation = orientation
        self.emotiv = EmotivStreamer(debug_mode=self.debug)
        self.kalman_x, self.kalman_y = KalmanFilter(), KalmanFilter()
        if not self.emotiv.connect():
            exit(1)

    def save_data_vertical(self, packets):
        # Estrae solo i dati EEG da ogni pacchetto
        records = [dict(zip(EMOTIV_CHANNELS, packet['eeg'])) for packet in packets]
        return pd.DataFrame(records)

    def save_data_horizontal(self,packets):
        # Crea una matrice dove ogni riga è un canale EEG e ogni colonna un timestamp
        eeg_matrix = [packet['eeg'] for packet in packets]
        df = pd.DataFrame(eeg_matrix).transpose()
        df.insert(0, "Channel", EMOTIV_CHANNELS)  # Aggiungi i nomi dei canali come prima colonna
        df.columns = ["Channel"] + [f"Time_{i+1}" for i in range(df.shape[1] - 1)]  # Rinomina le colonne
        return df


    def save_data_loop(self, index=0):
        os.makedirs(SHARED_DIR, exist_ok=True)

        packets_to_save = self.emotiv.data_store.copy()
        #filename = f"eeg_gyro_{datetime.now():%Y%m%d_%H%M%S}.csv"
        filename = f"eeg_{index}.csv"
        filepath = os.path.join(SHARED_DIR, filename)

        if self.orientation == "vertical":
            df = self.save_data_vertical(packets_to_save)
        elif self.orientation == "horizontal":
            df = self.save_data_horizontal(packets_to_save)
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        df.to_csv(filepath, index=False)
        print(f"PATH: {filepath}\n")

        self.emotiv.data_store.clear()
        return(filepath)


    def data_reader_loop(self):

        # Calcola il numero di sample da acquisire per la finestra desiderata
        num_samples = self.fs * self.window

        samples_collected = 0
        while samples_collected < num_samples:
            packet = self.emotiv.read_packet()
            if packet is None:
                continue

            self.emotiv.data_store.append(packet)
            samples_collected += 1

    def process_eeg_data(self, index=0):
        self.data_reader_loop()
        return self.save_data_loop(index=index)


        

