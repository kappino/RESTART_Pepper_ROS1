
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


'''def save_data_vertical(packets):
    records = []
    for packet in packets:
        record = {
            'timestamp': packet.get('timestamp'),
            'counter': packet.get('counter'),
            'gyro_x': packet.get('gyro_x'),
            'gyro_y': packet.get('gyro_y'),
            'battery': packet.get('battery'),
        }
        record.update({ch: val for ch, val in zip(EMOTIV_CHANNELS, packet['eeg'])})
        records.append(record)
    return pd.DataFrame(records)'''

def save_data_vertical(packets):
    # Estrae solo i dati EEG da ogni pacchetto
    records = [dict(zip(EMOTIV_CHANNELS, packet['eeg'])) for packet in packets]
    return pd.DataFrame(records)


'''def save_data_horizontal(packets):
    eeg_matrix = [packet['eeg'] for packet in packets]
    df = pd.DataFrame(eeg_matrix).transpose()
    df.insert(0, "Channel", EMOTIV_CHANNELS)
    df.columns = ["Channel"] + [str(i + 1) for i in range(df.shape[1] - 1)]
    return df'''


def save_data_horizontal(packets):
    # Crea una matrice dove ogni riga è un canale EEG e ogni colonna un timestamp
    eeg_matrix = [packet['eeg'] for packet in packets]
    df = pd.DataFrame(eeg_matrix).transpose()
    df.insert(0, "Channel", EMOTIV_CHANNELS)  # Aggiungi i nomi dei canali come prima colonna
    df.columns = ["Channel"] + [f"Time_{i+1}" for i in range(df.shape[1] - 1)]  # Rinomina le colonne
    return df


def save_data_loop(data_store, orientation="vertical",index=0):
    os.makedirs(SHARED_DIR, exist_ok=True)

    packets_to_save = data_store.copy()
    #filename = f"eeg_gyro_{datetime.now():%Y%m%d_%H%M%S}.csv"
    filename = f"eeg_{index}.csv"
    filepath = os.path.join(SHARED_DIR, filename)

    if orientation == "vertical":
        df = save_data_vertical(packets_to_save)
    elif orientation == "horizontal":
        df = save_data_horizontal(packets_to_save)
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    df.to_csv(filepath, index=False)
    print(f"PATH: {filepath}\n")

    data_store.clear()


def data_reader_loop(emotiv, kalman_x, kalman_y,fs,window_seconds):

    # Calcola il numero di sample da acquisire per la finestra desiderata
    num_samples = fs * window_seconds

    samples_collected = 0
    while samples_collected < num_samples:
        packet = emotiv.read_packet()
        if packet is None:
            continue

        emotiv.data_store.append(packet)
        samples_collected += 1


def main():

    parser = argparse.ArgumentParser(description="EEG Acquisition")
    parser.add_argument('--fs', type=int , help="Set sampling frequency", default=128)
    parser.add_argument('--index', type=int, default=0, help="Set index of the packet to write")
    parser.add_argument('--window', type=int, default=3, help="Set window size in seconds",)
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for simulated data")
    parser.add_argument('--orientation', choices=['vertical', 'horizontal'], default='horizontal',
                        help="CSV orientation for saved data")
    args = parser.parse_args()

    emotiv = EmotivStreamer(debug_mode=args.debug)
    kalman_x, kalman_y = KalmanFilter(), KalmanFilter()

    if not emotiv.connect():
        exit(1)


    try:
        data_reader_loop(emotiv, kalman_x, kalman_y, args.fs, args.window)
        save_data_loop(emotiv.data_store, args.orientation, args.index)

    except KeyboardInterrupt:
        print("Session closed by User.")

    finally:    
        if emotiv.device:
            emotiv.device.close()
    

if __name__ == "__main__":
    main()
