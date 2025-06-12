import torch
from torch.utils.data import Dataset, DataLoader
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT
import pandas as pd
import numpy as np

# Lista dei canali EEG disponibili su EPOC X
epocx_channels = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2', 'FC5', 'FC6']

# Dizionario delle posizioni dei canali (solo quelli presenti su EPOC X)
filtered_location_dict = {ch: SEED_IV_CHANNEL_LOCATION_DICT[ch] for ch in epocx_channels if ch in SEED_IV_CHANNEL_LOCATION_DICT}

class EEGDataset(Dataset):
    def __init__(self, eeg_csv, window_size=256, step_size=128, transform=None):
        """
        :param eeg_csv: Percorso al file CSV contenente i dati EEG.
        :param window_size: Numero di campioni per finestra (default: 256 = 1 sec a 256Hz).
        :param step_size: Passo tra una finestra e l'altra (default: 128 = sovrapposizione del 50%).
        :param transform: Trasformazioni da applicare ai dati.
        """
        self.transform = transform
        
        # Carica il CSV saltando la prima riga con i metadati
        df = pd.read_csv(eeg_csv, skiprows=1)
        
        # Rimuove il prefisso "EEG." dai nomi delle colonne
        df.columns = df.columns.str.strip().str.replace("^EEG\\.", "", regex=True)
        
        # Estrai solo i canali EEG come numpy array
        self.eeg_data = df[epocx_channels].values.astype(np.float32)
        
        # Parametri per segmentazione
        self.window_size = window_size
        self.step_size = step_size
        self.samples = self.create_windows()
    
    def create_windows(self):
        """Segmenta i dati EEG in finestre temporali sovrapposte."""
        windows = []
        for start in range(0, len(self.eeg_data) - self.window_size, self.step_size):
            windows.append(self.eeg_data[start:start + self.window_size])
        return np.array(windows)  # Shape: (num_finestre, window_size, num_canali)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        eeg_window = self.samples[idx]
        
        if self.transform:
            eeg_window = self.transform(eeg=eeg_window)['eeg']
        
        return eeg_window

def get_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.BandDifferentialEntropy(),
        transforms.ToGrid(filtered_location_dict),
        transforms.ToTensor()
    ])

    dataset = EEGDataset(dataset_path, window_size=256, step_size=128, transform=transform)
    return dataset

    # Creazione del DataLoader per l'addestramento
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Verifica della forma dei dati
    # sample = next(iter(dataloader))
    # print(f"Shape del batch: {sample.shape}")  # (batch_size, num_canali, grid_x, grid_y)
