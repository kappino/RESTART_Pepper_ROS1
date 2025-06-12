import torch
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LIST, SEED_IV_CHANNEL_LOCATION_DICT
import pandas as pd
import numpy as np
import mne

epoc_plus_channels = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2', 'FC5', 'FC6']

filtered_location_dict = {ch: SEED_IV_CHANNEL_LOCATION_DICT[ch] for ch in epoc_plus_channels}

'''def preprocess_data(eeg_csv):
    
    df = pd.read_csv(eeg_csv, skiprows=1)
    
    df.columns = df.columns.str.strip().str.replace("^EEG\\.", "", regex=True)
    
    eeg_data = df[epoc_plus_channels].values
    eeg_data = np.array(eeg_data)
    
    BDE = transforms.BandDifferentialEntropy()
    step1 = BDE(eeg=eeg_data)['eeg']
    
    Transform_to_grid = transforms.ToGrid(filtered_location_dict)
    step2 = Transform_to_grid(eeg=step1)['eeg']
    
    Tensor_transform = transforms.ToTensor()
    step3 = Tensor_transform(eeg=step2)['eeg']
    
    return step3'''


def preprocess_data(eeg_csv):
    
    df = pd.read_csv(eeg_csv)
    
    # Estrai i nomi dei canali
    channels = df['Channel']
    eeg_data = df.drop(columns=['Channel']).values
    
    '''eeg_data = df[epoc_plus_channels].values
    eeg_data = np.array(eeg_data)'''

    # === 2. Parametri di base ===
    original_sfreq = 256  
    target_sfreq = 200

    # === 3. Creazione del RawArray MNE ===
    info = mne.create_info(
        ch_names=channels.tolist(),
        sfreq=original_sfreq,
        ch_types='eeg'
    )
    # MNE richiede (n_channels, n_times)
    data = eeg_data.astype(np.float64)
    
    #QUA PARTONO LE TRASFORMAZIONI APPLICATE ANCHE DAI CREATORI DI SEEDIV
    '''
    the raw EEG data are first downsampled to a 200 Hz sampling rate. 
    To filter the noise and remove the artifacts, the EEG data are then processed with a bandpass filter between 1 Hz and 75 Hz
    '''
    raw = mne.io.RawArray(data, info)
    

    # === 4. Filtro passa-banda 1-75 Hz ===
    raw.filter(l_freq=1., h_freq=75., fir_design='firwin')

    # === 5. Downsampling a 200 Hz ===
    raw.resample(sfreq=target_sfreq)

    eeg_data = raw.get_data()
   
    #QUA PARTONO LE TRASFORMAZIONI PER INPUT AL MODELLO
    BDE = transforms.BandDifferentialEntropy()
    step1 = BDE(eeg=eeg_data)['eeg']
    
    Transform_to_grid = transforms.ToGrid(filtered_location_dict)
    step2 = Transform_to_grid(eeg=step1)['eeg']
    
    Tensor_transform = transforms.ToTensor()
    step3 = Tensor_transform(eeg=step2)['eeg']
    
    return step3
