import cv2
import random
import os

def estrai_frame_casuali(video_path, num_frame=3, output_dir='frame_estratti'):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica il video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Totale frame nel video: {total_frames}")

    # Seleziona N frame casuali (escludendo i primi e ultimi per sicurezza)
    frame_indices = sorted(random.sample(range(1, total_frames - 1), num_frame))
    print(f"Frame selezionati: {frame_indices}")

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(output_dir, f'frame_{i+1}_at_{frame_idx}.jpg')
            cv2.imwrite(frame_name, frame)
            print(f"Salvato: {frame_name}")
        else:
            print(f"Errore nel leggere il frame {frame_idx}")

    cap.release()
    print("Estrazione completata.")


