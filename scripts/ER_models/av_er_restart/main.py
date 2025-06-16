from pose_image_er import pose_image_er
from audio_er import audio_er
from extract_frames import estrai_frame_casuali
from collections import Counter
import os
from av_model import av_model
def path_files():
    cartella = "/home/audio_video_er/frame_estratti"  # Sostituisci con il percorso della tua cartella
    file_paths = []

    with os.scandir(cartella) as entries:
        for entry in entries:
            if entry.is_file():
                file_paths.append(entry.path)
    return file_paths
def main():
    av = av_model()
    print(av.emotion_recognition("/root/restart_proj/audio_video_files/angry_1.mp4", "/root/restart_proj/audio_video_files/angry_1.wav"))



    #estrai_frame_casuali("/home/audio_video_files/disgusted.mp4")
    #exit()
    #file_paths = path_files()
    '''file_paths = ["/root/restart_proj/catkin_ws/src/RESTART_Pepper_ROS1/scripts/ER_models/av_er_restart/example.jpg"]
    emotions = []
    scores = []
    pose_image = pose_image_er()
    for file in file_paths:
        pose_image.read_image(file)
        emotion, final_emotion_score, posture_type, final_emotion_score_with_sentiment = pose_image.classify_emotion()
        print("EMOTION: ", emotion)
        print("FINAL EMOTION SCORE: ", final_emotion_score)
        print("POSTURE_TYPE: ", posture_type)
        print("FINAL EMOTION SCORE WITH SENTIMENT: ", final_emotion_score_with_sentiment)
        emotions.append(emotion)
        scores.append(final_emotion_score_with_sentiment)
    conteggio = Counter(emotions)
    emozione_piu_frequente = conteggio.most_common(1)[0][0]
    avg_scores = sum(scores) / len(scores)
        
    audioer = audio_er()
    sentiment = audioer.analyze_sentiment("/root/restart_proj/catkin_ws/src/RESTART_Pepper_ROS1/scripts/ER_models/av_er_restart/test.wav")
    print("EMOTION AUDIO: ", sentiment)'''
main()