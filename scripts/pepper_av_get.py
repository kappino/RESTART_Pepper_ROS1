#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary libraries
from pepper import Pepper  # Main Pepper robot interface
import subprocess  # For running system commands
import time  # For timing operations
import threading  # For parallel execution
import os  # For filesystem operations
import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.srv import Check, CheckResponse
from care_er_ave.msg import audioVideo

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
video_dir = os.path.join(parent_dir, "video")
audio_dir = os.path.join(parent_dir, "audio")
input_dir = os.path.join(parent_dir, "inputs")

# Ensure required directories exist
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

print(video_dir)
print(audio_dir)

IP = "169.254.201.73"
PORT = 9559

class Pepper_commands():
    def __init__(self):
        rospy.init_node('Pepper_av_get', anonymous=True)
        #self.pub_check_audio_video = rospy.Publisher("verify_av", Bool, queue_size=10, latch=True) #This pub verify if pepper is online
        
        self.pub_send_output_path = rospy.Publisher('path_av', audioVideo, queue_size=10, latch=True)#This pub send the path when it is ready
        rospy.Subscriber('index', Int32, self.start)
        rospy.sleep(2)
        self.pepper_test = Pepper.create(IP, PORT)
        if self.pepper_test is None:
            rospy.loginfo("Node 'Pepper_av_get' can not be initialized, pepper is not online.")
            exit()
        else:
            rospy.loginfo("Node 'Pepper_av_get' has been initialized.")
          # Stabilize for recording
        rospy.Service('check_av_service', Check, self.handle_check_av)
        rospy.sleep(1)

    def handle_check_av(self, arg):
        return CheckResponse(is_available=(self.pepper_test is not None))
    
    def get_ssh_command(self, file_type, index):
        """
        Generates SSH command to transfer files from Pepper to local machine.
        
        Args:
            file_type (str): "audio" or "video" 
            index (int): File index number
            
        Returns:
            list: Command list for subprocess execution
        """
        file_extension = "wav" if file_type == "audio" else "avi"
        return [
            "sshpass", "-p", "pepperina2023", "scp",  # Using sshpass for password automation
            f"nao@{IP}:/home/nao/transfer/pepper_{file_type}_{index}.{file_extension}",  # Source Example: pepper_audio_0.wav or pepper_video_0.avi
            f"{parent_dir}/{file_type}/pepper_{file_type}_{index}.{file_extension}"  # Destination
        ]
    
    def start_recording_session(self, index):  
        """
        Starts synchronized audio and video recording on Pepper.
        
        Args:
            pepper_test: Initialized Pepper instance
        """
        self.pepper_test.record_audio_video(f"pepper_video_{index}", f"pepper_audio_{index}")
        
    def stop_recording_session(self):
        """
        Stops ongoing recordings and increments file counters.
        
        Args:
            pepper_test: Initialized Pepper instance
        """
        self.pepper_test.stop_recording()

    def start(self, data):
        self.index = data.data
        recording_thread = threading.Thread(target=self.start_recording_session, args=(self.index,), daemon=True)
        self.pepper_test.lock_head()
        recording_thread.start()
        time.sleep(3)
        recording_thread.join()
        self.stop_recording_session()
        self.pepper_test.unlock_head()

        # Transfer and process media
        process_video = subprocess.Popen(self.get_ssh_command("video", self.index))
        process_audio = subprocess.Popen(self.get_ssh_command("audio", self.index))
        process_video.wait()  # Blocca finché il comando non termina
        process_audio.wait()
        msg = audioVideo()
        msg.video_path = os.path.join(video_dir, f"pepper_video_{self.index}.avi")
        msg.audio_path = os.path.join(audio_dir, f"pepper_audio_{self.index}.wav")
        
        #output_file = self.merge_audio_video(self.index)
        self.pub_send_output_path.publish(msg)
        

if __name__ == '__main__':
    try:
        av_get = Pepper_commands()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass      
