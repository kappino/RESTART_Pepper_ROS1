#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.msg import audioVideo, audio_video_eeg
import random
import threading

class broker_node():
    def __init__(self):
        self.send_index = rospy.Publisher("index", Int32, queue_size=10)
        rospy.sleep(3)
    
        # Inizializza flags e index
        self.av_flag = False
        self.eeg_flag = False
        self.index = 0
        
        # Threads
        self.av_thread = None
        self.eeg_thread = None
        self.av_eeg_thread = None

        rospy.Subscriber('terapia_attiva', Bool, self.attiva_terapia)
        
    def attiva_terapia(self, data):
        rospy.loginfo("messaggio ricevuto")
        if data.data:
            try:
                mag = rospy.wait_for_message("verify_av", Bool, timeout = 10)
                self.av_flag = mag.data
                rospy.loginfo(self.av_flag)
            except rospy.ROSException:
                
                self.av_flag = False
        
            try:
                msg = rospy.wait_for_message("verify_eeg", Bool, timeout = 10)
                self.eeg_flag = msg.data
            except rospy.ROSException:
                
                self.eeg_flag = False
            if not self.av_flag and not self.eeg_flag:
                exit()
            
            elif self.av_flag and not self.eeg_flag: #av è disponibile mentre eeg no
               
                self.pub_av = rospy.Publisher("av", audioVideo, queue_size=10)
                # Avvia thread per work_only_av
                if self.av_thread is None or not self.av_thread.is_alive():
                    self.av_thread = threading.Thread(target=self.work_only_av)
                    self.av_thread.daemon = True  # Thread termina se il main thread termina
                    self.av_thread.start()
                    rospy.loginfo("Thread work_only_av avviato")
            
            elif self.eeg_flag and not self.av_flag: #eeg disponibile mentre av no
                self.pub_eeg = rospy.Publisher("eeg", String, queue_size=10)

                if self.eeg_thread is None or not self.eeg_thread.is_alive():
                    self.eeg_thread = threading.Thread(target=self.work_only_eeg)
                    self.eeg_thread.daemon = True  # Thread termina se il main thread termina
                    self.eeg_thread.start()
                    rospy.loginfo("Thread work_only_eeg avviato")
            
            elif self.av_flag and self.eeg_flag: #entrambi disponibili
                self.pub_av_eeg = rospy.Publisher("av_eeg", audio_video_eeg, queue_size=10)
               
                if self.av_eeg_thread is None or not self.av_eeg_thread.is_alive():
                    self.av_eeg_thread = threading.Thread(target=self.work_eeg_av)
                    self.av_eeg_thread.daemon = True  # Thread termina se il main thread termina
                    self.av_eeg_thread.start()
                    rospy.loginfo("Thread work_av_eeg avviato")

    def work_only_eeg(self):
        while True:
            self.send_index.publish(self.index)
            msg_eeg = rospy.wait_for_message("path_eeg", String, timeout=None)
            self.pub_eeg.publish(msg_eeg)
            self.index += 1
            rospy.sleep(1)
    

    def work_only_av(self):
        while True:
            self.send_index.publish(self.index)
            msg_av = rospy.wait_for_message("path_av", audioVideo, timeout=None)
            #print(msg_av)
            self.pub_av.publish(msg_av)
            self.index += 1

    def work_eeg_av(self):
        msg_av_eeg = audio_video_eeg()
        while True:
            self.send_index.publish(self.index)
            eeg_path = rospy.wait_for_message("path_eeg", String, timeout=None)
            msg_av_eeg.eeg = eeg_path.data
            rospy.sleep(2)
            msg_av = rospy.wait_for_message("path_av", audioVideo, timeout=None)
            rospy.sleep(2)
            msg_av_eeg.video_path = msg_av.video_path
            msg_av_eeg.audio_path = msg_av.audio_path
            self.pub_av_eeg.publish(msg_av_eeg)
            self.index+=1


            
if __name__ == '__main__':
    try:
        rospy.init_node('broker_eeg_av', anonymous=True)
        rospy.loginfo("Node 'broker_eeg_av' has been initialized.")
        av_get = broker_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

            
