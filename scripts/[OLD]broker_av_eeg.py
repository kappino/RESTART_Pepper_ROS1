#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.msg import audioVideo, audio_video_eeg
from care_er_ave.srv import Check
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
                # Aspetta che il service sia disponibile (timeout 10s)
                rospy.wait_for_service('check_av_service', timeout=10)
                check_av = rospy.ServiceProxy('check_av_service', Check)
                resp = check_av()  # Invia una richiesta vuota
                self.av_flag = resp.is_available
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn("AV Service non disponibile: %s", str(e))
                self.av_flag = False
            
            try:
                # Aspetta che il service sia disponibile (timeout 10s)
                rospy.wait_for_service('check_eeg_service', timeout=10)
                check_eeg = rospy.ServiceProxy('check_eeg_service', Check)
                resp = check_eeg()  # Invia una richiesta vuota
                self.eeg_flag = resp.is_available
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn("EEG Service non disponibile: %s", str(e))
                self.eeg_flag = False

            
            if not self.av_flag and not self.eeg_flag:
                exit()
            
            elif self.av_flag and not self.eeg_flag: #av è disponibile mentre eeg no
                self.pub_av = rospy.Publisher("av", audioVideo, queue_size=10)
                self.work_only_av()
            
            elif self.eeg_flag and not self.av_flag: #eeg disponibile mentre av no
                self.pub_eeg = rospy.Publisher("eeg", String, queue_size=10)
                self.work_only_eeg()
            
            elif self.av_flag and self.eeg_flag: #entrambi disponibili
                self.pub_av_eeg = rospy.Publisher("av_eeg", audio_video_eeg, queue_size=10)
                self.work_eeg_av()

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
            rospy.sleep(1)

    def work_eeg_av(self):
        msg_av_eeg = audio_video_eeg()
        while True:
            self.send_index.publish(self.index)
            eeg_path = rospy.wait_for_message("path_eeg", String, timeout=None)
            msg_av_eeg.eeg = eeg_path.data
            msg_av = rospy.wait_for_message("path_av", audioVideo, timeout=None)
            msg_av_eeg.video_path = msg_av.video_path
            msg_av_eeg.audio_path = msg_av.audio_path
            self.pub_av_eeg.publish(msg_av_eeg)
            rospy.sleep(1)
            self.index+=1


            
if __name__ == '__main__':
    try:
        rospy.init_node('broker_eeg_av', anonymous=True)
        rospy.loginfo("Node 'broker_eeg_av' has been initialized.")
        av_get = broker_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

            
