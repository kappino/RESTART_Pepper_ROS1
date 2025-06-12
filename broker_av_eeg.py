#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.msg import audioVideo, audio_video_eeg
from care_er_ave.srv import Check
'''
    This node ROS allow to synchronize data eeg and av data
'''

class broker_node():
    def __init__(self):
        rospy.init_node('broker_eeg_av', anonymous=True)
        self.send_index = rospy.Publisher("index", Int32, queue_size=10)
        rospy.sleep(3)
    
        # Inizializza flags e index
        self.av_flag = False
        self.eeg_flag = False
        self.index = 0

        rospy.Subscriber('terapia_attiva', Bool, self.attiva_terapia)
        rospy.loginfo("Node 'broker_eeg_av' has been initialized.")
        
    def attiva_terapia(self, data):
        rospy.loginfo("messaggio ricevuto")
        if data.data:
            '''try:
                #Check if AV source is online
                msg = rospy.wait_for_message("verify_av", Bool, timeout = 10)
                self.av_flag = msg.data
            except rospy.ROSException:
                self.av_flag = False
        
            try:
                #Chcek if EEG source is online
                msg = rospy.wait_for_message("verify_eeg", Bool, timeout = 10)
                self.eeg_flag = msg.data
            except rospy.ROSException:
                self.eeg_flag = False'''
            
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
                rospy.loginfo("CAN NOT FIND ANY SOURCES EEG OR VIDEO")
                return()
            
            elif self.av_flag and not self.eeg_flag: #av è disponibile mentre eeg no
                self.pub_av = rospy.Publisher("av", audioVideo, queue_size=10)
                #Start comunication only with Pepper
                self.work_only_av()
            
            elif self.eeg_flag and not self.av_flag: #eeg disponibile mentre av no
                self.pub_eeg = rospy.Publisher("eeg", String, queue_size=10)
                #Start comunication only with EEG
                self.work_only_eeg()
            
            elif self.av_flag and self.eeg_flag: #entrambi disponibili
                self.pub_av_eeg = rospy.Publisher("av_eeg", audio_video_eeg, queue_size=10)
                #Start comunication with AV and EEG
                self.work_eeg_av()

    def work_only_eeg(self):
        #Send message for start recording
        self.send_index.publish(self.index)
        #Wait the path of file from EEG Source
        msg_eeg = rospy.wait_for_message("path_eeg", String, timeout=None)
        #Publish the messagge to the emotion recognition node
        self.pub_eeg.publish(msg_eeg)
        self.index += 1
        rospy.sleep(1)
    

    def work_only_av(self):
        #Send message for start recording
        self.send_index.publish(self.index)
        #Wait the path of file from Pepper source
        msg_av = rospy.wait_for_message("path_av", audioVideo, timeout=None)
        #print(msg_av)
        #Publish the messaghe to the emotion recognition node
        self.pub_av.publish(msg_av)
        self.index += 1
        rospy.sleep(1)

    def work_eeg_av(self):
        msg_av_eeg = audio_video_eeg()
        '''
            Message msg_av_eeg composed by:
            eeg String
            video_path String
            audio_path String
        '''
        #Send message to the EEG and AV Pepper for start recording
        self.send_index.publish(self.index)
        #Wait for messagges path from EEG and Pepper
        eeg_path = rospy.wait_for_message("path_eeg", String, timeout=None)
        msg_av_eeg.eeg = eeg_path.data
        rospy.sleep(1)
        msg_av = rospy.wait_for_message("path_av", audioVideo, timeout=None)
        rospy.sleep(1)
        msg_av_eeg.video_path = msg_av.video_path
        msg_av_eeg.audio_path = msg_av.audio_path
        #Send the message to the emotion recognition model
        self.pub_av_eeg.publish(msg_av_eeg)
        self.index+=1

if __name__ == '__main__':
    try:
        av_get = broker_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

            
