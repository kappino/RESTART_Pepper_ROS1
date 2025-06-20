#!/usr/bin/env python
import sys
import os
EEG_PROCESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EEG_processing")
sys.path.append(EEG_PROCESS_PATH)
import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.srv import Check, CheckResponse
from eeg_process import EEG_process

PATH_EEG_EMOTIV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EEG_processing/")

class EEG_get():
    def __init__(self):
        rospy.init_node('eeg_data_publisher', anonymous=True)
        self.eeg_process = EEG_process()
        self.pub_verify_eeg = rospy.Publisher("verify_eeg", Bool, queue_size=10, latch=True)
        self.pub_eeg_path = rospy.Publisher("path_eeg", String, queue_size=10, latch=True)
        rospy.Service('check_eeg_service', Check, self.handle_check_eeg)
        rospy.sleep(2)
        self.pub_verify_eeg.publish(True)
        rospy.Subscriber('index', Int32, self.start)
        rospy.loginfo("Node eeg_get created")

    
    def handle_check_eeg(self, arg):
        return CheckResponse(is_available=True)

    def start(self, data):
        index = data.data
        path_eeg = self.eeg_process.process_eeg_data(index)
        print(path_eeg)
        self.pub_eeg_path.publish(path_eeg)
         

if __name__ == "__main__":
    eeg_eg = EEG_get()
    rospy.spin()

