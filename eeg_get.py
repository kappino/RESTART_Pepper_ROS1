#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.srv import Check, CheckResponse
import subprocess
import re
import os

PATH_EEG_EMOTIV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EEG_processing/")

class EEG_get():
    def __init__(self):
        rospy.init_node('eeg_data_publisher', anonymous=True)
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
        process = subprocess.Popen(
            ['conda', 'run', '-n', 'eeg_data', 'python', "eeg_process.py", "--debug", "--index", str(index)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd = PATH_EEG_EMOTIV
        )
        stdout, stderr = process.communicate()
        match = re.search(r'PATH:\s*(.+)', stdout)
        if match:
            path_eeg = match.group(1)
            print(path_eeg)
        
        self.pub_eeg_path.publish(path_eeg)
         

if __name__ == "__main__":
    eeg_eg = EEG_get()
    rospy.spin()

