#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String, Bool, Int32
from care_er_ave.msg import audioVideo
from care_er_ave.srv import Check, CheckResponse
class test_av():
    def __init__(self):
        rospy.init_node('broker_eeg_av', anonymous=True)
        rospy.Subscriber('index', Int32, self.send)
        rospy.Service('check_av_service', Check, self.handle_check_av)
        self.pub_av = rospy.Publisher('path_av', audioVideo, queue_size=10, latch=True)
        rospy.sleep(1)
    
    def handle_check_av(self, arg):
        return CheckResponse(is_available=True)

    def send(self,data):
        index = data.data
        msg = audioVideo()
        msg.video_path = "/root/restart_proj/audio_video_files/angry_1.mp4"
        msg.audio_path = "/root/restart_proj/audio_video_files/angry_1.wav"

        self.pub_av.publish(msg)

if __name__ == '__main__':
    test_av_node = test_av()
    rospy.spin()

    


        
        


