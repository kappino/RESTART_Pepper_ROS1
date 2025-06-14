#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import qi



class Pepper:
    _instance=None
    def __init__(self, session):
        self.session = session
        self.robot_posture = self.session.service("ALRobotPosture")
        self.motion = self.session.service("ALMotion")
        #COMMENTATO PER LAVORARE CON SIMULATORE!!!
        #self.video_recorder = self.session.service("ALVideoRecorder")
        #self.audio_recorder = self.session.service("ALAudioRecorder")
        self.text_to_speech = self.session.service("ALTextToSpeech")
        self.animated_speech = self.session.service("ALAnimatedSpeech")
        self.memory = self.session.service("ALMemory")
        self.leds = self.session.service("ALLeds")
        #self.sr = self.session.service("ALSpeechRecognition")

        self.joints = {
            'head': ['HeadYaw', 'HeadPitch'],
            'left_arm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw'],
            'right_arm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'],
            'arms': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
                    'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
        }
        # Initialize helper modules
        #self.poses_module = poses.Poses(self.motion, self.robot_posture)
        self.set_speech_service()
        self.wake_up()
    # Singleton pattern
    #This pattern avoid to create multiple session
    @classmethod
    def create(cls, IP = "127.0.0.1", PORT = 9559):
        if cls._instance is None:
            try:
                session = qi.Session()
                session.connect("tcp://" + IP + ":" + str(PORT))
                print("[INFO] Connessione riuscita.")
                cls._instance = cls(session)
            except RuntimeError as e:
                print("[ERROR] Connessione fallita.",e)
                return None
        return cls._instance
    
    def connect_to_pepper(self):
        session = qi.Session()
        try:
            session.connect("tcp://" + self.IP + ":" + str(self.PORT))
            print("[INFO] Robot connection established.\n")
            return session
        
        except RuntimeError as e:
            print("[ERROR] It's impossible to connect to Pepper. Check IP address or connetction to network\n.")
            raise e
        

    def start_video_recording(self, filename):
        """
        Starts video recording with Pepper's front camera.
        
        Configuration:
        - Camera ID 0 (front camera)
        - Resolution 2 (640x480)
        - 10 FPS frame rate
        - MJPG video format
        - Saves to /home/nao/transfer directory
        
        Args:
            filename (str): Name for output video file (without extension)
        """
        try:
            self.video_recorder.setCameraID(0)
            self.video_recorder.setResolution(2)
            self.video_recorder.setFrameRate(10)
            self.video_recorder.setVideoFormat("MJPG") 
            print("[PEPPER] Video recording started...\n")
            
            self.video_recorder.startRecording("/home/nao/transfer", filename)
        except Exception as e:
            print("[ERROR] An issue occurred during video setup:", e)


    def start_audio_recording(self, filename):
        """
        Starts audio recording using Pepper's microphones.
        
        Configuration:
        - WAV format
        - 16kHz sample rate
        - Single channel recording
        - Saves to /home/nao/transfer directory
        
        Args:
            filename (str): Name for output audio file (without extension)
        """
        try:
            # Channel configuration: [FrontLeft, FrontRight, RearLeft, RearRight]
            audio_channels = [0, 0, 1, 0]
            full_path = "/home/nao/transfer/" + filename + ".wav"
            self.audio_recorder.startMicrophonesRecording("/home/nao/transfer/" + filename + ".wav", "wav", 16000, audio_channels)
            print("[PEPPER] Audio recording started...\n")
        except Exception as e:
            print("[ERROR] An issue occurred during audio setup:", e)


    def stop_video_recording(self):
        try:
            video_info = self.video_recorder.stopRecording()
        except Exception as e:
            print("[ERROR] An issue occurred while stopping video recording:", e)

    def stop_audio_recording(self):
        """Stops any ongoing video recording and returns recording info."""
        try:
            self.audio_recorder.stopMicrophonesRecording()
        except Exception as e:
            print("[ERROR] Failed to stop audio recording:", e)
        

    def record_audio_video(self, video_filename, audio_filename):
        try:
            self.stop_video_recording()
            self.stop_audio_recording()
            
            self.start_video_recording(video_filename)
            self.start_audio_recording(audio_filename)
           
        except Exception as e:
            print("[ERROR] An issue occurred:", e)     

    def stop_recording(self):
        self.stop_video_recording()
        self.stop_audio_recording()   
        print("[INFO] Audio and video recording successfully completed!\n")

    def set_speech_service(self):
        """
        Configures speech recognition service with:
        - English language
        - Predefined vocabulary
        - Proper subscription
        """
        try:
            # Temporarily pause recognition during configuration
            #self.speech_recognition.pause(True)
            
            # Set language for both recognition and synthesis
            #self.speech_recognition.setLanguage("English")
            self.text_to_speech.setLanguage("Italian")
            
            # Configure vocabulary (without word spotting)
            #self.speech_recognition.setVocabulary(Pepper.VOCABULARY, False)
            
            # Activate recognition service
            #self.speech_recognition.subscribe("Recognizer")
            #self.speech_recognition.pause(False)
            
        except Exception as e:
            print("[ERROR] Failed to configure speech service: {str(e)}".format(e))
            raise

    def pepper_say(self, sentence):
        self.text_to_speech.say(sentence)
    def pepper_animated_say(self, sentence, config):
        self.animated_speech.say(sentence, config)

    # Imposta i LED degli occhi su un colore RGB (in percentuale da 0.0 a 1.0)
    def set_eye_color(self, hex_color, duration=1.0):
        # Imposta ogni componente colore separatamente
        self.leds.fadeRGB("FaceLeds", hex_color , duration)

    
    def wake_up(self):
        """Wake up the robot if not already awake"""
        self.motion.wakeUp()
        time.sleep(0.5)
        
    def rest(self):
        """Put the robot in rest position"""
        self.motion.rest()

    
    def go_to_posture(self, posture_name, speed=0.5):
        """Go to a predefined posture (Stand, Sit, Crouch, etc.)"""
        self.robot_posture.goToPosture(posture_name, speed)

    
    def move_joints(self, joint_names, angles, speed=0.2, is_absolute=True):
        """Move specific joints to target angles"""
        self.motion.angleInterpolation(joint_names, angles, speed, is_absolute)

    # =============== Head Movements ===============
    def move_head(self, yaw, pitch, speed=0.1):
        """Move head to specific yaw and pitch positions"""
        self.move_joints(self.joints['head'], [yaw, pitch], speed)

    
    def look_at(self, direction):
        """Look at predefined directions ('front', 'left', 'right', 'up', 'down')"""
        directions = {
            'front': [0.0, 0.0],
            'left': [1.0, 0.0],
            'right': [-1.0, 0.0],
            'up': [0.0, -0.5],
            'down': [0.0, 0.5]
        }
        if direction in directions:
            self.move_head(*directions[direction])

    def lock_head(self):
        """
        Locks Pepper's head in a stable position for recording or focused interaction.
        Sets head joints to neutral position and increases stiffness.
        """
        self.robot_posture.goToPosture("StandInit", 0.5)  # Ensure standing posture
        names = self.joints["heads"]
        angles = [0.0, 0.0]  # Neutral head position
        fraction_max_speed = 0.2  # Moderate speed for smooth movement
        self.motion.setAngles(names, angles, fraction_max_speed)
        
        self.motion.setStiffnesses("Head", 1.0)  # Maximum stiffness to prevent movement
        print("PEPPER BLOCCATO")
        
    def unlock_head(self):
        """Releases head stiffness allowing natural movement."""
        self.motion.setStiffnesses("Head", 0.0)  # Zero stiffness allows free movement

    # =============== Arm Movements ================
    def move_arms(self, left_arm_angles, right_arm_angles, speed=0.2):
        """Move both arms to specified angles"""
        all_angles = left_arm_angles + right_arm_angles
        self.move_joints(self.joints['arms'], all_angles, speed)

    
    def reset_arms(self, speed=0.2):
        """Reset arms to default position"""
        default_left = [1.0, 0.0, -1.5, -0.1, 0.0]
        default_right = [1.0, 0.0, 1.5, 0.1, 0.0]
        self.move_arms(default_left, default_right, speed)

    

