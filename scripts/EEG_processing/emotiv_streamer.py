import hid
from Crypto.Cipher import AES
from datetime import datetime
import random
import time


class EmotivStreamer:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.vid = 0x1234
        self.pid = 0xed02
        self.device = None
        self.cipher = None
        self.cypher_key = bytes.fromhex("31003554381037423100354838003750")
        self.data_store = []
        self._last_packet_time = 0
        self._sampling_interval = 1/128  # 128Hz sampling rate
    def connect(self):
        if self.debug_mode:
            print("Debug mode: Simulated connection to Emotiv device.")
            return True
        try:
            self.device = hid.device()
            self.device.open(self.vid, self.pid)
            self.device.set_nonblocking(1)
            self.cipher = AES.new(self.cypher_key, AES.MODE_ECB)
            print(f"Connected to Emotiv device {self.vid:04x}:{self.pid:04x}")
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    def read_packet(self):
        if self.debug_mode:
            # Simulate blocking read at 128Hz (every ~7.8ms)
            now = time.time()
            elapsed = now - self._last_packet_time
            wait_time = self._sampling_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_packet_time = time.time()

            simulated_packet = {
            'timestamp': datetime.now().isoformat(),
            'counter': random.randint(0, 255),
            'gyro_x': random.randint(-128, 127),
            'gyro_y': random.randint(-128, 127),
            'eeg': [random.randint(-2048, 2047) for _ in range(14)],  # assuming 14 EEG channels
            'battery': random.choice([0, 100])
            }
            return simulated_packet

        encrypted = bytes(self.device.read(32))
        if not encrypted:
            return None
        decrypted = self.cipher.decrypt(encrypted)
        return {
            'timestamp': datetime.now().isoformat(),
            'counter': decrypted[0],
            'gyro_x': decrypted[29] - 102,
            'gyro_y': decrypted[30] - 204,
            'eeg': [int.from_bytes(decrypted[i:i+2], 'big', signed=True) for i in range(1, 29, 2)],
            'battery': (decrypted[31] & 0x01) * 100
        }
