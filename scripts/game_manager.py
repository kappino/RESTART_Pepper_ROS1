#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from memory_game import MemoryGame  # Altri giochi da importare se servono

IP = "host.docker.internal"
PORT = 9559

class GameManager:
    def __init__(self):
        rospy.init_node("game_manager")
        rospy.Subscriber('/start_game', String, self.start_game)

    
    def start_game(self, data):
        game_name = data.data
        # Istanziazione gioco
        if game_name == "memory_game":
            self.game = MemoryGame("Gioco di memoria", pepper=self.pepper)
        else:
            rospy.logwarn(f"Gioco '{game_name}' non riconosciuto.")
            return
        self.run_game()

    def run_game(self):
        try:
            self.game.start()
        except KeyboardInterrupt:
            rospy.loginfo("Gioco interrotto manualmente.")
        except Exception as e:
            rospy.logerr(f"Errore durante l'esecuzione del gioco: {e}")
        finally:
            rospy.signal_shutdown("Gioco terminato.")

if __name__ == "__main__":
    GameManager()
