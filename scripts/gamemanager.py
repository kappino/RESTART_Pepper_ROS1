#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from pepper_connection import Pepper
from memory_game import MemoryGame  # Altri giochi da importare se servono

class GameManager:
    def __init__(self):
        rospy.init_node("game_manager")
        self.pepper = Pepper.create(IP="127.0.0.1", PORT=9559)

        if not self.pepper:
            rospy.logerr("Connessione con Pepper fallita. Arresto.")
            return

        rospy.loginfo("Connessione a Pepper riuscita.")
        self.game_name = rospy.get_param("~game_name", "memory_game")

        # Istanziazione gioco
        if self.game_name == "memory_game":
            self.game = MemoryGame("Gioco di memoria", pepper=self.pepper)
        else:
            rospy.logwarn("Gioco '{self.game_name}' non riconosciuto.")
            return

        self.run_game()

    def run_game(self):
        try:
            self.game.start()
        except KeyboardInterrupt:
            rospy.loginfo("Gioco interrotto manualmente.")
        except Exception as e:
            rospy.logerr("Errore durante l'esecuzione del gioco: {e}")
        finally:
            rospy.signal_shutdown("Gioco terminato.")

if __name__ == "__main__":
    GameManager()
