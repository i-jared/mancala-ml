from abc import ABC, abstractmethod
import numpy as np
from network import QNetwork


class Agent(ABC):
    @abstractmethod
    def decide(self, state: np.ndarray) -> int:
        pass


class RobotAgent(Agent):
    def __init__(self, player : int):
        self.player = player

    def random(self, board: np.ndarray) -> int:
        return rando(board, self.player)

    def decide(self, q_pred: np.ndarray, board: np.ndarray) -> int:
        """robot must be player 0"""
        action = np.argmax(np.where(board[:6] > 0, q_pred, 0))
        return action if self.player == 0 else action + 7

class HumanAgent(Agent):
    def decide(self, state: np.ndarray) -> int:
        valid = False
        while not valid:
            action = int(input("Enter a move: "))
            valid = state[action] > 0
        return action

class RandomAgent(Agent):
    def __init__(self, player: int): 
        self.player = player
        
    def decide(self, board: np.ndarray) -> int:
        return rando(board, self.player)


def rando(board: np.ndarray, player: int)-> int:
    if player == 0:
        start = 0
        end = 6
    else:
        start = 7
        end = 13
    valid_indices = np.where(board[start:end] > 0)[0]
    return np.random.choice(valid_indices)