from agents import RandomAgent, HumanAgent, RobotAgent
from layer import Activation, QLayer
from mancala import MancalaBoard
import time

from network import QNetwork


def test_random_agent():
    agent = RandomAgent(0)

    game_over = False
    env = MancalaBoard()
    board = env.board
    while not game_over:
        action = agent.decide(board)
        board, _, game_over = env.step(action)
        time.sleep(5)
        print(env)


def test_human_agent():
    agent1 = HumanAgent()
    agent2 = RandomAgent(1)

    game_over = False
    env = MancalaBoard()
    board = env.board
    current_player = 0
    while not game_over:
        print(env)
        if current_player == 0:
            player = agent1
        else:
            player = agent2
        action1 = player.decide(board)
        board, _, game_over, current_player = env.step(action1)


def test_robot_agent():
    net1 = QNetwork(
        [
            QLayer(14, 64, Activation.RELU),
            QLayer(64, 64, Activation.RELU),
            QLayer(64, 6, Activation.NONE),
        ],
        0.01,
    )
    agent1 = RobotAgent(net1, 0)
    agent2 = RandomAgent(1)

    game_over = False
    env = MancalaBoard()
    board = env.board
    current_player = 0
    while not game_over:
        if current_player == 0:
            player = agent1
        else:
            player = agent2
        action1 = player.decide(board)
        board, _, game_over, current_player = env.step(action1)
        time.sleep(5)
        print(env)


if __name__ == "__main__":
    test_random_agent()
