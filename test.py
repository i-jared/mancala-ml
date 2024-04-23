import sys
import json
import math
import os
import numpy as np
import torch
from agents import HumanAgent, RandomAgent
from mancala import MancalaBoard
from torch_network import NeuralNetwork, build_layers


def test_model_pytorch_human(model_player: int, model_name: str):
    # load model weights
    model_path = f"output/{model_name}.pth"
    config_path = f"output/{model_name}_config.json"
    if not os.path.exists(model_path):
        model_path = f"output/saved/{model_name}/{model_name}.pth"
        config_path = f"output/saved/{model_name}/{model_name}_config.json"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found in both attempted locations: {model_path} and ../{model_name}.pth"
            )

    with open(config_path, "r") as file:
        config = json.load(file)
        layers = build_layers(config["layer_dims"], config["activations"])

    model = NeuralNetwork(layers=layers)
    model.load_state_dict(torch.load(model_path))
    start = 0 if model_player == 0 else 7
    end = 6 if model_player == 0 else 13
    human = HumanAgent()

    env = MancalaBoard()
    state = torch.tensor(env.reset() / 48.0).float()
    done = False

    player = 0
    while not done:
        if player == model_player:
            with torch.no_grad():
                q_pred = model(state)
            q_values = torch.where(state[start:end] > 0, 1, 0) * q_pred
            action = (
                torch.where(q_values != 0, q_values, -math.inf)
            ).argmax().item() + start

            state, player, done = env.step_test(action, player)
            state = torch.tensor(state / 48.0).float()
        else:
            action = human.decide(env.board)
            state, player, done = env.step_test(action, player)
            state = torch.tensor(state / 48.0).float()
        print(env)


def test_model_pytorch_random(model_player: int, model_name: str):
    model_path = f"output/{model_name}.pth"
    config_path = f"output/{model_name}_config.json"
    if not os.path.exists(model_path):
        model_path = f"output/saved/{model_name}/{model_name}.pth"
        config_path = f"output/saved/{model_name}/{model_name}_config.json"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found in both attempted locations: {model_path} and ../{model_name}.pth"
            )

    with open(config_path, "r") as file:
        config = json.load(file)
        layers = build_layers(config["layer_dims"], config["activations"])

    env = MancalaBoard()

    model = NeuralNetwork(layers=layers).float()
    model.load_state_dict(torch.load(model_path))
    robot = RandomAgent((model_player + 1) % 2)
    wins = []

    ## Test going first
    for _ in range(10000):
        player = 0
        done = False
        state = torch.tensor(env.reset() / 48.0).float()
        while not done:
            if player == model_player:
                with torch.no_grad():
                    q_pred = model(state)
                action = torch.where(
                    state[0 + (model_player * 7) : 6 + (model_player * 7)] > 0,
                    q_pred,
                    -math.inf,
                ).argmax().item() + (
                    model_player * 7
                )  # max
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
            else:
                action = robot.decide(env.board)
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
        wins.append(
            env.board[6 + (model_player * 7)] > env.board[13 - (model_player * 7)]
        )

    print(f"win percentage: {sum(wins) / len(wins)}")


def test_random():
    env = MancalaBoard()
    thing1 = RandomAgent(0)
    thing2 = RandomAgent(1)
    wins = []

    for _ in range(1000):
        player = 0
        done = False
        env.reset()
        while not done:
            if player == 0:
                action = thing1.decide(env.board)
            else:
                action = thing2.decide(env.board)
            _, player, done = env.step_test(action, player)
        wins.append(env.board[6] > env.board[13])
    print(f"random win percentage (going first): {sum(wins) / len(wins)}")


if __name__ == "__main__":
    if len(sys.argv) > 4:
        print(
            "Usage: python test.py [model_type ('human', 'random')] [player (0, 1)] [run (0,1,...)]"
        )
        sys.exit(1)
    try:
        type: str = sys.argv[1]
        player = int(sys.argv[2])
        run = int(sys.argv[3])
        print(f"Testing {type} for player {player} on run {run}")
    except ValueError:
        print(
            'Error: player and run must be integers, type must be "human" or "random"'
        )
        sys.exit(1)

    if type == "human":
        test_model_pytorch_human(player, f"{player}_{run}")
    elif type == "random":
        test_model_pytorch_random(player, f"{player}_{run}")
    else:
        print('Error: type must be "human" or "random"')
        sys.exit(1)
