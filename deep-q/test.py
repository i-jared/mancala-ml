import argparse
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
    model_path = f"output/saved/{model_name}/weights.pth"
    config_path = f"output/saved/{model_name}/config.json"
    if not os.path.exists(model_path):
        model_path = f"output/{model_name}.pth"
        config_path = f"output/{model_name}_config.json"
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


def test_model_pytorch_robot(model_player: int, model_name: str, adv_model_name: str):
    adv_model_path = f"output/saved/{adv_model_name}/weights.pth"
    adv_config_path = f"output/saved/{adv_model_name}/config.json"
    if not os.path.exists(adv_model_path):
        adv_model_path = f"output/{adv_model_name}.pth"
        adv_config_path = f"output/{adv_model_name}_config.json"
        if not os.path.exists(adv_model_path):
            raise FileNotFoundError(
                f"Model file not found in both attempted locations: {adv_model_path} and ../{adv_model_name}.pth"
            )

    model_path = f"output/saved/{model_name}/weights.pth"
    config_path = f"output/saved/{model_name}/config.json"
    if not os.path.exists(model_path):
        model_path = f"output/{model_name}.pth"
        config_path = f"output/{model_name}_config.json"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found in both attempted locations: {model_path} and ../{model_name}.pth"
            )

    with open(config_path, "r") as file:
        config = json.load(file)
        layers = build_layers(config["layer_dims"], config["activations"])

    with open(adv_config_path, "r") as file:
        adv_config = json.load(file)
        adv_layers = build_layers(adv_config["layer_dims"], adv_config["activations"])

    env = MancalaBoard()

    model = NeuralNetwork(layers=layers).float()
    model.load_state_dict(torch.load(model_path))
    adv_model = NeuralNetwork(layers=adv_layers).float()
    adv_model.load_state_dict(torch.load(adv_model_path))
    env.init_adversary(adv_model)

    wins = []

    start = 0 if model_player == 0 else 7
    end = 6 if model_player == 0 else 13
    adv_start = 0 if model_player == 1 else 7
    adv_end = 6 if model_player == 1 else 13

    ## Test going first
    for _ in range(10000):
        player = 0
        done = False
        state = torch.tensor(env.reset() / 48.0).float()
        while not done:
            if player == model_player:
                with torch.no_grad():
                    q_pred = model(state)
                action = (
                    torch.where(
                        state[start:end] > 0,
                        q_pred,
                        -math.inf,
                    )
                    .argmax()
                    .item()
                    + start
                )
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
            else:
                with torch.no_grad():
                    q_pred = adv_model(state)
                action = (
                    torch.where(
                        state[adv_start:adv_end] > 0,
                        q_pred,
                        -math.inf,
                    )
                    .argmax()
                    .item()
                    + start
                )

                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
        wins.append(
            env.board[6 + (model_player * 7)] > env.board[13 - (model_player * 7)]
        )

    print(f"win percentage: {sum(wins) / len(wins)}")


def test_model_pytorch_random(model_player: int, model_name: str):
    model_path = f"output/saved/{model_name}/weights.pth"
    config_path = f"output/saved/{model_name}/config.json"
    if not os.path.exists(model_path):
        model_path = f"output/{model_name}.pth"
        config_path = f"output/{model_name}_config.json"
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
    parser = argparse.ArgumentParser(description="Test different model types.")
    parser.add_argument(
        "-m", "--model_type",
        type=str,
        choices=["human", "random", "robot"],
        help="Type of model: 'human', 'robot' or 'random'",
    )
    parser.add_argument(
        "player", type=int, choices=[0, 1], help="Player number: 0 or 1"
    )
    parser.add_argument("run", type=int, help="Run number: integer value")
    parser.add_argument(
        "-a",
        "--adv_path",
        type=str,
        help="Path to the adversary model, required if model_type is 'robot'",
    )
    args = parser.parse_args()

    if args.model_type == "robot" and not args.adv_path:
        parser.error("adv_path is required when model_type is 'robot'")

    print(f"Testing {args.model_type} for player {args.player} on run {args.run}")

    if args.model_type == "human":
        test_model_pytorch_human(args.player, f"{args.player}_{args.run}")
    elif args.model_type == "random":
        test_model_pytorch_random(args.player, f"{args.player}_{args.run}")
    elif args.model_type == "robot":
        test_model_pytorch_robot(
            args.player, f"{args.player}_{args.run}", args.adv_path
        )
    else:
        print('Error: type must be "human" or "random"')
        sys.exit(1)
