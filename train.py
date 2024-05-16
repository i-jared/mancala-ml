import argparse
from collections import deque
import json
import math
import os
import sys
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from typing import List

from mancala import MancalaBoard
from torch_network import NeuralNetwork, build_layers


def plot(data: List[float], model: str = None, show: bool = True):
    plt.plot(data)
    plt.xlabel("Iteration")
    plt.ylabel(model if model is not None else "Data")
    plt.title(model if model is not None else "Plot")
    plt.savefig(f"output/{'fig' if model is None else model}.png")
    if show:
        plt.show(block=show)
    plt.clf()


def train_model_pytorch(
    player: int,
    run: int,
    rewards: List[float],
    adversary_path: str,
    pretrain_weights_path: str,
):
    # get device
    device = "mps"
    print(f"Using {device} device")

    # initialize model and optimizer
    env = MancalaBoard()

    start = 0 if player == 0 else 7
    end = 6 if player == 0 else 13

    config = {
        "player": player,
        "run": run,
        "tau": 0.001,
        "discount": 0.99,
        "epsilon": 1.0,
        "decay_rate": 0.99,
        "min_epsilon": 0.05,
        "epochs": 1600,
        "learning_rate": 5e-6,
        "batch_size": 32,
        "update_frequency": 1,
        "max_buffer_size": 10000,
        "layer_dims": [14, 128, 64, 6],
        "activations": ["relu", "relu", "linear"],
        "rewards": rewards,  # win, lose, repeat, capture, gain 1, opp gains 1
    }

    env.init_rewards(rewards)

    layers = build_layers(config["layer_dims"], config["activations"])
    model = NeuralNetwork(layers=layers).float().to(device)
    target_model = NeuralNetwork(layers=layers).float().to(device)
    if pretrain_weights_path is not None:
        model.load_state_dict(torch.load(pretrain_weights_path))
        target_model.load_state_dict(torch.load(pretrain_weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    discount = config["discount"]
    epsilon = config["epsilon"]
    decay_rate = config["decay_rate"]
    min_epsilon = config["min_epsilon"]
    tau = config["tau"]
    batch_size = config["batch_size"]
    update_frequency = config["update_frequency"]
    replay_buffer = deque(maxlen=config["max_buffer_size"])
    epochs = config["epochs"]

    # save config
    with open(f"output/{player}_{run}_config.json", "w") as f:
        json.dump(config, f, indent=4)
    if adversary_path is not None:
        adv_config_path = f"{adversary_path}/config.json"
        if not os.path.exists(adv_config_path):
            adv_config_path = f"{adversary_path}_config.json"
        with open(adv_config_path, "r") as file:
            adv_config = json.load(file)
            adv_layers = build_layers(
                adv_config["layer_dims"], adv_config["activations"]
            )
        adversary_model = NeuralNetwork(layers=adv_layers).float().to(device)
        adv_weight_path = f"{adversary_path}/weights.pth"
        if not os.path.exists(adv_weight_path):
            adv_weight_path = f"{adversary_path}.pth"
        adversary_model.load_state_dict(torch.load(adv_weight_path))
        env.init_adversary(adversary_model)

    reward_hist = []
    cost_hist = []
    win_hist = []

    torch.set_default_dtype(torch.float)

    steps = 0
    for i in range(epochs):
        # for i in range(1):
        state = torch.tensor(env.reset(player) / 48.0).float().to(device)
        done = False
        total_reward = 0.0
        ave_cost = 0.0
        while not done:
            # choose action
            available_actions = torch.where(state[start:end] > 0)[0]
            if torch.rand(1) < epsilon:
                action = (
                    available_actions[
                        torch.randint(len(available_actions), (1,))
                    ].item()
                    + start
                )  # random
            else:
                with torch.no_grad():
                    q_pred = model(state)

                q_values = torch.where(state[start:end] > 0, 1, 0) * q_pred
                action = (
                    torch.where(state[start:end] > 0, q_values, -math.inf)
                ).argmax().item() + start  # max

            # make the action
            next_state, reward, done = env.step(action, player)
            next_state = torch.tensor(next_state / 48.0).float().to(device)
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                steps += 1
                if steps % update_frequency == 0:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.stack(states)
                    actions = torch.tensor(actions).unsqueeze(1).to(device)
                    rewards = torch.tensor(rewards).float().unsqueeze(1).to(device)
                    raw_next_states = torch.stack(next_states)
                    dones = torch.tensor(dones).unsqueeze(1).to(device)

                    mask = dones == 0
                    next_states = torch.masked_select(raw_next_states, mask).reshape(
                        -1, 14
                    )

                    # make predictions
                    q_preds = model(states).gather(1, actions - start)
                    next_state_qs = torch.zeros(batch_size, device=device)
                    # TODO: below might be the error... am i selecting valid
                    with torch.no_grad():
                        next_state_qs[mask.squeeze()] = (
                            target_model(next_states).max(1).values
                        )
                    q_actuals = rewards.squeeze() + discount * next_state_qs

                    loss_fn = nn.SmoothL1Loss()
                    loss = loss_fn(q_preds, q_actuals.unsqueeze(1))
                    ave_cost += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
                    optimizer.step()

            state = next_state
            # update target model
            model_dict = model.state_dict()
            target_dict = target_model.state_dict()
            for key in model_dict:
                target_dict[key] = model_dict[key] * tau + target_dict[key] * (1 - tau)
            target_model.load_state_dict(target_dict)
        epsilon = max(epsilon * decay_rate, min_epsilon)
        win_hist.append(env.board[end] > env.board[start - 1])
        reward_hist.append(total_reward)
        cost_hist.append((ave_cost / env.turn))
        if i % 100 == 0:
            print(
                f"Episode {i:04d}, cost: {(sum(cost_hist[-100:]) / 100):.2f}, reward: {(sum(reward_hist[-100:]) / 100):.2f}, win rate: {(sum(win_hist[-100:]) / 100):.2f}, eps: {epsilon:.2f}"
            )

    torch.save(model.state_dict(), f"output/{player}_{run}.pth")

    # get better data for graph
    cumulative_win_percentage = [
        sum(win_hist[max(0, i - 100) : i]) / min(i, 100)
        for i in range(1, len(win_hist) + 1)
    ]
    cumulative_reward_hist = [
        sum(reward_hist[max(0, i - 100) : i]) / min(i, 100)
        for i in range(1, len(reward_hist) + 1)
    ]

    return cost_hist, cumulative_reward_hist, cumulative_win_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for the game.")
    parser.add_argument("player", type=int, help="Player number (integer)")
    parser.add_argument("run", type=int, help="Run number (integer)")
    parser.add_argument(
        "-a", "--adversary", type=str, help="Path to adversary model folder (optional)"
    )
    parser.add_argument(
        "-p", "--pretrain", type=str, help="Path to pre-trained weights (optional)"
    )

    args = parser.parse_args()

    player = args.player
    run = args.run
    adversary_path = args.adversary
    pretrain_weights_path = args.pretrain

    #    for i in range(25):
    #        player, run = 1, i + 15
    #        rewards = [
    #        random.uniform(1, 5.0),  # win
    #        random.uniform(1, 5.0),  # lose
    #        random.uniform(0, 0.3),  # repeat
    #        random.uniform(0, 0.3),  # capture x N
    #        random.uniform(0, 0.2),  # new pieces in goal
    #        random.uniform(0, 0.2),  # new pieces in opp goal
    #        ]

    rewards = [3.0, 2.5, 0.1, 0.2, 0.1, 0.05]

    cost_hist, reward_hist, win_hist = train_model_pytorch(
        player, run, rewards, adversary_path, pretrain_weights_path
    )
    plot(cost_hist, model=f"{player}_{run}_cost", show=False)
    plot(reward_hist, model=f"{player}_{run}_reward", show=False)
    plot(win_hist, model=f"{player}_{run}_wins", show=False)
