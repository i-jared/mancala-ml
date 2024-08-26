import argparse
from collections import deque
import time
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
        "tau": 0.003,
        "discount": 1.00,
        "epsilon": 1.0,
        "decay_rate": 0.999,
        "min_epsilon": 0.01,
        "epochs": 5000,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "update_frequency": 1,
        "max_buffer_size": 10000,
        "layer_dims": [14, 256, 128, 64, 6],
        "activations": ["relu", "relu", "relu", "linear"],
        "rewards": rewards,  # win, lose, repeat, capture, gain 1, opp gains 1
    }

    env.init_rewards(rewards)

    layers = build_layers(config["layer_dims"], config["activations"])
    model = NeuralNetwork(layers=layers).float().to(device)
    target_model = NeuralNetwork(layers=layers).float().to(device)
    if pretrain_weights_path is not None:
        model.load_state_dict(torch.load(pretrain_weights_path))
        target_model.load_state_dict(torch.load(pretrain_weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.993)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, total_steps=config["epochs"])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000], gamma=0.5)

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
        scheduler.step()
        win_hist.append(env.board[end] > env.board[start - 1])
        reward_hist.append(total_reward)
        cost_hist.append((ave_cost / env.turn))
        if i % 100 == 0:
            # if (i != 0):
                # scheduler.step(sum(cost_hist[-100:]) / 100)
            print(
                f"Episode {i:04d}, cost: {(sum(cost_hist[-100:]) / 100):.2f}, reward: {(sum(reward_hist[-100:]) / 100):.2f}, win rate: {(sum(win_hist[-100:]) / 100):.2f}, eps: {epsilon:.2f}, lr: {scheduler.get_last_lr()[0]:.2e}"
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


# def monte_carlo_tree_search(env, model, num_simulations=1000, exploration_constant=1.4):
#     class MCTSNode:
#         def __init__(self, state, parent=None, action=None):
#             self.state = state
#             self.parent = parent
#             self.action = action
#             self.children = {}
#             self.visits = 0
#             self.value = 0

#         def is_fully_expanded(self):
#             return len(self.children) == len(env.get_valid_moves(self.state))

#         def select_child(self):
#             return max(self.children.values(), key=lambda node: node.ucb_score(exploration_constant))

#         def expand(self):
#             action = random.choice([a for a in env.get_valid_moves(self.state) if a not in self.children])
#             next_state, _ = env.step(self.state, action)
#             child = MCTSNode(next_state, self, action)
#             self.children[action] = child
#             return child

#         def backpropagate(self, result):
#             self.visits += 1
#             self.value += result
#             if self.parent:
#                 self.parent.backpropagate(result)

#         def ucb_score(self, c):
#             if self.visits == 0:
#                 return float('inf')
#             return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

#     root = MCTSNode(env.get_state())

#     for _ in range(num_simulations):
#         node = root
#         while node.is_fully_expanded():
#             node = node.select_child()
        
#         if not node.is_fully_expanded():
#             node = node.expand()
        
#         state = node.state
#         while not env.is_game_over(state):
#             action = env.get_random_action(state)
#             state, _ = env.step(state, action)
        
#         result = env.get_result(state)
#         node.backpropagate(result)

#     return max(root.children.items(), key=lambda item: item[1].visits)[0]

# def mcts_policy(env, model, state):
#     return monte_carlo_tree_search(env, model)

# # Example usage in the training loop
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         action = mcts_policy(env, model, state)
#         next_state, reward, done, _ = env.step(action)
#         # Add experience to replay buffer or update model directly
#         state = next_state
    
#     # Update model periodically or after each episode
#     update_model(model, replay_buffer)



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

    rewards = [
        1.0,   # win
        0.0,   # lose
        0.0,   # repeat
        0.0,  # capture x N
        0.00,  # new pieces in goal
        0.00, # new pieces in opp goal
    ]

    start = time.time()
    cost_hist, reward_hist, win_hist = train_model_pytorch(
        player, run, rewards, adversary_path, pretrain_weights_path
    )
    end = time.time()
    print('time: ', end-start)
    plot(cost_hist, model=f"{player}_{run}_cost", show=False)
    plot(reward_hist, model=f"{player}_{run}_reward", show=False)
    plot(win_hist, model=f"{player}_{run}_wins", show=False)
