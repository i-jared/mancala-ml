from collections import deque
import math
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from typing import List

from mancala import MancalaBoard
from torch_network import NeuralNetwork


def plot(data: List[float], model: str = None):
    plt.plot(data)
    plt.xlabel("Iteration")
    plt.ylabel(model if model is not None else "Data")
    plt.title(model if model is not None else "Plot")
    plt.savefig(f"output/{'fig' if model is None else model}.png")
    plt.show(block=True)



def train_model_pytorch(player: int, run: int):
    # get device
    device = "mps"
    print(f"Using {device} device")

    # initialize model and optimizer
    model = NeuralNetwork().float().to(device)
    target_model = NeuralNetwork().float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    env = MancalaBoard()
    start = 0 if player == 0 else 7
    end = 6 if player == 0 else 13

    discount = 1.01
    epsilon = 1.0
    decay_rate = 0.99
    min_epsilon = 0.05
    tau = 0.005

    max_buffer_size = 10000
    batch_size = 32
    replay_buffer = deque(maxlen=max_buffer_size)

    reward_hist = []
    cost_hist = []
    win_hist = []

    torch.set_default_dtype(torch.float)

    for i in range(10000):
    # for i in range(1):
        state = torch.tensor(env.reset(player) / 48.0).float().to(device)
        done = False
        total_reward = 0.0
        ave_cost = 0.0
        while not done:
            # choose action
            available_actions = torch.where(state[start:end] > 0)[0]
            if (torch.rand(1) < epsilon):
                action = available_actions[torch.randint(len(available_actions), (1,))].item() + start #random
            else:
                with torch.no_grad():
                    q_pred = model(state)

                q_values = torch.where(state[start:end] > 0 , 1, 0)*q_pred
                action = (torch.where(state[start:end] > 0, q_values, -math.inf)).argmax().item() + start #max

            # make the action
            next_state, reward, done = env.step(action, player)
            next_state = torch.tensor(next_state / 48.0).float().to(device)
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).float().unsqueeze(1).to(device)
                raw_next_states = torch.stack(next_states)
                dones = torch.tensor(dones).unsqueeze(1).to(device)

                mask = dones == 0
                next_states = torch.masked_select(raw_next_states, mask).reshape(-1,14)

                # make predictions
                q_preds = model(states).gather(1, actions)
                next_state_qs = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    next_state_qs[mask.squeeze()] = target_model(next_states).max(1).values
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
        if (i % 100 == 0):
            print(f'Episode {i:04d}, cost: {(sum(cost_hist[-100:]) / 100):.2f}, reward: {(sum(reward_hist[-100:]) / 100):.2f}, win rate: {(sum(win_hist[-100:]) / 100):.2f}, eps: {epsilon:.2f}')

    torch.save(model.state_dict(), f'output/{player}_{run}.pth')
    
    # get better data for graph
    cumulative_win_percentage = [sum(win_hist[max(0, i-100):i])/min(i,100) for i in range(1, len(win_hist) + 1)]
    cumulative_reward_hist = [sum(reward_hist[max(0, i-100):i])/min(i,100) for i in range(1, len(reward_hist) + 1)]

    return cost_hist, cumulative_reward_hist, cumulative_win_percentage




if __name__ == "__main__":
    player, run = 0, 2
    cost_hist, reward_hist, win_hist = train_model_pytorch(player, run)
    plot(cost_hist, model=f'cost_{player}_{run}')
    plot(reward_hist, model=f'reward_{player}_{run}')
    plot(win_hist, model=f'wins_{player}_{run}')

