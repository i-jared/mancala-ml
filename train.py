from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from typing import List, Tuple

from mancala import MancalaBoard


def plot(data: List[float], model: str = None):
    plt.plot(data)
    plt.xlabel("Iteration")
    plt.ylabel(model if model is not None else "Cost")
    plt.title(model if model is not None else "Cost")
    plt.savefig(f"{'cost' if model is None else model}.png")
    plt.show(block=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train_model_pytorch():
    # get device
    device = "mps"
    print(f"Using {device} device")

    # initialize model and optimizer
    model = NeuralNetwork().float().to(device)
    target_model = NeuralNetwork().float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    env = MancalaBoard()

    discount = 1.01
    epsilon = 1.0
    decay_rate = 0.9999
    min_epsilon = 0.05
    target_update_freq = 10

    max_buffer_size = 10000
    batch_size = 32
    replay_buffer = deque(maxlen=max_buffer_size)

    reward_hist = []
    cost_hist = []
    win_hist = []

    torch.set_default_tensor_type(torch.FloatTensor)

    for i in range(100000):
        state = torch.tensor(env.reset() / 48.0).float().to(device)
        done = False
        total_reward = 0.0
        ave_cost = 0.0
        while not done:
            # choose action
            available_actions = torch.where(state[:6] > 0)[0]
            if (torch.rand(1) < epsilon):
                action = available_actions[torch.randint(len(available_actions), (1,))].item() # random
            else:
                with torch.no_grad():
                    q_pred = model(state)
                action = (torch.where(state[:6] > 0 , 1, 0)*q_pred).argmax().item() # max

            # make the action
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state / 48.0).float().to(device)
            deque.append((state, action, reward, next_state, done))
            total_reward += reward

            if len(deque) >= batch_size:
                batch = random.sample(deque, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards).unsqueeze(1).to(device)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones).unsqueeze(1).to(device)

                # make predictions
                q_preds = model(states).gather(1, actions)
                next_state_qs = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    # TODO: add a mask here...
                    next_state_qs = target_model(next_states).max(1).values
                q_actuals = rewards + discount * next_state_qs

                loss_fn = nn.SmoothL1Loss()
                loss = loss_fn(q_preds, q_actuals.unsqueeze(1))
                ave_cost += loss.item()
                optimizer.zero_grad()
                loss.backward()
                # TODO: implement policy_net
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

            state = next_state

        if i % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        epsilon = max(epsilon * decay_rate, min_epsilon)
        win_hist.append(env.board[6] > env.board[13])
        reward_hist.append(total_reward)
        cost_hist.append((ave_cost / env.turn))
        if (i % 100 == 0):
            print(f'Episode {i:04d}, cost: {(sum(cost_hist[-100:]) / 100):.2f}, reward: {(sum(reward_hist[-100:]) / 100):.2f}, win rate: {(sum(win_hist[-100:]) / 100):.2f}, eps: {epsilon:.2f}')

    torch.save(model.state_dict(), 'model.pth')
    return cost_hist, reward_hist, win_hist

if __name__ == "__main__":
    cost_hist, reward_hist, win_hist = train_model_pytorch()
    plot(cost_hist, model='cost')
    plot(reward_hist, model='reward')
    plot(win_hist, model='wins')
