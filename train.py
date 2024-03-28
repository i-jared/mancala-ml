from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
    reward_hist = []
    cost_hist = []
    win_hist = []
    discount = 1.01
    epsilon = 1.0
    decay_rate = 0.9999
    min_epsilon = 0.05
    target_update_freq = 10

    torch.set_default_tensor_type(torch.FloatTensor)

    for i in range(100000):
        state = torch.tensor(env.reset() / 48.0).float().to(device)
        done = False
        total_reward = 0.0
        ave_cost = 0.0
        while not done:
            # choose action
            q_pred = model(state)
            available_actions = torch.where(state[:6] > 0)[0]
            if (torch.rand(1) < epsilon):
                action = available_actions[torch.randint(len(available_actions), (1,))].item() # random
            else:
                action = (torch.where(state[:6] > 0 , 1, 0)*q_pred).argmax().item() # max

            # make the action
            state, reward, done = env.step(action)
            state = torch.tensor(state / 48.0).float().to(device)
            total_reward += reward

            # calculate the loss
            next_state_vals = torch.zeros(6, device=device)
            with torch.no_grad():
                next_state_vals[action] = target_model(state).max(0).values
            q_actual = reward + discount * next_state_vals
            mask = torch.zeros(6, device=device)
            mask[action] = 1
            loss_fn = nn.SmoothL1Loss()
            loss = (loss_fn(q_pred * mask, q_actual) * mask).sum()
            ave_cost += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
