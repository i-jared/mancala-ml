import torch
from agents import HumanAgent, RandomAgent
from mancala import MancalaBoard
from torch_network import NeuralNetwork


def test_model_pytorch_human():
    model = NeuralNetwork()
    model.load_state_dict(torch.load('output/model.pth'))
    human = HumanAgent()

    env = MancalaBoard()
    state = torch.tensor(env.reset() / 48.0).float()
    done = False
    player = 0

    while not done:
        if player == 0:
            with torch.no_grad():
                q_pred = model(state)
            action = (torch.where(state[:6] > 0 , 1, 0)*q_pred).argmax().item() # max
            state, player, done = env.step_test(action, player)
            state = torch.tensor(state / 48.0).float()
        else:
            action = human.decide(env.board)
            state, player, done = env.step_test(action, player)
            state = torch.tensor(state / 48.0).float()
        print(env)

def test_model_pytorch_random():
    model = NeuralNetwork()
    model.load_state_dict(torch.load('output/model.pth'))
    env = MancalaBoard()
    robot = RandomAgent(1)
    wins = []

    ## Test going first
    for _ in range(1000):
        player = 0
        done = False
        state = torch.tensor(env.reset() / 48.0).float()
        while not done:
            if player == 0:
                with torch.no_grad():
                    q_pred = model(state)
                action = (torch.where(state[:6] > 0 , 1, 0)*q_pred).argmax().item() # max
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
            else:
                action = robot.decide(env.board)
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
        wins.append(env.board[6] > env.board[13])

    print(f"win percentage (going first): {sum(wins) / len(wins)}")


    ## Now test going second
    robot = RandomAgent(0)
    wins = []

    for _ in range(1000):
        player = 0
        done = False
        state = torch.tensor(env.reset() / 48.0).float()
        while not done:
            if player == 1:
                with torch.no_grad():
                    q_pred = model(torch.cat((state[7:], state[:7])))

                action = (torch.where(state[7:13] > 0 , 1, 0)*q_pred).argmax().item() # max
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
            else:
                action = robot.decide(env.board)
                state, player, done = env.step_test(action, player)
                state = torch.tensor(state / 48.0).float()
        wins.append(env.board[6] > env.board[13])
    print(f"win percentage (going second): {sum(wins) / len(wins)}")

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
    test_random()

