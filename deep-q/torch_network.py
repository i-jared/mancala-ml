from typing import List
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, layers: List):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def build_layers(layer_dims: List[int], activations: List[str]) -> List[nn.Module]:
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        layers.append(_choose_activation(activations[i]))
    return layers

def _choose_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {activation}")

