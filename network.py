from layer import QLayer, Activation
from typing import List
import numpy as np


class QNetwork:
    """A simple neural network with a list of layers and a learning rate"""

    def __init__(
        self, layers: List[QLayer], learning_rate: float, regularization: float = None
    ):
        self.layers = layers
        self.regularization = regularization
        self.learning_rate = learning_rate

    def forward(self, a: np.ndarray) -> np.ndarray:
        """forward propagation of the network, returning the output of the last layer"""
        assert len(a) == self.layers[0].input_size
        for i in range(len(self.layers)):
            a = self.layers[i].forward(a)
        return a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Backward propagation of the network, updating the weights and biases of each layer"""
        gradient: np.ndarray = (y_pred - y_true) / y_pred.size
        for layer in reversed(self.layers):
            gradient = self.apply_gradient(gradient, layer)

    def apply_gradient(self, gradient: np.ndarray, layer: QLayer) -> np.ndarray:
        """Calculates the gradient of the loss function with respect to the weights and biases of the layer"""
        if layer.activation == Activation.RELU:
            gradient[layer.z <= 0] = 0
        elif layer.activation == Activation.SIGMOID:
            gradient *= layer.a * (1 - layer.a)
        elif layer.activation == Activation.NONE:
            pass

        layer.W -= (
            self.learning_rate * np.outer(layer.x, gradient)
            + (self.regularization * layer.W)
            if self.regularization is not None
            else 0
        )
        layer.b -= self.learning_rate * gradient

        return np.dot(layer.W, gradient)
