from enum import Enum
import numpy as np


class Activation(Enum):
    RELU = 2
    SIGMOID = 1
    NONE = 0

class QLayer():
    def __init__(self, input_size : int, output_size : int, activation : Activation = Activation.NONE):
        self.output_size = output_size
        self.input_size = input_size
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.z = None
        self.a = None
        self.activation = activation
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activate(self.z)
        return self.a

    def activate(self, z : np.ndarray) -> np.ndarray:
        if (self.activation == Activation.RELU):
            return np.maximum(0,z)
        elif (self.activation == Activation.SIGMOID):
            return 1 / (1 + np.exp(-z))
        elif self.activation == Activation.NONE:
            return z