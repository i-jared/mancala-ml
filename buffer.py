import random


class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []

    def size(self):
        return len(self.data)
    
    def getRandomBatch(self, batch_size):
        return random.sample(self.data, batch_size)

    def add(self, value):
        if len(self.data) < self.capacity:
            self.data.append(value)
        else:
            self.data.pop(0)
            self.data.append(value)

