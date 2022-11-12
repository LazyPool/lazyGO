import numpy as np
import random
from collections import deque


class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxsize = size
        self.len = 0

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s1Arr = np.float32([arr[0] for arr in batch])
        a1Arr = np.float32([arr[1] for arr in batch])
        r1Arr = np.float32([arr[2] for arr in batch])
        s2Arr = np.float32([arr[3] for arr in batch])

        return s1Arr, a1Arr, r1Arr, s2Arr

    def len(self):
        return self.len

    def add(self, s1, a1, r1, s2):
        transition = (s1, a1, r1, s2)
        self.len += 1
        if self.len == self.maxsize:
            self.len = self.maxsize
        self.buffer.append(transition)
