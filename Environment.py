class Environment:
    def __init__(self):
        self.board = [[[0 for i in range(8)] for j in range(8)] for j in range(2)]

    def perform(self, action):
        int x, y, z = actioin[0], action[1], action[2]
        self.board[z][y][x] += 1
        return reward(), terminal()

    def reward(self):
        if self.unlegal():
            return -999
        if self.linked():
            return 999
        return 0

    def unlegal(self):
        if sum(black) != sum(white):
            return true
        if 2 in self.board:
            return true
        return false

    def linked(self):
        return false

    def clear(self):
        return

