class Environment:
    def __init__(self):
        self.board = [[[0 for i in range(8)] for j in range(8)] for j in range(2)]

    def perform(self, action):
        int x, y, z = actioin[0], action[1], action[2]
        self.board[z][y][x] += 1
        return self.feedback()

    def feedback(self):
        unlegal = self.unlegal()
        linked = self.linked()
        if unlegal:
            reward = -999
        if linked():
            reward = 999
        terminal = unlegal or linked
        return reward, terminal

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

