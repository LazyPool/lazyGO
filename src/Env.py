import torch
import copy

class Environment:
    def __init__(self):
        self.board = torch.zeros((2, 8, 8))

    def perform(self, action):
        # place the chess
        z, y, x = action[0], action[1], action[2]
        self.board[z][y][x] += 1

        # return the reward and terminal
        return self.feedback()

    def feedback(self):
        # judge if unlegal or linked
        unlegal = self.unlegal()
        linked = self.linked()

        # caculate reward
        reward = 0
        if unlegal:
            reward = -999
        if linked:
            reward = 999

        # judge if terminal
        terminal = unlegal or linked

        # return result
        return reward, terminal

    def unlegal(self):
        # statistics
        count1 = torch.sum(self.board, 0)
        count2 = torch.sum(self.board, (1,2))
        
        # if a chess conflict
        if 2 in count1:
            return True

        # if order wrong
        delta = count2[0] - count2[1]
        if delta < 0 or delta > 1:
            return True

        # if legal
        return False

    def linked(self):
        board = self.board[0] - self.board[1]

        # judge row
        for i in range(8):
            for j in range(4):
                if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] == board[i][j+4] != 0:
                    return True

        # judge col
        for j in range(8):
            for i in range(4):
                if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] == board[i+4][j] != 0:
                    return True

        # judge left>right
        for i in range(4):
            for j in range(4):
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == board[i+4][j+4] != 0:
                    return True

        # judge left<right
        for i in range(4, 8):
            for j in range(4):
                if board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == board[i-4][j+4] != 0:
                    return True

        # not linked
        return False

    def clear(self):
        self.board *= 0

    def getState(self):
        return copy.deepcopy(self.board.flatten())
