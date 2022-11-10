import copy
import torch



class Environment:
    def __init__(self):
        self.sDim = 8*8
        self.aDim = 8*8
        self.board = torch.zeros((8, 8))
    

    def reset(self):
        self.board *= 0
        return copy.deepcopy(self.board)


    def step(self, action):
        board = self.board.view(-1)
        board[action] += 1
        return self.feedback()


    def feedback(self):
        state2 = copy.deepcopy(self.board)

        unlegal = self.unlegal()
        linked = self.linked()

        reward = 0
        if linked:
            reward = 999
        if unlegal:
            reward = -999

        terminal = unlegal or linked

        return state2, reward, terminal


    def unlegal(self):
        if 2 in self.board:
            return True

        return False


    def linked(self):
        board = self.board

        for i in range(8):
            for j in range(4):
                if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] == board[i][j+4] != 0:
                    return True

        for j in range(8):
            for i in range(4):
                if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] == board[i+4][j] != 0:
                    return True

        for i in range(4):
            for j in range(4):
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == board[i+4][j+4] != 0:
                    return True

        for i in range(4, 8):
            for j in range(4):
                if board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == board[i-4][j+4] != 0:
                    return True

        return False
