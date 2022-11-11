from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
