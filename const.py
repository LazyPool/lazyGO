from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

gamma = 0.99
lr = 0.02
betas = (0.9, 0.999)
    
random_seed = 543

EPISODE = 3000
STEPS = 100
