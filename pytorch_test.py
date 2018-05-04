import numpy as np
import torch.optim as optim

num_iterations = 5
N = 100000
D = 200

A = np.random.randn(N, D)
b = np.random.randn(N, 1)
# Y = np.random.rand(N, 1)

admm = optim.ADMM([A, b, 0.01, 1], "Lasso") #, parallel = True)

print(admm.getLoss())
for i in range(0, num_iterations):
    print('O Val:', admm.step())

print('Weights: ',admm.getParams())
