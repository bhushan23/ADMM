import numpy as np
from admm import ADMM


num_iterations = 20
N = 100
D = 20

A = np.random.randn(N, D)
b = np.random.randn(N, 1)
# Y = np.random.rand(N, 1)

admm = ADMM(A, b, parallel = True)

print(admm.LassoObjective())
for i in range(0, num_iterations):
    admm.step()
    print('O Val:', admm.LassoObjective())
