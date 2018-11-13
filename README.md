# ADMM
Implemented ADMM for solving convex optimization problems such as Lasso, Ridge regression

## Introduction
Alternating Direction Method of Multiplier is framework for solving objecting function with divide-and-conquer approach.

ADMM works in two steps

  1. Divide <br> 
    a. Break down original problem into small problems <br>
    b. Distribute these small problem to <N> processors / computing resources <br>
    c. Every processor solves the smaller problem
  2. Conquer <br>
    a. Combine solution from \<N\> processors into one
    

## Our work
We implemented ADMM in PyTorch framework [Click here to view](https://github.com/bhushan23/pytorch/blob/admm/torch/optim/admm.py)
for Lasso and Ridge regression.

## Results
### Lasso Solver
|![Lasso Loss](https://github.com/bhushan23/ADMM/blob/master/results/Lasso.png)|![Lasso Prediction](https://github.com/bhushan23/ADMM/blob/master/results/Compare_Lasso.png)|
|:---:|:---:|
|*ADMM Lasso Loss*|*ADMM vs Scikit Lasso Solver*|
### Ridge regression
|![Ridge regression Loss](https://github.com/bhushan23/ADMM/blob/master/results/Ridge.png)|![Ridge regression Prediction](https://github.com/bhushan23/ADMM/blob/master/results/Compare_Ridge.png)|
|:---:|:---:|
|*ADMM Ridge regression Loss*|*ADMM vs Scikit ridge regression Solver*|
### ADMM vs Newton vs Gradient Descent
![Contour Plot of ADMM vs GD vs Newtons method for Lasso Problem](https://github.com/bhushan23/ADMM/blob/master/results/Contour_plot.png)

Contour plot does shows that ADMM reaches to the optimal solution fast and then takes smaller steps as it reaches to near to the solution. Hence, it confirms that ADMM is middle solution to many problems which can solve problems nearly as fast as newton and is not just restricted to quadratic problems.

## Issues
1. ADMM needs distributed infrastructure to scale to general problems
2. Gradient of individual small problems needs to be known in order to divide the problem
3. How to divide the problem into smaller problems?
    a. This is reason behind we need to manually devise the smaller problem and then scale for parallalization
    
## Slides 
[Click here to view Presentation](https://docs.google.com/presentation/d/1hp89fBR87ODlX0qzbC7akHV19GmTjg49pyUIqGAw5g8/edit?usp=sharing)

## Report
[Click here to view Report](https://github.com/bhushan23/ADMM/blob/master/REPORT_ADMM_IN_PYTORCH.pdf)

## Other resources 
1. My Convex Optimization assignments - https://github.com/bhushan23/Convex-Optimization
2. Reading material - By Professor Steven Boyd - http://web.stanford.edu/~boyd/admm.html
3. Implementation - By Niru Maheswaranathan - https://github.com/nirum/ADMM
4. General Convex optimization problems implementation - By Stanford Convex Optimization group - https://github.com/cvxgrp/cvxpy


