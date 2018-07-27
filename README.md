# ADMM
Implemented ADMM for solving convex optimization problems such as Lasso, Ridge regression

## Introduction
Alternating Direction Method of Multiplier is framework for solving objecting function with divide-and-conquer approach.

ADMM works in two steps

  1. Divie 
    a. Break down original problem into small problems
    b. Distribute these small problem to <N> processors / computing resources
    c. Every processor solves the smaller problem
  2. Conquer
    a. Combine solution from <N> processors into one
    

## Our work
We implemented ADMM in PyTorch framework [Click here to view](https://github.com/bhushan23/pytorch/blob/admm/torch/optim/admm.py)
for Lasso and Ridge regression.

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
1. Reading material - By Professor Steven Boyd - http://web.stanford.edu/~boyd/admm.html
2. Implementation - By Niru Maheswaranathan - https://github.com/nirum/ADMM
3. General Convex optimization problems implementation - By Stanford Convex Optimization group - https://github.com/cvxgrp/cvxpy


