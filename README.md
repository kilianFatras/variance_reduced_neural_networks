# Stochastic variance reduced algorithms : Implementation of SVRG and SAGA optimization algorithms for deep learning.

Stochastic gradient (SGD) is the most optimization algorithm used to backpropagate through neural networks because it has a smaller cost than gradient descent. However, it has a very slow convergence rate and it needs to have a decreasing learning rate.

In 2013 and 2014, two new 'hybrid algorithms' came out. Stochastic Variance Reduced Gradient (SVRG) and Stochatic Average Gradient Augmented (SAGA) are hybrids because they use an unbiaised estimate of the gradient but they have a vanishing variance. This 2 algorithms have an exponentiel convergence speed.

I have coded these algorithm with the PyTorch library.

Usage : 
This code is written in Python3 and I used jupyter notebook. You will need to download the following library : numpy, PyTorch, matplotlib, copy and time. 

SVRG paper : https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf

SAGA paper : https://arxiv.org/pdf/1407.0202.pdf

To start with PyTorch : http://pytorch.org/tutorials/

I  worked on this project for my data science course at UC Berkeley. 
