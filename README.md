# Stochastic variance reduced algorithms : Implementation of SVRG and SAGA optimization algorithms for deep learning.

Stochastic gradient (SGD) is the most used optimization algorithm to backpropagate through neural networks because it has a smaller cost than gradient descent. However, it has a very slow convergence rate and it needs to have a decreasing learning rate to converge.

In 2013 and 2014, two new 'hybrid algorithms' came out. Stochastic Variance Reduced Gradient (SVRG) and Stochatic Average Gradient Augmented (SAGA) are hybrids because they use an unbiaised estimate of the gradient but they have a vanishing variance. These 2 algorithms have an exponentiel convergence speed. While their behaviour is weel known for machine learning purpose, they are not used for deep learning topics. For instance, you can now use SAGA in the scikit learn library (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

For this project, I wanted to code these algorithms for deep learning purpose. I have coded these algorithms with the PyTorch framework.

Usage : 
This code is written in Python3 and I used jupyter notebook. You will need to download the following libraries : numpy, PyTorch, matplotlib, copy and time. Note that I tested my code on the CIFAR10 dataset :). I also tried it for MNIST and SVRG worked fine.

Currently, SAGA works for a batch of size 1. An extension would be to adapt it for any batch size.

SVRG paper : https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf

SAGA paper : https://arxiv.org/pdf/1407.0202.pdf

To start with PyTorch : http://pytorch.org/tutorials/

I  worked on this project for my data science course at UC Berkeley. 
