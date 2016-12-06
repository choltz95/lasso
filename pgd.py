#Chester Holtz - choltz2@u.rochester.edu
#CSC 576 Advanced Machine Learning and Optimization, Professor Liu

import numpy as np
from numpy import linalg as la

MAX_ITER = 100

def pgd(A, b, lmd, max_iter=MAX_ITER):
    """
    Solves a problem of the form ||Ax - b||_2 ^2 + lambda ||w||_1 with proximal gradient descent
    """
    beta = np.zeros(A.shape[1])
    r = 0.00005

    fx_prev = 0

    x1 = np.zeros(p) # initialize to store
    fx_1 = []

    # use linear search to find r as step length
    A1 = np.dot(A, A.transpose())
    U, S, V = np.linalg.svd(A1)
    r = 1.0/S[0] # r = 1/||A A^T||

    index = 0
    for _ in range(MAX_ITER): # until convergence
	y1 = x1 - r * np.dot(A.transpose(), (np.dot(A, x1) - b)) #y_k = x_k - r * \delta f (x_k) = x_k - r * A' dot (Ax-b)
	x1 = np.multiply(np.sign(y1), np.maximum((np.absolute(y1) - lmd * r), 0)) #x_k+1 = sign(y_k) dot max(|y_k| - lambda*r, 0)
	fx1 = 0.5 * (la.norm((np.dot(A,x1) - b)))**2 + lmd * la.norm(x1, 1) #f_xk = 1/2 ||Axk - b||^2 + lmd ||xk||_1
	fx_1.append(fx1)

        print fx1

n = 250
p = 500
A = np.random.randn(n,p)
x = np.zeros(p)
lmbda = np.sqrt(2*n*np.log(p))
e = np.random.randn(n)
for i in range(0,20):
    k = np.random.randint(n)
    s = np.random.normal(0,10)
    x.itemset(k,s)

b = np.dot(A,x)+e
pgd(A,b,lmbda)
