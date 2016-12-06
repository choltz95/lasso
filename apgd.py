#Chester Holtz - choltz2@u.rochester.edu
#CSC 576 Advanced Machine Learning and Optimization, Professor Liu

import numpy as np
from numpy import linalg as la

MAX_ITER = 100

def apgd(A, b, lmd, max_iter=MAX_ITER):
    """
    Solves a problem of the form ||Ax - b||_2 ^2 + lambda ||w||_1 with apgd
    """
    beta = np.zeros(A.shape[1])
    r = 0.00005

    fx_prev = 0

    x = np.zeros(p)
    y = np.zeros(p)
    t = 1
    fx = []

    # use linear search to find r as step length
    A1 = np.dot(A, A.transpose())
    U, S, V = np.linalg.svd(A1)
    r = 1.0/S[0] #r = 1/||A A^T||

    index = 0
    for _ in range(MAX_ITER): # until convergence
        z = x - r * np.dot(A.T, np.dot(A,y) - b)
        x_prev = x
        x =  np.multiply(np.sign(z), np.maximum(np.absolute(z) - lmd * r, 0))
        t_prev = t
        t = 0.5+0.5*np.sqrt(1+4*t**2)
        y = x + (t_prev-1)/t*(x-x_prev) 
        f_x = 0.5*(la.norm(np.dot(A,x_prev)-b))**2 + lmd*la.norm(x_prev,1)

	print f_x


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
apgd(A,b,lmbda)
