#Chester Holtz - choltz2@u.rochester.edu
#CSC 576 Advanced Machine Learning and Optimization, Professor Liu

import numpy as np
from numpy import linalg as la

MAX_ITER = 100

def cd(A, b, lmd):
    """
    Solves a problem of the form ||Ax - b||_2 ^2 + lambda ||w||_1 with cd
    """
    beta = np.zeros(A.shape[1])
    r = 0.00005
    for _ in range(MAX_ITER): # until convergence
        for i in range(A.shape[1]): # for each coordinate
            tmp = np.dot(np.delete(A,i,1),np.delete(beta,i))
            c = b - tmp
            A_i = A[:,i]
            a = la.norm(A_i)**2
            d = np.dot(A_i,c)
            beta[i] = np.sign(d/a) * np.maximum((np.absolute(d/a) - lmd * r),0)
        fx = 0.5 * (la.norm((np.dot(A,beta) - b)))**2 + lmd * la.norm(beta,1)
        
        print fx

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
cd(A,b,lmbda)
