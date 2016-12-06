#Chester Holtz - choltz2@u.rochester.edu
#CSC 576 Advanced Machine Learning and Optimization, Professor Liu

import numpy as np
from numpy import linalg as la

MAX_ITER = 100

def admm(A, b,lmd):
    """
    Solves a problem of the form ||Ax - b||_2 ^2 + lambda ||w||_1 with admm
    """ 
    m, n = A.shape
    A_t_A = A.T.dot(A)
    w, v = np.linalg.eig(A_t_A)

    x_hat = np.zeros(n)
    z_hat = np.zeros(n)
    u = np.zeros(n)

    # calculate regression co-efficient and stepsize
    r = np.amax(np.absolute(w))
    l_over_rho = np.sqrt(2*np.log10(n)) * r / 2.0
    rho = 1/r

    A_t_y = A.T.dot(b)
    Q = A_t_A + rho * np.identity(n)
    Q = np.linalg.inv(Q)
    Q_dot = Q.dot
    sign = np.sign
    maximum = np.maximum
    absolute = np.absolute

    for _ in xrange(MAX_ITER): # until convergence
        x_hat = Q_dot(A_t_y + rho*(z_hat - u)) # min x by posterior ols
        u = x_hat + u
        z_hat = sign(u) * maximum(0, absolute(u)-l_over_rho) # min by soft threshold
        u = u - z_hat # multiplier update
        
        f_x = la.norm((np.dot(A,x_hat) - b))**2 + lmd*la.norm(x_hat,1)
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

b = np.dot(A,x) + e
admm(A,b, lmbda)
