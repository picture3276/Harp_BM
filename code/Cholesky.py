import torch as th

import time

import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  


dtype = th.float64

# gpuid = 0
# device = th.device("cuda:"+ str(gpuid))
device = th.device("cpu")

# print("Execution device: ",device)
# print("PyTorch version: ", th.__version__ )
# print("CUDA version: ", th.version.cuda)
# print("CUDA device:", th.cuda.get_device_name(gpuid))

# Batched vech2L input V is nb x n(n+1)/2
def bvech2L(V,nb,n):
    count = 0
    L = th.zeros((nb,n,n))
    for j in range(n):
        for i in range(j,n):
            L[...,i,j]=V[...,count]
            count = count + 1
    return th.tensor(L , device=device, dtype=dtype)


# Batched Cholesky decomp
def cholesky(A):
    L = th.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[...,i,k].clone() * L[...,j,k].clone()

            L[...,i,j] = th.sqrt(A[...,i,i] - s) if (i == j) else                       (1.0 / L[...,j,j].clone() * (A[...,i,j] - s))
    return L


# Batched inverse of lower triangular matrices
def inverseL(L):
    n = L.shape[-1]
    invL = th.zeros_like(L)
    for j in range(0,n):
        invL[...,j,j] = 1.0/L[...,j,j]
        for i in range(j+1,n):
            S = 0.0
            for k in range(i+1):
                S = S - L[...,i,k]*invL[...,k,j].clone()
            invL[...,i,j] = S/L[...,i,i]

    return invL


def test_loop(n,m):

    nn = int(n*(n+1)/2)

    th.manual_seed(42)
    X = th.rand((m,nn), device=device, dtype=dtype)
    L = th.add(bvech2L(X,m,n),th.tensor(th.eye(n), device=device, dtype=dtype))
    A = th.matmul(L,th.transpose(L, 1, 2))
    print("Shape of A {}".format(A.shape))

    start_time = time.time()

    cholA = th.zeros_like(A)
    for i in range(m):
        cholA[i,:,:] = th.potrf(A[i], upper=False)

    runtime = time.time() - start_time
    print("loop version took {} seconds ".format(runtime))
    return runtime


def test_batch(n=4,m=1000):

    nn = int(n*(n+1)/2)

    th.manual_seed(42)
    X = th.rand((m,nn), device=device, dtype=dtype)
    L = th.add(bvech2L(X,m,n), th.tensor(th.eye(n), device=device, dtype=dtype))
    A = th.matmul(L,th.transpose(L, 1, 2))
    print("Shape of A {}".format(A.shape))

    start_time = time.time()

    cholA = th.zeros_like(A)

    cholA = cholesky(A)

    runtime = time.time() - start_time
    print("batched version took {} seconds ".format(runtime))
    return runtime


n = 10
m = 10000
tv_l = test_loop(n,m)
tv_b = test_batch(n,m)
spu = tv_l / tv_b
print("The speed up is: ", spu)

