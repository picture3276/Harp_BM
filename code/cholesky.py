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




n = 10
m = 10000
tv_l = test_loop(n,m)



