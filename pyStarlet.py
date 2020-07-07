
import sparse2d as sp2
import numpy as np

# pyStarlet

def forward(X,h = [0.0625,0.25,0.375,0.25,0.0625],J = 1):
    
    nX = np.shape(X)
    Lh = np.size(h)
    
    W = sp2.Starlet2D(nX[1],nX[2],nX[0],J,Lh).forward_omp(np.real(X),np.array(h))
    
    return W
    
    
def backward(W,h = [0.0625,0.25,0.375,0.25,0.0625]):
    
    nX = np.shape(W)
    Lh = np.size(h)
    
    rec = sp2.Starlet2D(nX[1],nX[2],nX[0],nX[3]-1,Lh).backward_omp(np.real(W))
    
    return rec
    
def forward1d(X,h = [0.0625,0.25,0.375,0.25,0.0625],J = 1):
    """
    
    soit x le vecteur qu'on veut transformer en starlet
    
    X=np.array(x).reshape((1, np.shape(x)))
    
    W : de taille 1*(taille de x)*J+1
    """
    nX = np.shape(X)
    Lh = np.size(h)
    
    W = sp2.Starlet2D(nX[1],1,nX[0],J,Lh).forward1d_omp(np.real(X),np.array(h))
    
    return W
    
def adjoint1d(W,h = [0.0625,0.25,0.375,0.25,0.0625]):
    
    nX = np.shape(W)
    Lh = np.size(h)
    
    W = sp2.Starlet2D(nX[1],1,nX[0],nX[2]-1,Lh).adjoint1d(np.real(W),np.array(h))
    
    return W
    
def backward1d(W,h = [0.0625,0.25,0.375,0.25,0.0625]):
    """
    W : de taille 1*t*J+1
    
    rec : de taille 1*t 
    """
    
    rec = np.sum(W,axis=2)
    
    return rec