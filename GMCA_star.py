# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:53:56 2020

@author: ielham
"""

from astropy.io import fits
from astropy.visualization import simple_norm

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from copy import deepcopy as dp 
import AMCA_Direct as amca
import pyStarlet as ps

def GMCA_star_V1(Xin, J_2D, rank):
    '''X dee taillee (m, N, N)'''

    m=np.shape(Xin)[0]
    N=np.shape(Xin)[1]

    mw = ps.forward(Xin.reshape((m,N,N)),J=J_2D)
    X = mw[:,:,:,J_2D].reshape(m,N**2)
    for j in range(J_2D):
        X = np.hstack([X,mw[:,:,:,j].reshape(m,N**2)])
    
    dS={}
    dS['n']=rank
    dS['m']=m
    dS['t']=N**2
    dS['kSMax']=3
    dS['iteMaxXMCA']=400
    A, S=amca.AMCA(X, dS, 0,1)
    
    Srec=np.zeros((rank, N**2))
    for j in range(J_2D+1):
        Srec += S[:,j*N**2:(j+1)*N**2]
#        piS = np.dot(Srec.T,np.linalg.inv(np.dot(Srec,Srec.T)))
#    Arec = np.dot(Xin.reshape(m, N**2),piS)
    
    return(A, Srec)
#%%

def GMCA_star_V2(Xin, J_2D, rank):
    '''X dee taillee (m, N, N)'''

    m=np.shape(Xin)[0]
    N=np.shape(Xin)[1]
    mw = ps.forward(Xin.reshape((m,N,N)),J=J_2D)
    X = mw[:,:,:,:J_2D].reshape(m,J_2D*N**2)

    
    dS={}
    dS['n']=rank
    dS['m']=m
    dS['t']=N**2
    dS['kSMax']=3
    dS['iteMaxXMCA']=400
    A, S=amca.AMCA(X, dS, 0,1)
    
    SFinW = np.zeros((rank,N**2,J_2D+1))
    SFinW[:,:,0:J_2D] = S.reshape(rank,N**2,J_2D)
  
    SFinW[:,:,J_2D] = np.dot(np.linalg.pinv(A),mw[:,:,:,J_2D].reshape(m, N**2))
    
    Sb = ps.backward1d(SFinW)
    #
    Sc=np.reshape(Sb, (rank, N, N))
    return(A, Sc)

    
#%%
