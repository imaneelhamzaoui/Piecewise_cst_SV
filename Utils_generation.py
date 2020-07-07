# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:15:43 2019

@author: ielham
"""

"""
Fichier pour creer des sources 
exactement ou approximativement sparse
en starlets
"""


import BSS_J as bssJ
import pyStarlet as ps
import numpy as np 
from scipy import fftpack as ft
from scipy import stats 
import numpy as np
import scipy as sp
import pyStarlet as ps
from copy import deepcopy as dp 

#%%
def Sources(t ,w1, ptot,ptot1=.2,  dynamic=2, n=2, J=3, Opt=1 , w2=1):
    '''
    Creation of the sources, exactly sparse in DCT. 
    Output: a n*t matrix, sparse in DCT.
    '''    
    if Opt==1:
        Sw=np.zeros((n, t, J+1))
        for l in range(J):
            S=np.zeros((n,t))  
            
            X,X0,A0,S,N,sigma_noise,kern=bssJ.Make_Experiment_Coherent(t_samp=t,ptot=ptot,w=w1,dynamic=dynamic)
            Sw[:,:, l]=dp(S)
            
        X,X0,A0,S,N,sigma_noise,kern=bssJ.Make_Experiment_Coherent(t_samp=t,ptot=ptot1,w=w2,dynamic=dynamic)
        Sw[:,:,-1]=dp(S)
        Su=ps.backward1d(Sw)
        
    else:
        Sw=np.zeros((n, t, J+1))
        for l in range(J):
            S=np.zeros((n,t))  
            
            X,X0,A0,S,N,sigma_noise,kern=bssJ.Make_Experiment_Coherent(t_samp=t,ptot=ptot,w=w1,dynamic=dynamic)
            Sw[:,:, l]=dp(S)
        Su=ps.backward1d(Sw)
        
    return Su
    
def XN(A, S, m=5, t=1024, noise_level=25):
    X0=np.sum(A*S.T, axis=2)
    N=np.random.randn(m,t) 
    
    sigma_noise = np.power(10.,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    N = sigma_noise*N
    
    X=X0+N
    
    return (X, N, sigma_noise)