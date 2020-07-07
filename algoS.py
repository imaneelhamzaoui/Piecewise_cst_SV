# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:58:15 2020

@author: ielham
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 22:18:20 2019

@author: ielham
"""

import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
from copy import deepcopy as dp 
from scipy import fftpack as ft
from scipy import stats 
import pyStarlet as ps
import numpy as np
#%%
def lasso_direct(X, Ain, Sin, kend=3, stepgg=1., resol=3,lim=2e-6):
    """
    FISTA with sources sparse in starlets
    """
    S=dp(Sin)
    A=dp(Ain)
    n=np.shape(S)[0]
    (m,h,n)=np.shape(A)
    ee=np.zeros((h))
    for k in range(h):
       ee[k]=LA.norm(np.dot(A[:,k,:].T, A[:,k,:]), ord=2)
    L=np.max(ee)
    convS_sub=1
    
    t_=1
    y=S.copy()
    while convS_sub>lim :
        S_old=S.copy()
        diff=np.zeros((n,h))
        for k in range(h):
            diff[:,k]=np.dot(A[:,k,:].T, np.reshape((X-np.sum(A*y.T, axis=2))[:,k], (m,1)))[:,0]
        prod=y+stepgg/L*diff
        for i in range(n):
         
            S[i,:]=seuillage(prod[i,:],diff[i,:]*stepgg/L,kend)
            
        t=(1.+np.sqrt(1+4*(t_)**2))/2.
        y=S+(t_-1.)/(t)*(S-S_old)
        t_=t
        convS_sub=np.linalg.norm(S_old-S)/np.linalg.norm(S)
        print(convS_sub)
    return(S)

def lasso_star(X, Ain, Sin, kend=3, stepgg=1., resol=3,lim=2e-6):
    """
    FISTA with sources sparse in starlets
    """
    S=dp(Sin)
    A=dp(Ain)
    n=np.shape(S)[0]
    (m,h,n)=np.shape(A)
    ee=np.zeros((h))
    for k in range(h):
       ee[k]=LA.norm(np.dot(A[:,k,:].T, A[:,k,:]), ord=2)
    L=np.max(ee)
    convS_sub=1
    
    t_=1
    y=S.copy()
    while convS_sub>lim :
        S_old=S.copy()
        diff=np.zeros((n,h))
        for k in range(h):
            diff[:,k]=np.dot(A[:,k,:].T, np.reshape((X-np.sum(A*y.T, axis=2))[:,k], (m,1)))[:,0]
        prod=y+stepgg/L*diff
        for i in range(n):
            prodi=dp(prod[i,:])
            diffi=dp(diff[i,:])
            
            
            Si=seuillage_star(prodi, diffi*stepgg/L, kend, resol, h)
            S[i,:]=Si
            
        t=(1.+np.sqrt(1+4*(t_)**2))/2.
        y=S+(t_-1.)/(t)*(S-S_old)
        t_=t
        convS_sub=np.linalg.norm(S_old-S)/np.linalg.norm(S)
        print(convS_sub)
    return(S)
    

    
   
def seuillage_star(Sini,diffi,K,res, t, eps=1e-3):
    Si=dp(Sini)
    diff=dp(diffi)
    S_=ps.forward1d(Si.reshape(1, t), J=res)
    grad_=ps.forward1d(diff.reshape(1, t), J=res)
    ww=np.zeros((t))
    for j in range(res):
        thr=K*mad(grad_[0,:,j])
        valmax=np.max(abs(S_[0,:,j]))
        ww=eps/(eps+abs(S_[0,:,j])/valmax)
        S_[0,:,j]=softThres(S_[0,:,j], thr*ww)
        
    S_ret=ps.backward1d(S_.reshape(1, t, res+1))[0,:]
    return(S_ret)



#%%

def seuillage(Sini,diffi,K):
    
    S_=dp(Sini)
    grad_=dp(diffi)
    
    S_=softThres(S_, K*mad(grad_))

    return(S_)


def seuillage_weights(Sini,diffi,K):
    
    S_=dp(Sini)
    grad_=dp(diffi)
    
    thr=K*mad(grad_)
    WS=thr/(thr+np.abs(S_))
    S_ret=softThres(S_, WS*thr)

    return(S_ret)   
#%%
"""

Tools
"""

def mad(xin = 0):


    z = np.median(abs(xin - np.median(xin)))/0.6735

    return z        

def softThres(x,thres):
    
    return x*(np.array(np.abs(x)>thres,int))-np.sign(x)*thres*np.array(np.abs(x)>thres,int)
    

def pseudoinv(X, A):
    m=np.shape(X)[0]
    t=np.shape(X)[1]
    n=np.shape(A)[2]
    Xs= dp(X)
    S=np.zeros((n, t))
       
        
    piA=np.zeros((n,  t,m))
    for i in range (t):
        A1=A[:,i,:]
        Ra= np.dot(A1.T,A1)
        Ua, Sa, Va= np.linalg.svd(Ra)
        iRa=np.dot(Va.T, np.dot(np.diag(1./Sa),Ua.T))#pseudo-inverse de Ra
        piA[:,i,:]= np.dot(iRa, A1.T)#Pseudo-inverse de A
        
        S[:,i]=np.dot(piA[:,i,:], Xs[:,i])
              
    return(S)