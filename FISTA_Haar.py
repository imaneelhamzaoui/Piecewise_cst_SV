# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:44:02 2020

@author: ielham
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:00:31 2020

@author: ielham
"""

import numpy as np 
from scipy import fftpack as ft
from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 

import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
import pyStarlet as ps
import initialization as pyl
import Utils_generation as utils  
import BSS_J as bssJ

from pyHaar import *


def FISTA(S, X, ggA, R,seuili, ite,lim=5e-10,stepg=.9, iteprox=6000, limprox=1e-8, J=2):
 
    import time
    time0 = time.time()
    
    seuil=dp(seuili)
    
    nb_pix=np.shape(S)[1]
    nb_obs=np.shape(X)[0]
    nb_sources=np.shape(S)[0]
    
    Aref=dp(ggA)
    r2=dp(Aref)
    Si=dp(S)
    
    L_constant=np.max(Si**2, axis=1)
    
    gamma=stepg*1./L_constant
    A=dp(R)
    t=1
    temp=dp(A)
    grad=dp(A)
    Aold=dp(A)
    tempold=dp(temp)
    ecartre=[] 
    
    for indx in range (ite):
        
        for l in range(nb_pix):
            produit=np.dot(temp[:,l,:],Si[:,l]) 
            residu=X[:,l]-produit
            grad[:,l,:]=np.dot(-residu.reshape((nb_obs,1)), Si[:,l].reshape((1, nb_sources)))
            r2[:,l,:]=temp[:,l,:]-gamma*grad[:,l,:]
        
           
        A=prox(r2, Aref, seuil, iteprox, lim=limprox, J=J)
 
        
        t0 = dp(t)
        t = (1. + np.sqrt(1. + 4. * t0 * t0 )) / 2.
        temp = A+ (t0 - 1.)  * (A- Aold)/t
        
        print(LA.norm(grad))
        
        er=np.max(LA.norm(A-Aold, axis=(0,2))/LA.norm(Aold, axis=(0,2)))
       
        Aold=dp(A)
        
        print("er :",er)
        
        elapsed_time=time.time() - time0
        
        ecartre.append(er)
        
        if er<lim :
            break
        
    return (A,r2,ecartre,elapsed_time)
#    
def prox(z, Aref, seuil_total, ite, lim, J):
    """
    Dykstra algorithm to compute prox
    """
    import time
    
    time0=time.time()
    (m,t,n)=np.shape(z)
    x=dp(z)
    p=np.zeros((m,t,n))
    q=np.zeros((m,t,n))
    xold=dp(x)
    
    yold=np.zeros((m,t,n))
    
    for U in range(ite):
        
        ytemp=x+p
        
        ytemp[ytemp<1e-16]=1e-16
        
        y=ytemp/LA.norm(ytemp, axis=0)
        
        p=dp(x+p-y)
        
        xtemp=y+q
        
        for H in range(n):
            
            normeH=norm_rec(xtemp[:,:,H]-Aref[:,:,H], J )+1e-32#de taille (t-5*res)
            
            red_fH,redH,raH=forward_UWT_pad(xtemp[:,:,H]-Aref[:,:,H], J )
            
            rseuil=dp(red_fH)
            
            seuil=seuil_total[:,:,H]
            
            for j in range(J):
                
                facteur=(normeH[:,j]>1e-16)*(1-(seuil[:,j]/(normeH[:,j])))*((1-(seuil[:,j]/(normeH[:,j])))>0)
            
                rseuil[:,:, j]=red_fH[:,:, j]*facteur
                
            
            rfin=backward_UWT_pad(rseuil, redH,raH)
            
            x[:,:,H]=rfin+Aref[:,:,H]
        
        
        q=dp(y+q-x)
        
        
        ez=np.sum(abs(abs(y)-abs(x)))/np.sum(abs(x+1e-16))
      
       
        if ez<lim:
            
            break
        
        
        else:
            xold=dp(x)
            yold=dp(y)  
    
    return(x)    
    



def mad(xin = 0):

    import numpy as np

    z = np.median(abs(xin - np.median(xin)))/0.6735

    return z      