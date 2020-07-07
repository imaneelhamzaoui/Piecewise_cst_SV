# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:52:47 2020

@author: ielham
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:57:32 2020

@author: ielham
"""

import numpy as np 
from pyHaar import *
from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 
from numpy import linalg as LA

#%%

def thr1(Sr, Ntilde,dof, pvalue,J):
    #de taille J
    import scipy.special
    
    gm=np.zeros(J)
    
    gammaValue=np.sqrt(2)*scipy.special.gamma((dof+1.)/2.)/scipy.special.gamma(dof/2.)
    
    gstd=np.sqrt(dof-gammaValue**2)
    
    prodN=norm_rec(np.dot(Ntilde, np.diag(Sr)),J)
    
    for j in range(J):
        sigma_n=mad(prodN)
    
        sigmaF=(float(sigma_n))/float(gstd)
    
        gm[j]=np.median(stats.chi.ppf(pvalue, dof, scale=sigmaF))
    
    return(gm)
    
def thr2(Sr, sigma, dof, pvalue, J):
    #dee taille J
    gm=np.zeros((J))
    
    SdN=norm_rec(np.diag(Sr), J)+1e-16
    
    for j in range(J):
    
        gm[j]=np.median(stats.chi.ppf(pvalue, dof, scale=sigma*SdN[:,j]))
    
    return(gm)
    
def thr3(Sr, Ntilde,sigma, dof, pvalue, J):
    
    import scipy.special
    gm=np.zeros((J))
    g1=thr2(Sr, sigma, dof, pvalue,J)
    
    gammaValue=np.sqrt(2)*scipy.special.gamma((dof+1.)/2.)/scipy.special.gamma(dof/2.)
    
    gstd=np.sqrt(dof-gammaValue**2)
    
    prodN=norm_rec(np.dot(Ntilde, np.diag(Sr)),J)
   
    for j in range(J):
        if np.sum(prodN[:,j]>g1[j])>1:
            print(np.sum(prodN[:,j]>g1[j]))
     
            prodF=dp(prodN[prodN[:,j]>g1[j]])
            sigma_n=mad(prodF)
        
            sigmaF=(float(sigma_n))/float(gstd)
        
            gm[j]=np.maximum(stats.chi.ppf(pvalue, dof, scale=sigmaF), g1[j])
        
        else:
            gm[j]=dp(g1[j])
    
    return(gm)
    

def threshold_interm(Option,sigma,Si, Ai, Arefi, X, dof,  Weights,eps, stepg, pvalue, J):
    
    (n,t)=np.shape(Si)
    S=dp(Si)
    A=dp(Ai)
    Aref=dp(Arefi)
    
    Ntilde=X-np.sum(A*S.T, axis=2)
    
    nb_sources, nb_pix=np.shape(Si)
    
    gamma= stepg*1./np.max(Si**2, axis=1)
    
    seuilf=np.zeros((t, J, n))
    
    ww=np.zeros((t, J, n))
    
    for H in range(nb_sources):
        
        norme= norm_rec(A[:,:,H]-Aref[:,:,H], J)
        for j in range(J):
            normemax=np.max(norme[:,j])
            for k in range(t):
                ww[k,j,H]=eps/(eps+norme[k,j]/normemax)
        
        if Option==1:
            h=gamma[H]*thr1(S[H,:], Ntilde,dof, pvalue,J)
            
        elif Option==2:
            h=gamma[H]*thr2(S[H,:], sigma, dof, pvalue,J)
            
        elif Option==3:
            h=gamma[H]*thr3(S[H,:], Ntilde,sigma, dof, pvalue, J)       
        
        
        if Weights:
            for j in range(J):
                seuilf[:,j,H]=h[j]*ww[:,j, H]     
        else:
            for j in range(J):
                seuilf[:,j,H]=h[j]
    return(seuilf, ww)
    
def threshold_finalstep(Option,sigma,perc,Si, Ai, Arefi, X, dof,  Weights, stepg=.8, pvalue=.996,eps=1e-3, J=2):
    """
    Option : 
        1 : Threshold computed on the MAD operator of the norm of the propagated noise
        
        2 : Threshold based on the statistic of the inupt noise
        
        3 : Threshold based on the MAD operator of the residual of the noise 
        over the noise-dependent threshold
    """
    (nb_obs, nb_pix, nb_sources)=np.shape(Ai)
    
    seuil, ww=threshold_interm(Option,sigma,Si, Ai, Arefi, X, dof,  Weights,eps, stepg, pvalue, J)
    
    
    
    for H in range(nb_sources):
        norme= norm_rec(Ai[:,:,H]-Arefi[:,:,H], J)
        seuilT=dp(seuil[:,:, H])
        
        
        for j in range(J):
            
            normeR=norme[:,j]
            seuil_i=(seuilT[:,j])
            
            if Weights:
                indNZ = np.where(abs(normeR) -seuil_i/ww[:,j, H] > 0)[0]
            else:
                indNZ = np.where(abs(normeR) -seuil_i > 0)[0]
            if len(indNZ)==0:
                seuil[:,j,H]=dp(seuil_i)
                print('no elt detected')    
            else:
                I = abs(normeR[indNZ]).argsort()[::-1]
                Kval=np.int(np.floor(perc*len(indNZ)))
                if Kval>len(I)-1 or I[Kval]>len(indNZ)-1 :
                    seuil[:,j,H]=dp(seuil_i)
                    print('threshold source-dpdt only')
                else:
                    print('threshold based on the nbr of coefs', Kval, len(indNZ))
                    IndIX = np.int(indNZ[I[Kval]])
                    thr=abs(normeR[IndIX])
                    if Weights:
                        seuil[:,j,H]=ww[:,j,H]*dp(thr)
                    else:
                        seuil[:,j,H]=dp(thr)
                
    return(seuil)    
    


def mad(xin = 0):
    
    import numpy as np
    
    z = np.median(abs(xin - np.median(xin)))/0.6735
    
    return z 