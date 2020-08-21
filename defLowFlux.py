# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:07:46 2020

@author: ielham
"""

import scipy.io as sio

from astropy.io import fits
from astropy.visualization import simple_norm

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from copy import deepcopy as dp 
import GMCA_star as gStar
#%%

from scipy import stats 
def mad(xin = 0):
    
    import numpy as np
    
    z = np.median(abs(xin - np.median(xin)))/0.6735
    
    return z 

def defXcut(X, pvalue, nb_obs, most_rest=1):

    sigma=mad(X)
    g=stats.chi.ppf(pvalue, nb_obs, scale=sigma)
    Emap=LA.norm(X, axis=0)
    plt.imshow(Emap), plt.colorbar()

    C=dp(Emap>g)

    Cx=LA.norm(C, axis=0)
    nX1=np.argwhere(Cx==Cx[Cx>0][0])[0][0]
    nX2=np.argwhere(Cx==Cx[Cx>0][-1])[-1][0]
    
    Cy=LA.norm(C, axis=1)
    nY1=np.argwhere(Cy==Cy[Cy>0][0])[0][0]
    nY2=np.argwhere(Cy==Cy[Cy>0][-1])[-1][0]

    if most_rest:
        nX=np.maximum(nX1, len(Cx)-nX2)
        nY=np.maximum(nY1, len(Cx)-nY2)

        nCut=np.maximum(nX, nY)
    
    else:
        nX=np.minimum(nX1, len(Cx)-nX2)
        nY=np.minimum(nY1, len(Cx)-nY2)

        nCut=np.minimum(nX, nY)        


    Xcut=dp(X[:,nCut:-nCut, nCut:-nCut])
    
    return(Xcut, nCut)
    
def defXcut_thres(X, thres, nb_obs, most_rest=1):
    
    Emap=LA.norm(X, axis=0)
    plt.imshow(Emap), plt.colorbar()

    C=dp(Emap>thres)

    Cx=LA.norm(C, axis=0)
    nX1=np.argwhere(Cx==Cx[Cx>0][0])[0][0]
    nX2=np.argwhere(Cx==Cx[Cx>0][-1])[-1][0]
    
    Cy=LA.norm(C, axis=1)
    nY1=np.argwhere(Cy==Cy[Cy>0][0])[0][0]
    nY2=np.argwhere(Cy==Cy[Cy>0][-1])[-1][0]

    if most_rest:
        nX=np.maximum(nX1, len(Cx)-nX2)
        nY=np.maximum(nY1, len(Cx)-nY2)

        nCut=np.maximum(nX, nY)
    
    else:
        nX=np.minimum(nX1, len(Cx)-nX2)
        nY=np.minimum(nY1, len(Cx)-nY2)

        nCut=np.minimum(nX, nY)        


    Xcut=dp(X[:,nCut:-nCut, nCut:-nCut])
    
    return(Xcut, nCut)
#%%
    
def permu_gl_deux(A_glob, S_glob, A_patch ):
    '''
    allows to corrrect permutations btwn global and local results
    Ouput:
    (A_g2, Sf): global (A,S) couple corrected with A of size(m*nx*ny*rank)
    '''
    from copy import deepcopy as dp 
    import numpy.linalg as LA
    
    temp=dp(A_glob)
    tempS=dp(S_glob)
    Af=dp(A_glob)
    Sf=dp(S_glob)
    
    (nb_obs, rank)=np.shape(A_glob)
    (rank, nx, ny)=np.shape(S_glob)
    
    T0=LA.norm(A_patch[:,0]-temp[:,0])
    T1=LA.norm(A_patch[:,0]-temp[:,1])
    
    if T0>T1:
        Af[:,0]=dp(temp[:,1])
        Af[:,1]=dp(temp[:,0])
        Sf[0,:]=dp(tempS[1,:])
        Sf[1,:]=dp(tempS[0,:])
    
    A_gg=np.repeat(Af.reshape(nb_obs,1,rank), nx, 1)
    A_g2=np.repeat(A_gg.reshape(nb_obs,nx, 1,rank), ny, 2)
   
    return(A_g2, Sf)