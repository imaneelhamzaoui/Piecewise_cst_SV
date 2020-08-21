# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:02:24 2020

@author: ielham
"""
import numpy as np
import pyStarlet as ps

def Init(kend,Xin, size_patch, rank, J_2D):
    
    (m,nx,ny)=np.shape(Xin)
    Af=np.zeros((m, nx, ny, rank))
    Sf=np.zeros((rank, nx, ny))
    nb_obs=m
    if nx!=ny:
        print('No square data matrix')
        
    nb_patch=nx/size_patch
    
    if size_patch*nb_patch!=nx:
        print('choose another size of patch')
    
    mw = ps.forward(Xin.reshape((nb_obs,nx,ny)),J=J_2D)
        
    '''ref'''
    Xref=[0,nx]
    Aref, Sref=GMCA_patch(kend,mw, Xref, Xref, 2)
    
    for i in range(nb_patch):
        for j in range(nb_patch):
            Xx=[i*size_patch , (i+1)*size_patch]
            Xy=[j*size_patch, (j+1)*size_patch]
            a, s=GMCA_patch(kend,mw,Xx, Xy, rank)
            temp1, temp2=perm(a, s, Sref[:,Xx[0]:Xx[1], Xy[0]:Xy[1]])
            temp_, Sf[:,Xx[0]:Xx[1], Xy[0]:Xy[1] ]=sign(temp1, temp2, Sref[Xx[0]:Xx[1], Xy[0]:Xy[1]])
            temp_2=np.repeat(temp_.reshape(m,1,rank), Xx[1]-Xx[0], 1)
            Af[:,Xx[0]:Xx[1], Xy[0]:Xy[1],: ]=np.repeat(temp_2.reshape(m,Xx[1]-Xx[0], 1,rank), Xy[1]-Xy[0], 2)
            
    return(Af, Sf, Aref, Sref)

def GMCA_global(kend,Xin, rank, J_2D):
    
    (m,nx,ny)=np.shape(Xin)
    mw = ps.forward(Xin.reshape((m,nx,ny)),J=J_2D)
        
    '''ref'''
    Xref=[0,nx]
    Aref, Sref=GMCA_patch(kend,mw, Xref, Xref, 2)
    
    
    return( Aref, Sref)
    

def GMCA_patch(kend,mw, Xx, Xy, rank):
    import AMCA_Direct as amca
    
    J_2D=np.shape(mw)[-1]-1
    nb_obs=np.shape(mw)[0]
    N=Xx[1]-Xx[0]
    Xp=mw[:,Xx[0]:Xx[1], Xy[0]:Xy[1],:]
    X1=Xp[:,:,:,:J_2D].reshape(nb_obs, (J_2D)*N**2)
    dS={}
    
    dS['n']=rank
    dS['m']=nb_obs
    dS['t']=N**2
    dS['kSMax']=kend
    dS['iteMaxXMCA']=400
    A, S=amca.AMCA(X1, dS, 0,1)
    SFinW = np.zeros((rank,N**2,J_2D+1))
    SFinW[:,:,0:J_2D] = S.reshape(rank,N**2,J_2D)
    
    SFinW[:,:,J_2D] = np.dot(np.linalg.pinv(A),Xp[:,:,:,J_2D].reshape(nb_obs, N**2))

    Sb = ps.backward1d(SFinW)

    Sc=np.reshape(Sb, (rank, N, N))
    
    return(A, Sc)
    
def perm(Aout, Sout, Sref):
    import numpy.linalg as LA
    from copy import deepcopy as dp
    
    Af=np.zeros((np.shape(Aout)))
    Sf=np.zeros((np.shape(Sout) ))
    s1=dp( abs(Sout[0,:,:]))
    
    if LA.norm(s1-abs(Sref[0,:,:]))<LA.norm(s1-abs(Sref[1,:,:])):
        
        Sf=dp(Sout)
        
        Af=dp(Aout)
    else:
        
        Sf[0,:,:]=dp(Sout[1,:,:])
        
        Sf[1,:,:]=dp(Sout[0,:,:])
        
        Af[:,0]=dp(Aout[:,1])
        
        Af[:,1]=dp(Aout[:,0])
        
    return (Af, Sf)
#%
def sign(A ,S, Sref):
    
    from copy import deepcopy as dp
    
    Af=np.zeros((np.shape(A)))
    Sf=np.zeros((np.shape(S) ))
    
    for i in range(np.shape(S)[0]):
        if np.sum(A[:,i]<0)==0:
            
            Sf[i,:,:]=dp(S[i,:,:])
            
            Af[:,i]=dp(A[:,i])
            
        else:
            
            Sf[i,:,:]=-dp(S[i,:,:])
            
            Af[:,i]=-dp(A[:,i])
        
    return (Af, Sf)
#%%