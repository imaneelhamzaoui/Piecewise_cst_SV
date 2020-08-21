# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:05:00 2019

@author: ielham
"""

"""
Init

"""
#import AMCA_Direct as amcaD
#import AMCA_Starlet as amcaS
import numpy as np

import numpy.linalg as LA
from copy import deepcopy as dp



#        Aref, Sref=amca.AMCA(X, dS, aMCA=dPatch['aMCA'])
#    
#    n_patches = np.int(np.floor(dS['t']/dPatch['PatchSize']))
#
#    t=dS['t']
#    m=dS['m']
#    n=dS['n']
#    
#    
#    
#    Aout = np.zeros((m,t,n))
#    Sout = np.zeros((n,t))
#
#    for p in range(n_patches):
#
#        ts = p*PatchSize
#        te = (p+1)*PatchSize
#
#        Xb = X[:,ts:te]
#
#        temp1, temp2 = amca.AMCA(Xb, dS, aMCA=1)
#        
#        #temp3, temp4 = perm(temp1, temp2, Sref[:,ts:te], dS)
#        
#        gA_g,gS_g = sign(temp1, Aref,temp2)
#
#        Sout[:,ts:te] = gS_g
#
#        for r in range(ts,te):
#            Aout[:,r,:] = gA_g
#
#        
#
#    if te < t:
#
#        Xb = X[:,te:t]
#        
#        temp1, temp2 = amca.AMCA(Xb, dS, aMCA=1)
#            
#        #temp3, temp4 = perm(temp1, temp2, Sref[:,ts:te], dS)
#            
#        gA_g,gS_g = sign(temp1,Aref, temp2)
#        
#        Sout[:,te:t] = gS_g
#        for r in range(te,t):
#            Aout[:,r,:] = gA_g

#==============================================================================
# #%%
# def perm(A, S, Sref, dS):
# 
#     S_p=abs(S)
#     
#     Sref_p=abs(Sref)
#     
#     Sf=np.zeros((np.shape(S)))
#     Af=np.zeros((np.shape(A)))
#     
#     if dS['n']==2:
#         
#         if LA.norm(S_p[0,:]-Sref_p[0,:]) < LA.norm(S_p[0,:]-Sref_p[1,:]):
#             
#             Sf=dp(S)
#             Af=dp(A)
#             
#         else:
#             
#             Sf[0,:] = dp(S[1,:])
#             
#             Sf[1,:] = dp(S[0,:])
#             
#             Af[0,:] = dp(A[1,:])
#             
#             Af[1,:] = dp(A[0,:])
#             
#             
#     return (Af,Sf)
# 
#==============================================================================
############################################################
################# STARLET-LIKE FILTERING
############################################################

def length(x=0):

    l = np.max(np.shape(x))
    return l


################# 1D convolution with the "a trous" algorithm

def Apply_H1_FM(x=0,h=0,scale=1):

    m = length(h)
    if scale > 0:
        p = (m-1)*np.power(2,(scale-1)) + 1
        g = np.zeros( p)
        z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
        g[z.astype(int)] = h
    else:
        g = np.copy(h)

    y = filter_1d_FM(x,g)

    return y

def Filter1D(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1):

    import copy as cp
    c = cp.copy(x)
    cnew = cp.copy(x)
    for scale in range(J):
        cnew = Apply_H1_FM(c,h,scale)
        c = cp.copy(cnew)
    return c


def RotMatrix(theta):

    """
     Rotation matrix
    """

    M = np.zeros((2,2))
    M[0,0] = np.cos(theta)
    M[1,1] = np.cos(theta)
    M[0,1] = np.sin(theta)
    M[1,0] = -np.sin(theta)

    return M

def Exp_Sn(xref,v,theta):

    """
     Exp-map of the n-sphere
    """

    m = len(xref)
    F = np.zeros((m,2))
    F[:,0] = xref
    F[:,1] = v
    F = np.dot(F,RotMatrix(-theta))

    return F[:,0]

def Log_Sn(xref,x):

    """
     Log-map of the n-sphere
    """

    m = len(x)

    G = np.zeros((m,))
    Gv = np.zeros((m,))

    a = np.sum(x*xref)/np.sqrt(np.sum(xref**2)*np.sum(x**2))

    if a > 1:
        a = 1
    if a < -1:
        a = -1

    G = np.arccos(a)  # Computing the angles

    v = x - a*xref
    Gv = v / (1e-24 + np.linalg.norm(v))   # Unit vector in the tangent subspace

    return G,Gv


#
#
#

def Grad(w,x,X):

    [m,T]=np.shape(X)
    g = np.zeros((m,))

    for t in range(T):
        gt,gv = Log_Sn(x,X[:,t])
        g = g - w[t]*gt*gv

    return g

def FrechetMean(X,u = None,w=None,t = None,itmax = 100,tol=1e-12):

    [m,T]=np.shape(X)
    g = np.zeros((m,))

    if u is None:
        u = np.random.rand(m,)
    if t is None:
        t = 1. # gradient path length
    if w is None:
        w = np.ones((m,))
        w = w / np.sum(w)

    for it in range(itmax):

        g = Grad(w,u,X)
        theta = np.linalg.norm(g)
        gv = g/(1e-6+theta)
        u_new = Exp_Sn(u,-gv,t*theta)
        diff = np.linalg.norm(u - u_new)
        u = np.copy(u_new)

        if diff < tol:
            break

    return u

#
#
#

def filter_1d_FM(xin=0,h=[0.0625,0.25,0.375,0.25,0.0625]): # h being the weights

    import numpy as np
    import scipy.linalg as lng
    import copy as cp

    x = np.squeeze(cp.copy(xin));
    [t,n] = np.shape(x);
    m = len(h);
    y = cp.copy(x);

    z = np.zeros((t,m));

    m2 = np.int(np.floor(m/2))
   
    
    for r in range(m2):

        u = x[:,0:m-(r+m2)-1];
        u = u[:,::-1]
        z = np.concatenate([u,x[:,0:r+m2+1]],axis=1)
        zr =np.mean(z,axis=1)
        zr = zr/np.linalg.norm(zr)

        #y[:,r] = np.sum(z*h,axis=1) # SHOULD USE THE FRECHET MEAN

        y[:,r] = FrechetMean(z,u=zr,w=h,itmax = 1000,tol=1e-6)

    a = np.arange(np.int(m2),np.int(n-m+m2),1)

    for r in a:

        #y[:,r] = np.sum(x[:,r-m2:m+r-m2]*h,axis=1)
        zr = np.mean(x[:,r-m2:m+r-m2],axis=1)
        zr = zr/np.linalg.norm(zr)

        y[:,r] = FrechetMean(x[:,r-m2:m+r-m2],u=zr,w=h,itmax = 100,tol=1e-6)

    a = np.arange(np.int(n-m+m2+1)-1,n,1)

    for r in a:

        u = x[:,n - (m - (n-r) - m2 -1)-1:n]
        u = u[:,::-1]
        z = np.concatenate([x[:,r-m2:n],u],axis=1)
        zr = np.mean(z,axis=1)
        zr = zr/np.linalg.norm(zr)

        #y[:,r] = np.sum(z*h,axis=1)
        y[:,r] = FrechetMean(z,u=zr,w=h,itmax = 100,tol=1e-6)

    return y

