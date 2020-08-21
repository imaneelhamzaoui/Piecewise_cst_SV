# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 20:44:12 2020

@author: ielham
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:49:57 2020

@author: ielham
"""

import numpy as np

from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 
import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
import pyStarlet as ps


import generation_physpecChandra as gen


from astropy.io import fits

import fct_gen_Chandra as genChan
import MonInit as In
#%%

nb_obs=10
nb_pix=128
nb_sources=2
(m,t,n)=(nb_obs, nb_pix, nb_sources)
SNR=65


'''spectra of reference'''
spec1=gen.Emission_law(m=nb_obs,pos=0.15,width=0.1)
spec2=gen.Emission_law(m=m,pos=0.8,width=0.1)



A2=np.zeros((m,t,t,n))
A2[:,:,:,1]=genChan.get_VS(2,spec1,R0=64,xcen=63, ycen=62, nx=t, ny=t)

A2[:,:,:,0]=genChan.get_VS(1,spec2,R0=64,xcen=63, ycen=62, nx=t, ny=t)

'''Normalization of the global mixing matrix''' 
for r in range(n):
    for l in range(m):
        a=np.where(A2[l,40,:,r]!=0)[0][0]
        A2[l,A2[l,:,:,r]==0,r]=dp(A2[l,60,a,r])

#
'''Sources resized'''
im_Fe1=LA.norm(fits.getdata('cube_ref_Fe-shift1.fits'),axis=0).T
im_Fe2=LA.norm(fits.getdata('cube_ref_Fe-shift2.fits'),axis=0).T

Se=np.zeros((n, t, t))
#%
f=sp.interpolate.interp2d(range(128), range(128), im_Fe1)
Se[0, :,:]=f(np.linspace(0, 128, t),np.linspace(0, 128, t))

f=sp.interpolate.interp2d(range(128), range(128), im_Fe2)
Se[1,:,:]=f(np.linspace(0, 128, t),np.linspace(0, 128, t))

Sa=Se.reshape(n,t*t)
X1=(A2[:,:,:,0].reshape(m, t**2)*Sa[0,:]).reshape(m,t,t)
X2=(A2[:,:,:,1].reshape(m, t**2)*Sa[1,:]).reshape(m,t,t)
X0=X1+X2

X1=X0.reshape(m,t**2)


N = np.random.randn(nb_obs,t*t)
sigma_noise = np.power(10.,(-SNR/20))*np.linalg.norm(X0.reshape(nb_obs, t*t),ord='fro')/np.linalg.norm(N,ord='fro')
N = sigma_noise*N.reshape(m,t,t)


'''
X: data matrix'''
X=X0+N
#%%
'''GMCA with 16 patchs of size (20 x 20)'''
(A0, Aout, Sout, Patch)=In.LGMCA(X, n, nb_obs, nb_patch=4, J=2,thres=1, kend=3)
#%%
(A0, Aout, Sout, Patch)=In.LGMCA(X, n, nb_obs, nb_patch=8, J=2,thres=1, kend=3)
#%
j=0
plt.subplot(121)
plt.imshow(A2[j,:,:,0]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,0],vmin=np.min(A2[j,:,:,0]), vmax=np.max(A2[j,:,:,0])), plt.colorbar()
plt.figure()
#%
j=-1
plt.subplot(121)
plt.imshow(A2[j,:,:,-1]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,-1],vmin=np.min(A2[j,:,:,-1]), vmax=np.max(A2[j,:,:,-1])), plt.colorbar()
#%
d={}
d['Aout']=Aout
d['Sout']=Sout
d['A0']=A0
sio.savemat('64patches', d)

#%%
(A0, Aout, Sout, Patch)=In.LGMCA(X, n, nb_obs, nb_patch=5, J=2,thres=1, kend=3)
#
j=0
plt.subplot(121)
plt.imshow(A2[j,:,:,0]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,0],vmin=np.min(A2[j,:,:,0]), vmax=np.max(A2[j,:,:,0])), plt.colorbar()
plt.figure()
#%
j=-1
plt.subplot(121)
plt.imshow(A2[j,:,:,-1]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,-1],vmin=np.min(A2[j,:,:,-1]), vmax=np.max(A2[j,:,:,-1])), plt.colorbar()
#%
d={}
d['Aout']=Aout
d['Sout']=Sout
d['A0']=A0
sio.savemat('25patches', d)

#%%
j=0
plt.subplot(121)
plt.imshow(A2[j,:,:,0]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,0],vmin=np.min(A2[j,:,:,0]), vmax=np.max(A2[j,:,:,0])), plt.colorbar()
plt.figure()
#%
j=-1
plt.subplot(121)
plt.imshow(A2[j,:,:,-1]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,-1],vmin=np.min(A2[j,:,:,-1]), vmax=np.max(A2[j,:,:,-1])), plt.colorbar()

d={}
d['Aout']=Aout
d['Sout']=Sout
d['A0']=A0
sio.savemat('16patches', d)
#%%
(A0, Aout, Sout, Patch)=In.LGMCA(X, n, nb_obs, nb_patch=2, J=2,thres=1, kend=3)
#
j=0
plt.subplot(121)
plt.imshow(A2[j,:,:,0]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,0],vmin=np.min(A2[j,:,:,0]), vmax=np.max(A2[j,:,:,0])), plt.colorbar()
plt.figure()
#%
j=-1
plt.subplot(121)
plt.imshow(A2[j,:,:,-1]), plt.colorbar()
plt.subplot(122)
plt.imshow(Aout[j,:,:,-1],vmin=np.min(A2[j,:,:,-1]), vmax=np.max(A2[j,:,:,-1])), plt.colorbar()
#%
d={}
d['Aout']=Aout
d['Sout']=Sout
d['A0']=A0
sio.savemat('4patches', d)


