# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:23:16 2020

@author: ielham
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:54:33 2020

@author: ielham
"""

import numpy as np 
import pywt

def forward_UWT_pad(datai, J, axis=1):
    """
    rec_df : detail to manipulate for thresholding... (m*t*J)
    rec_d : m*3t*J
    """
    maxlev=J
    
    if axis:

        (m,t1)=np.shape(datai)
        t=3*t1
        data=np.zeros((m,t))
        for j in range(m):
            data[j,:]=np.pad(datai[j,:], (t1,t1), 'linear_ramp')
        rec_d=np.zeros((m,t,maxlev))
        rec_a=np.zeros((m,t,maxlev))
        rec=pywt.swt(data, 'haar',level=J, axis=1)
        for k in range(maxlev):
            rec_d[:,:,k]=rec[k][1]
            rec_a[:,:,k]=rec[k][0]
        rec_d_f=rec_d[:,t1:-t1,:]
    else:
        t1=len(datai)
        data=np.pad(datai, (t1,t1), 'linear_ramp')
        t=len(datai)*3
        #maxlev = pywt.swt_max_level(len(data))
        maxlev=J
        rec_d=np.zeros((t,maxlev))
        rec_a=np.zeros((t,maxlev))
        rec=pywt.swt(data, 'haar', level=J)
        for k in range(maxlev):
            rec_d[:,k]=rec[k][1][:]
            rec_a[:,k]=rec[k][0][:]
        rec_d_f=rec_d[t1:-t1,:]
    return(rec_d_f,rec_d, rec_a)




def backward_UWT_pad(rec_d_f,rec_d, rec_a, axis=1):
    
    if axis:
        (m, t3, maxlev)=np.shape(rec_d)
        t=t3/3
        rec=np.zeros((m, t))
        rec_d[:,t:-t,:]=rec_d_f
        for l in range(m):
            coeffs=zip(rec_a[l,:,:].T, rec_d[l,:,:].T)
            rec[l,:]=pywt.iswt(coeffs, 'haar')[t:-t]
    else:
        t=np.shape(rec_d_f)[0]
        rec_d[t:-t,:]=rec_d_f
        coeffs=zip(rec_a.T, rec_d.T)
        rec=pywt.iswt(coeffs, 'haar')[t:-t]
    return(rec)

def norm_rec(data, J):
    import numpy.linalg as LA
    rec_d_f,rec_d, rec_a=forward_UWT_pad(data, J, axis=1)
    rec_n=LA.norm(rec_d_f,axis=0)
    
    return(rec_n)

  
    