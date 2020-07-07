# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:28:37 2017

@author: ckervazo
"""

#%%
import copy as cp
import numpy as np
import numpy.linalg as lng
from copy import deepcopy as dp 
import matplotlib.pyplot as plt
#%%

def BSSEvalCorrected(A0,S0,gA,gS,n=[]):
     '''
     Computes the SDR, SAR, SIR, SNR
    
     Inputs:
     A0 : reference mixing matrix
     S0 : reference sources (each source is a row)
     gA : estimated mixing matrix
     gS : estimated sources
     n : noise applied to the observations X (if n == [], we consider that there is no noise)
     
     Outputs:
     SDR,SAR,SIR,SNR: global criteria on all the sources
     sRES[0,j]: SDR for the jth source,
     sRES[1,j]: SAR for the jth source,
     sRES[2,j]: SIR for the jth source,
     sRES[3,j]: SNR for the jth source.
     
     s_target,e_interf,e_noise,e_artif: projections used
     
     '''
     import numpy as np
     
     
     nS = np.shape(A0)[1]
     
    # A,S = CorrectPerm(A0,S0,gA,gS)
     S=dp(gS)
     S0norm = np.dot(np.diag(1./np.sqrt(np.sum(S0**2,axis=1))),S0) # S0 renormalized with norm 1
     s_target = np.dot(np.diag(np.diag(np.dot(S0norm,S.T))),S0norm) # Projection of each source on the corresponding TRUE source
     
     R_s0s0 = np.dot(S0,S0.T) # Gram matrix of the true sources
     Ps = np.dot(np.dot(np.dot(S,S0.T),np.linalg.inv(R_s0s0)),S0) # Projection of each sources on all the true sources
     e_interf = Ps - s_target
     
     if n == []:# if the noise is not given, we consider that there is no noise (e_inter = 0 and Ps0,n = Ps0)
         SN = cp.deepcopy(S0)
     else:
         SN = np.vstack((S0,n))
         
     R_snsn = np.dot(SN,SN.T) # Gram matrix of the family composed of the sources and the noise
     Psn = np.dot(np.dot(np.dot(S,SN.T),np.linalg.inv(R_snsn)),SN) # Projection of each source on all the sources and the noise
     e_noise = Psn - Ps

     e_artif = S - Psn

     SDR = 10*np.log10((np.linalg.norm(s_target,ord='fro')**2)/(np.linalg.norm(e_interf + e_noise + e_artif,ord='fro')**2)) # Global SDR
     SIR = 10*np.log10((np.linalg.norm(s_target,ord='fro')**2)/(np.linalg.norm(e_interf,ord='fro')**2)); # Global SIR : takes into account the interferences
     SNR = 10*np.log10((np.linalg.norm(s_target + e_interf,ord='fro')**2)/(np.linalg.norm(e_noise,ord='fro')**2));# Global SNR : takes into account the noise
     SAR = 10*np.log10((np.linalg.norm(s_target + e_interf + e_noise,ord='fro')**2)/(np.linalg.norm(e_artif,ord='fro')**2))# Global SAR : takes into account the artifacts

#     sRES = np.zeros((nS,3))
#     for r in range(nS):
#         sRES[r,0] = 10*np.log10(np.linalg.norm(s_target[r,:],ord=2)**2/(np.linalg.norm(e_interf[r,:] + e_noise[r,:] + e_artif[r,:],ord=2)**2)) # SDR for each source
#         sRES[r,1] = 10*np.log10((np.linalg.norm(s_target[r,:] + e_interf[r,:] + e_noise[r,:],ord=2)**2)/(np.linalg.norm(e_artif[r,:],ord=2)**2))# SAR for each source
#         sRES[r,2] = 10*np.log10(np.linalg.norm(s_target[r,:],ord=2)**2/(np.linalg.norm(e_interf[r,:],ord=2)**2));# SIR for each source
#        # sRES[r,3] = 10*np.log10((np.linalg.norm(s_target[r,:] + e_interf[r,:],ord=2)**2)/(np.linalg.norm(e_noise[r,:],ord=2)**2));# SNR for each source

         

         
     return SNR,SDR,SAR,SIR,s_target,e_interf,e_noise,e_artif
     
     


def CorrectPerm(cA0,S0,cA,S,optEchAS=0):

    A0 = cp.copy(cA0)
    A = cp.copy(cA)
    
    nX = np.shape(A0)
    
    for r in range(0,nX[1]):
        S[r,:] = S[r,:]*(1e-24+lng.norm(A[:,r]))
        A[:,r] = A[:,r]/(1e-24+lng.norm(A[:,r]))
        S0[r,:] = S0[r,:]*(1e-24+lng.norm(A0[:,r]))
        A0[:,r] = A0[:,r]/(1e-24+lng.norm(A0[:,r]))
        
    try:
        Diff = abs(np.dot(lng.inv(np.dot(A0.T,A0)),np.dot(A0.T,A)))
    except np.linalg.LinAlgError:
        Diff = abs(np.dot(np.linalg.pinv(A0),A))    
        print('ATTENTION, PSEUDO INVERSE POUR CORRIGER LES PERMUTATIONS')            
        
        

    Sq = np.ones(np.shape(S))
    ind = np.linspace(0,nX[1]-1,nX[1])
    
    for ns in range(0,nX[1]):
        indix = np.where(Diff[ns,:] == max(Diff[ns,:]))[0]
        ind[ns] = indix[0]
    
    Aq = A[:,ind.astype(int)]
    Sq = S[ind.astype(int),:]

    for ns in range(0,nX[1]):
        p = np.sum(Sq[ns,:]*S0[ns,:])
        if p < 0:
            Sq[ns,:] = -Sq[ns,:]
            Aq[:,ns] = -Aq[:,ns]
    
    #Sq = Sq[::-1,:]
    #Q = Q[:,::-1]    
    
    if optEchAS==1:
        return Aq,Sq,A0,S0
    else:
        return Aq,Sq
def XN(A, S, m=5, t=1024, noise_level=25):
    X0=np.sum(A*S.T, axis=2)
    N=np.random.randn(m,t) 
    
    sigma_noise = np.power(10.,(-noise_level/20.))*np.linalg.norm(X0,ord='fro')/np.linalg.norm(N,ord='fro')
    N = sigma_noise*N
    
    X=X0+N
    
    return (X, N, sigma_noise)     
    #%%
#nbr_real=np.shape(d['A2'])[0]
#snr=np.zeros((nbr_real))
#sir=np.zeros((nbr_real))
#sar=np.zeros((nbr_real))
#sdr=np.zeros((nbr_real))
#
#snro=np.zeros((nbr_real))
#siro=np.zeros((nbr_real))
#saro=np.zeros((nbr_real))
#sdro=np.zeros((nbr_real))
#
#SNR=45
#for k in range(nbr_real):   
#        
#        (X, N, sigma_noise)=XN(d['A2'][k,:,:,:], d['Se'][k,:,:], m=5, t=1500, noise_level=SNR)
#        
#        SNR,SDR,SAR,SIR,s_target,e_interf,e_noise,e_artif=BSSEvalCorrected(d['A2'][k,:,0,:],d['Se'][k,:,:],d['A'][k,:,0,:],d['S'][k,:,:], n=N)
#
#        SNRo,SDRo,SARo,SIRo,s_target,e_interf,e_noise,e_artif=BSSEvalCorrected(d['A2'][k,:,0,:],d['Se'][k,:,:],d['Aout'][k,:,0,:],d['Sout'][k,:,:], n=N)
#
#        snr[k]=SNR
#        sar[k]=SAR
#        sir[k]=SIR
#        sdr[k]=SDR
#        
#        snro[k]=SNRo
#        saro[k]=SARo
#        siro[k]=SIRo
#        sdro[k]=SDRo
#print(SNR)
#print(SDR)
#print(SAR)
#print(SIR)

#yn=np.zeros((9, 5))
#yd=np.zeros((9, 5))
#yi=np.zeros((9, 5))
#ya=np.zeros((9, 5))
#
#yno=np.zeros((9, 5))
#ydo=np.zeros((9, 5))
#yio=np.zeros((9, 5))
#yao=np.zeros((9, 5))

#k=8
##yn[k,:]=dp(snr)
##yd[k,:]=dp(sdr)
##ya[k,:]=dp(sar)
##yi[k,:]=dp(sir)
##
##yno[k,:]=dp(snro)
##ydo[k,:]=dp(sdro)
##yao[k,:]=dp(saro)
##yio[k,:]=dp(siro)
#yn[k,:-1]=dp(snr)
#yd[k,:-1]=dp(sdr)
#ya[k,:-1]=dp(sar)
#yi[k,:-1]=dp(sir)
#
#yno[k,:-1]=dp(snro)
#ydo[k,:-1]=dp(sdro)
#yao[k,:-1]=dp(saro)
#yio[k,:-1]=dp(siro)
##%%
#
#da['yn']=yn
#da['yd']=yd
#da['ya']=ya
#da['yi']=yi
#
#da['yno']=yno
#da['ydo']=ydo
#da['yao']=yao
#da['yio']=yio
#
#
##%%
#z=np.zeros((5))
#zo=np.zeros((5))
#s=np.zeros((5))
#so=np.zeros((5))
#
#for k in range(4):
#    z[k]=np.mean(ya[:,k])
#    zo[k]=np.mean(yao[:,k])
#    s[k]=np.std(ya[:,k])
#    so[k]=np.std(yao[:,k])
##%%
#z[-1]=np.mean(ya[:-1,-1])
#zo[-1]=np.mean(yao[:-1,-1])
#s[-1]=np.std(ya[:-1,-1])
#so[-1]=np.std(yao[:-1,-1])
#
#
##%%
#da={}
#
#da['snr4']=snr
#da['sar4']=sar
#da['sir4']=sir
#da['sdr4']=sdr
#
#da['snro4']=snro
#da['saro4']=saro
#da['siro4']=siro
#da['sdro4']=sdro
#
##%%
#z=np.zeros((6))
#zo=np.zeros((6))
#
#z[0]=np.mean(da['sdr1e-3'])
#z[1]=np.mean(da['sdr1e-2'])
#z[2]=np.mean(da['sdr.1'])
#z[3]=np.mean(da['sdr.3'])
##z[4]=np.mean(da['snr.5'])
#z[4]=np.mean(da['sdr.7'])
#z[5]=np.mean(da['sdr.9'])
#
#
#zo[0]=np.mean(da['sdro1e-3'])
#zo[1]=np.mean(da['sdro1e-2'])
#zo[2]=np.mean(da['sdro.1'])
#zo[3]=np.mean(da['sdro.3'])
##zo[4]=np.mean(da['dnro.5'])
#zo[4]=np.mean(da['sdro.7'])
#zo[5]=np.mean(da['sdro.9'])
#
##%%
#x=[1e-3, 1e-2, .1, .3, .7, .9]
#
#
#
#
#
