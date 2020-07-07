import numpy as np

from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 
from scipy.fftpack import idct,dct
import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
import pyStarlet as ps
import initialization as pyl
from scipy import stats

import Utils_generation as utils  
import algoS as algS
import FISTA_Haar as algoA
import stopping_criterion
import BSS_J as bssJ
import generation_physpec as gen
import thres_Haar as thres
#%%


nb_obs=5
nb_pix=1500
(m,t,n)=(5, 1500, 2)
SNR=45
dS={}
dS['n']=2 
dS['m']=5
dS['t']=nb_pix
dS['kSMax']=3
dS['iteMaxXMCA']=2000
dPatch={}
dPatch['PatchSize']=500
dPatch['J']=2


#%%

#Sces sparse in direct domain
Se,Sc_part,ind_cor,ind_nocor=bssJ.MixtMod_pos(n=n,t=nb_pix,sigma1=1,p1=0.,ptot=.14)

'''Opt for A and SV'''
nbr_bloc=3
lowest=6 #Minimal size of patches


A2=np.zeros((m, t,n))

#var 1 and var2: Each source SV
ru=1
while ru>0:
    var1=gen.VS_piecewcst(t, nbr_bloc,low_piece=lowest, filt=1)
    var2=gen.VS_piecewcst(t, nbr_bloc,low_piece=lowest, filt=1)

    ru=np.sum(np.isnan(var1))+np.sum(np.isnan(var2))
    
A2[:,:,0],a=gen.Get_SV_EmissionLaw(t=1500, minmax_pos= [.05, .25], width=.1,var_vec=var1)    
A2[:,:,1], b=gen.Get_SV_EmissionLaw(t=1500, minmax_pos=[.7, .9], width=.1,var_vec=var2)  

plt.subplot(2,1,1)
plt.plot(A2[:,:,0].T), plt.title('Source 1')
plt.subplot(2, 1, 2)
plt.plot(A2[:,:,1].T), plt.title('Source 2')

(X, N, sigma_noise)=utils.XN(A2, Se, m=nb_obs, t=nb_pix, noise_level=SNR)
 #%%

'''Run L-GMCA (without filtering)'''   
#(Aout, Sout): result of L-GMCA
#A0: Frechet mean of Aout
Aout, Sout, Ar, Sr,temp2 ,ss= pyl.GMCAperpatch(0,X, dS, dPatch, Init=1, aMCA=0)
A0=np.repeat(gen.moyFrechet(Aout).reshape(m,1,n), t, 1)


'''Run svGMCA: warm-up stage (without weights)''' 
seuil_1=thres.threshold_finalstep(3,sigma_noise,110/100.,Sout, Aout, A0, X, 5,  0, J=2)
(A_1, r,ecartre,elapsed_time)= algoA.FISTA(Sout, X, A0, Aout,seuil_1, 8800 ,lim=3e-7 ,stepg=.8, iteprox=6000, limprox=5e-7, J=2)
A=np.zeros((400, 5,nb_pix ,2))
S=np.zeros((400, 2, nb_pix ))
seuil=np.zeros((400, nb_pix, 2, 2))  
S[0,:,:]=dp(Sout)
A[0,:,:,:]=dp(A_1)
inner_ecart=[]
outer_ecart=[]
outer_time=[]
l=dp(0)
for k in np.arange(l, l+30):     
   print(k)
   S[k+1,:,: ]=algS.lasso_direct(X, A[k,:,:,:], S[k,:,:],kend=3, stepgg=1.,resol=2, lim=5e-6)
   seuil[k+1,:,:,:] = thres.threshold_finalstep(3,sigma_noise,(110)/100.,S[k+1,:,:], A[k,:,:,:], A0, X, 5,  0)
   (A[k+1,:,:,:],r,ecartre,elapsed_time)= algoA.FISTA(S[k+1,:,:], X, A0, A[k,:,:,:],seuil_1, 6000,lim=1e-6 ,stepg=.8, iteprox=6000, limprox=3e-6)
   A0=np.repeat(gen.moyFrechet(A[k+1,:,:,:]).reshape(m,1,n), t, 1)
   inner_ecart.append(ecartre)
   if ecartre[-1]>5e-5:
       break
   outer_time.append(elapsed_time)
   eA=stopping_criterion.spectral_var_rel(A[k+1,:,:,:], A[k,:,:,:])
   if eA<1e-4:
       break
   outer_ecart.append(eA)

'''Refinement stage (with weights)'''
l=dp(k+1)
for k in np.arange(l, l+40):     
   print(k)
   S[k+1,:,: ]=algS.lasso_direct(X, A[k,:,:,:], S[k,:,:],kend=3, stepgg=.9,resol=2, lim=5e-6)
   seuil[k+1,:,:,:] = thres.threshold_finalstep(3,sigma_noise,(105.)/100.,S[k+1,:,:], A[k,:,:,:], A0, X, 5,  1)
   (A[k+1,:,:,:],r,ecartre,elapsed_time)= algoA.FISTA(S[k+1,:,:], X, A0, A[k,:,:,:],seuil_1, 6000,lim=1e-6 ,stepg=.8, iteprox=6000, limprox=1e-6)
   A0=np.repeat(gen.moyFrechet(A[k+1,:,:,:]).reshape(m,1,n), t, 1)
   inner_ecart.append(ecartre)
   if ecartre[-1]>5e-5:
       break
   outer_time.append(elapsed_time)
   eA=stopping_criterion.spectral_var_rel(A[k+1,:,:,:], A[k,:,:,:])
   if eA<1e-6:
       break
   outer_ecart.append(eA)
   

