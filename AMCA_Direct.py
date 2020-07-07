# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:16:32 2019

@author: ielham
"""
import numpy as np
import scipy as sp
import pyStarlet as ps
from scipy.fftpack import idct,dct
from copy import deepcopy as dp
from copy import copy

"""
Initialization with PCA 
"""
def InitPCA(X, dS):
    
    R = np.dot(X,X.T)
    D,V = np.linalg.eig(R)
    A = V[:,0:dS['n']]
    A=np.real(A) #Initialization of the mixing matrix 
    A[A<1e-16]=1e-16
    A=A/np.linalg.norm(A,axis=0)

    return(A)
#%%
    
def amcaps(X, dS , dPatch, aMCA, Init):
    

    (A,S)=AMCA(X, dS, aMCA=aMCA,Init=Init)  
   
    
    return (A,S)
    
#%%
def softThres(x,thres,typeThres):
    '''
    Hard or Soft Thresholding operator.
    Inputs:
    - x the signal, of size n1*n2 (can be a scalar)
    - thres, the thresholding values, of size n3*n4; with n3=n1 or n3=1; and n4=n2 or n4=1.
    - typeThres: should be 1 for the soft thresolding and 0 for hard thresholding
    Output:
    - soft/hard thresholded version of x. If thres is a scalar, every entry of x is thresholded by this same value.
    If thres is a column vector, then each row of x is thresholded by the corresponding value of thres. Reciprocally, if 
    thres is a row vector, the i th column of x is thresholded by thres_i. Last, if thres is a matrix, with dimension n1 by n2, each entry of
    x is thresholded by corresponding value in thres.
    '''
    
    return x*(np.array(np.abs(x)>thres,int))-np.sign(x)*thres*np.array(np.abs(x)>thres,int)*typeThres
    
###################################################
    
def madAxis(xin,axis='none'):
    ''' 
    Compute the median absolute deviation of a matrix, global or along each row.
    Inputs:
    - xin: the signal. If axis='none', xin can be an array or a list. Otherwise, should be a 2D matrix.
    - axis: if 'none', the mad is performed globally on xin. Otherwise, the mad of each row of xin is computed.
    Output:
    - if axis='none', returns a scalar/otherwise a column vector with the same number of rows than xin, each row containing the mad of the corresponding row of xin.
    '''
    if axis=='none':
        z = np.median(abs(xin - np.median(xin)))/0.6735

        return z
    else:
        z = np.median(abs(xin - np.median(xin,axis=1).reshape((np.shape(xin)[0],1))),axis=1)/0.6735
        return z.reshape((np.shape(xin)[0],1))    
        
def AMCA(Xini, ddS, aMCA,Init):
    import numpy.linalg as lng
    '''Perform GMCA or AMCA (sparse BSS) on the observations Xini, J.Bobin et al., Sparsity and Adaptivity for the Blind Separation of Partially Correlated Sources.
        
    Inputs:
    - Xini: the observations, m by t matrix. Xini corresponds to the observations expressed in a dictionary in which the sources to be estimated are sparse.
    - Aini: the initial mixing matrix, m by n matrix
    - aMCA: if aMCA=1, then AMCA is performed; if aMCA=0, GMCA is performed. 
    Outputs:
    - S, the estimated sources, matrix of size n by t
    - A, the estimated mixing matrix, size m by t
    '''
    dS=dp(ddS)
    dS['t']=np.shape(Xini)[1]
    
    if  Init==1:
        
        Aini=InitPCA(Xini, dS)
    
    else:
        Aini=abs(np.random.randn(dS['m'], dS['t']))
        Aini=Aini/lng.norm(Aini, axis=0)
    X=copy(Xini)
    A=copy(Aini)    
    S=np.zeros((dS['n'],dS['t']))
    
    #Initialize the weights for AMCA, with the largest entries.
    W = 1./np.linalg.norm(X+1e-10,axis=0,ord=1)**2;    
    W/=np.max(W)
    
    if aMCA==0:
        W=np.ones((dS['t'])) # Set the weights to 1 if GMCA is performed
    
        
    kend = dS['kSMax'] # Final thresholds of the sources: kend * sigma (k-mad)
  
    
    nmax=np.float(dS['iteMaxXMCA'])#Number of loops



    kIni=10. #Starting value for the threshold, k-mad
    dk=(kend-kIni)/nmax#Decrease of the k, for the kmad
    perc = 1./nmax
 
 
 
    ##### Start of GMCA/AMCA####
    it=0
    while it<nmax:#Stop when the maximal number of iterations is reached
        it += 1
        
    
    
        #######   Estimation of the sources ###
        sigA = np.sum(A*A,axis=0)
        indS = np.where(sigA > 0)[0]
        if np.size(indS) > 0: 
            Ra = np.dot(A[:,indS].T,A[:,indS])  
            Ua,Sa,Va = np.linalg.svd(Ra)
            cd_Ra = np.min(Sa)/np.max(Sa)

            ###Least squares estimate####
            if cd_Ra > 1e-5: #If A has a moderate condition number, performs the least squares with pseudo-inverse
                iRa = np.dot(Va.T,np.dot(np.diag(1./Sa),Ua.T))
                piA = np.dot(iRa,A[:,indS].T)    
                S[indS,:] = np.dot(piA,X)
 
            else:   #If A has a large condition number, update S with an 'incomplete' gradient descent                
                La = np.max(Sa)
                for it_A in range(250):
                    S[indS,:] = S[indS,:] + 1/La*np.dot(A[:,indS].T,X - np.dot(A[:,indS],S[indS,:]))
              
            Stemp = S[indS,:] 
            
            ###thresholding####
            for r in range(np.size(indS)):
                St = Stemp[r,:]
                indNZ=np.where(abs(St) > ((kIni+it*dk   )*madAxis(St)))[0]#only consider the entries larger than k-mad
                if len(indNZ)<dS['n']:
                    indNZ=np.where(abs(St)>=np.percentile(np.abs(St), 100.*(1-np.float(dS['n'])/dS['t'])))[0]
                Kval = np.min([np.floor(perc*(it)*len(indNZ)),dS['t']-1.])
                I = (abs(St[indNZ])).argsort()[::-1]
                Kval = np.int(min(max(Kval,dS['n']),len(I)-1))
                thrd=abs(St[indNZ[I[Kval]]])# threshold based on the percentile of entries larger than k-mad
                St[abs(St)<thrd]=0
                indNZ = np.where(abs(St) > thrd)[0]
                St[indNZ] = St[indNZ] - thrd*np.sign(St[indNZ]) #l1 thresholding
                Stemp[r,:] = St 
                
            S[indS,:] = Stemp
            Sref=copy(S)

                
        ####### Weights Update ####  
        if aMCA==1  and it>1: #Weights update for AMCA
                      
                        alpha=0.1**((it-1.)/(nmax-1.))/2.#p- of the lp norm of the weights
                        Ns = np.sqrt(np.sum(Sref*Sref,axis=1))
                        IndS = np.where(Ns > 0)[0] 
                        if len(IndS)>0:
                            Sref[IndS,:] = np.dot(np.diag(1./Ns[IndS]),Sref[IndS,:]) #normalized sources
                            W = np.power(np.sum(np.power(abs(Sref[IndS,:]),alpha),axis=0),1./alpha)
                            ind = np.where(W > 0)[0]
                            jind = np.where(W == 0)[0]
                            W[ind] = 1./W[ind];   
                            W/=np.max(W[ind])
                            if len(jind) > 0:
                                W[jind] = 1
                                
                            W/=np.max(W)#Weights


        Aold=dp(A)
        #### Update of A #########
        Ns = np.sqrt(np.sum(S*S,axis=1))
        indA = np.where(Ns > 0)[0]
        if len(indA) > 0:
            Sr = copy(S)*W # weighted sources
            Rs = np.dot(S[indA,:],Sr[indA,:].T)
            Us,Ss,Vs = np.linalg.svd(Rs)
            cd_Rs = np.min(Ss)/np.max(Ss)
            if cd_Rs > 1e-4: # if the sources have a fair condition number, use the pseudo-inverse
                piS = np.dot(Sr[indA,:].T,np.linalg.inv(Rs));
                A[:,indA] = np.dot(X,piS)
                A[A<1e-16]=1e-16
                A = np.dot(A,np.diag(1./(1e-24 + np.sqrt(np.sum(A*A,axis=0)))));
            else:#if the condition number of the projected sources is too large, do an 'interrupted' gradient descent
                Ls = np.max(Ss)
                indexSub=0
                while indexSub<250:
                    A[:,indA] = A[:,indA] + 1/Ls*np.dot(X - np.dot(A[:,indA],S[indA,:]),Sr[indA,:].T)
                    A[A<1e-16]=1e-16
                    A[:,indA] = np.dot(A[:,indA],np.diag(1./(1e-24 + np.sqrt(np.sum(A[:,indA]*A[:,indA],axis=0)))));
                    indexSub+=1

    return A, S
    
    
def refinment(X, A, S, dS):
    itemax=dS['iteref']
    
    for _ in range(itemax):
        madS=madAxis(A.T.dot((X-A.dot(S))), axis=1)
        L=np.linalg.norm((A.T.dot(A)),2)
        S=(softThres((S+1./L*A.T.dot(X-A.dot(S))),madS*dS['kref']*1./L,1))
    
    return (S)
    
    