"""

Code to generate physically relevant spectra

"""

import numpy as np
from numpy import linalg as LA
from scipy import fftpack as ft
from scipy.fftpack import idct,dct
import FrechetMean as fm
from scipy import stats

# Some basic function

def power_law(m=5,index=-2.4):
    """
    Power law spectrum

    opt: m - array dimension
         index - spectral index of the PL
    """

    temp = np.exp(index*np.linspace(1,m,m)/m)

    return temp/np.linalg.norm(temp)

def Emission_law(m=5,pos=0.5,width=0.2):
    """
    Gaussian-shapes emission law

    opt: m - array dimension
         pos - position of the emission peak (between 0 and 1)
         width - width of the peak
    """

    temp = np.linspace(0,m-1,m)/m - pos
    temp = np.exp(-temp**2./width)

    return temp/np.linalg.norm(temp)

def VS_give(x,t, higher_freq, activ_param, lower_freq=1,Energy=1):
    
    x[0]=0
    x=Energy*x/(LA.norm(x))#normalization
    
        
    res=ft.idct(x, norm='ortho')
    
    res+=abs(np.min(res))
    a=np.max(res)
    #res=res/np.max(res)#x non-negative
    return((res)/(1.*a))


def gen_VS(rho, freq_cent, t, sigma):
    import scipy as sp
    
    wi=6*sigma #beyond this value, the gaussian will be almost null anyway
    
    e=gaussian_filter(wi, sigma)
     
    d1=np.zeros((t))

    x=np.array(sp.stats.bernoulli.rvs(rho,size=(t)))
    
    if freq_cent>wi/2.:
        
        d1[freq_cent-wi/2:freq_cent+wi/2]=e*x[freq_cent-wi/2:freq_cent+wi/2]
    else:
        len_e=freq_cent+wi/2
        
        d1[:len_e]=e[3*sigma-freq_cent:3*sigma-freq_cent+len_e]*x[:len_e]
    return(d1)

def VS_dct(t, higher_freq, activ_param, lower_freq=1,Energy=1):
    import scipy as sp
    ru=0
    
    while ru==0:
        x=np.zeros((t))
        x[lower_freq:higher_freq]=np.array(sp.stats.bernoulli.rvs(activ_param,size=(higher_freq-lower_freq)))
  
        x[0]=0
        ru=np.sum(x!=0)
    x=Energy*x/(LA.norm(x))#normalization
    
        
    res=ft.idct(x, norm='ortho')
    
    res+=abs(np.min(res))
    a=np.max(res)
    #res=res/np.max(res)#x non-negative
    return((res)/(1.*a))

def VS_oracle(t, Patchsize, amp=1., filt=1, sigma=30):
    """
    generation of piecewise SV appropriate to oracle L-GMCA
    """
    vec=np.zeros((t))
    
    e=t/Patchsize
    for k in range(e):
        vec[int(Patchsize*k):int(Patchsize*(k+1))]=(np.random.randn(1))*np.ones((int(Patchsize)))
    
    vec-=(np.min(vec))
    a=np.max(vec)
    res=vec/(1.*a)
    
    if filt and sigma>0:

       res=filtered(t, sigma, vec/(1.*a))
    return(res)
        
def VS_piecewcst(t, freq=.2,low_piece=200, amp=1., filt=1, sigma=30):
    """
    SV of size t
    

    opt: t - nbr of samples
         freq - percentage of the mean size of a block the nbr of samples
         --- the lowerr freq, the more VS ---
         low_piece : lowest nbrr of samples for each block
         amp: maximal amplitude
         sigma: std of the gaussian filter
         
    amp must be btwn 0 and 1
    low_piece must be superior than t*freq
    """
    
    nbr_bloc=int(t/(t*freq))
    g=np.zeros((nbr_bloc))
    
    for k in range(nbr_bloc-1):
        C=t-np.sum(g)-nbr_bloc*low_piece
        if C>low_piece:
            g[k]=np.random.randint(low_piece, C)
        else:
            g[k]=low_piece
        
            
    g[-1]=int(t-np.sum(g))
    np.random.shuffle(g)
    print(g)
    vec=np.zeros((t))
    e=len(g)
    
    
    for k in range(e):
        vec[int(np.sum(g[:k])):int(np.sum(g[:k+1]))]=(np.random.randn(1))*np.ones((int(g[k])))
    
    vec-=(np.min(vec))
    a=np.max(vec)
    res=vec/(1.*a)
    
    if filt and sigma>0:
        res=filtered(t, sigma, vec/(1.*a))
    return(res)
    

# Creating spectral variabilities

def Get_SV_PowerLaw(m=5,t=1024,minmax_index=[2.4,3.4],var_vec=None):

    """
    Creating a single sv matrix with power law spectra

    opt: m - array dimension
         t - number of samples
         minmax_index - spectral index of the PL (min and max values)
         var_vec - if not None, provides the vector of variabilities for the power law (default is a simple cosine variation)
                   must be normalized to lie within the [0,1] interval

    output: SV - m x t matrix (spectral variabilities for a single source)
            Param - vector for the spectral index variabilities
    """

    if var_vec is None:
        var_vec = np.cos(np.pi*np.linspace(0,t-1,t)/t)

    Param = (minmax_index[1]-minmax_index[0])*var_vec+minmax_index[0]
    SV = np.zeros((m,t))

    for r in range(t):
        SV[:,r] = power_law(m,index=Param[r])

    return SV,Param

# Creating spectral variabilities

def Get_SV_EmissionLaw(m=5,t=1024,minmax_pos=[0.2,0.3],var_vec=None,width=0.2,freq=1): # we could actually the width as well if needed

    """
    Creating a single sv matrix with emission law spectra

    opt: m - array dimension
         t - number of samples
         minmax_index - spectral index of the PL (min and max values)
         var_vec - if not None, provides the vector of variabilities for the power law (default is a simple cosine variation)
                   must be normalized to lie within the [0,1] interval
         width - width of the Gaussian shape

    output: SV - m x t matrix (spectral variabilities for a single source)
            Param - vector for the emission peak position variabilities
    """

    if var_vec is None:
        var_vec = (np.cos(freq*np.pi*np.linspace(0,t-1,t)/t)+1)/2

    Param = (minmax_pos[1]-minmax_pos[0])*var_vec+minmax_pos[0]
    SV = np.zeros((m,t))

    for r in range(t):
        SV[:,r] = Emission_law(m,pos=Param[r],width=width)

    return SV,Param

def moyFrechet(A2):
    (m,t,n)=np.shape(A2)
    Afre=np.zeros((m,n))
    for i in range(n):
        Af=fm.FrechetMean(A2[:,:,i].reshape((m, t)))
        #if A2 is constant Frechet mean returns NaN
        if np.sum(np.isnan(Af))>0:
            Af=(A2[:,0,i])
        Afre[:,i]=Af
    

    return(Afre)
    
def gaussian_filter(t_samp, sigma):
    x = np.linspace(1,t_samp,t_samp)-t_samp/2

    kern = 1 / (sigma * np.sqrt(2*np.pi))*np.exp(-(x)**2/(2*sigma**2))
 
    return (kern)
    
def filtered(t_samp, sigma, x):
    from copy import deepcopy as dp
    
    #avoiding border effects
    y=np.zeros((t_samp+int(6*sigma)))
    y[:3*sigma]=x[0]*np.ones((3*sigma))
    y[-3*sigma:]=x[-1]*np.ones((3*sigma))
    y[3*sigma:-3*sigma]=dp(x)
    
    return(np.convolve(y,gaussian_filter(t_samp, sigma),mode='same')[3*sigma:-3*sigma])

def laplacian_filter(t_samp, w):
    
    x = np.linspace(1,t_samp,t_samp)-t_samp/2

    kern = np.exp(-abs(x)/(w/np.log(2)))
    kern = kern/np.max(kern)
    return(kern)
    
