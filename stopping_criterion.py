# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:35:36 2020

@author: ielham
"""

import numpy as np 
from scipy import fftpack as ft
from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 
from scipy.fftpack import idct,dct
import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
import pyStarlet as ps
import initialization as pyl


def spectral_var_rel(Ak, Ak_1):
    
    (nb_obs, nb_pix, nb_sources)=np.shape(Ak)
    
    Akf=LA.norm(ft.dct(Ak, axis=1, norm='ortho'), axis=0)
    
    Akf_1=LA.norm(ft.dct(Ak_1, axis=1, norm='ortho'), axis=0)
    
    g=LA.norm(Akf-Akf_1)/LA.norm(Akf_1)
    
    return(g)
    #%%
