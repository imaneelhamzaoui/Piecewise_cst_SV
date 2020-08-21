from astropy.io import fits
from astropy.visualization import simple_norm

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy.linalg as LA


#%%

'''

Examples:

E,spec_Fe=np.genfromtxt('data/line_FeXXV_rebin1.txt',unpack=True)
E,spec_synch=np.genfromtxt('data/pow_Index2_rebin1.txt',unpack=True)
E, spec_Fe, spec_synch = E[i1:i2], spec_Fe[i1:i2], spec_synch[i1:i2]
spec_Fe, spec_synch = spec_Fe/np.sum(spec_Fe), spec_synch/np.sum(spec_synch)

im_Fe1=fits.getdata('data/im_ref_Fe.fits')
im_Fe2=fits.getdata('data/im_ref_Fe2.fits')
im_synch=fits.getdata('data/im_ref_synch.fits')


cube_synch = im_synch[:,:,None] * spec_synch[None, None, :]

VS_blue = get_VS(spec_Fe,xcen=120, ycen=115, nx=230, ny=230)
VS_blue=VS_blue/LA.norm(VS_blue,axis=0)

VS_red = get_VS(spec_Fe,xcen=120, ycen=115, nx=230, ny=230)
VS_red=VS_red/LA.norm(VS_red,axis=0)
'''






#%%
def dist_circle(nx,ny,xcen,ycen):
    " Create a 2D numpy array where each value is its distance to a given center. "
    import numpy as np
    mask=np.zeros([nx,ny])
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    x,y = np.meshgrid(x, y)
    x,y = x.transpose(), y.transpose()
    xcen, ycen = ycen, xcen #transpose centers as well

    mask = np.sqrt((x-xcen)**2 + (y-ycen)**2 )
    return mask
    
    
def VmapF(R0=90,xcen=120, ycen=115, nx=230, ny=230):
    
    
    Vmax = 4000*1 # km/s
    
    r=dist_circle(nx,ny,xcen,ycen)
    
    theta=np.arcsin(r/R0)
    Vmap=Vmax*np.cos(theta)
    Vmap=np.nan_to_num(Vmap)
    
    return(Vmap)
    
#%%
    
def get_VS(typeFe,spec,R0=90,xcen=120, ycen=115, nx=230, ny=230):
    
    Eref=6.5 #keV energy reference
    c = 300e3 # speed of light in km/s
    if typeFe==1:
        Vmap=VmapF(R0,xcen, ycen, nx, ny)
    elif typeFe==2:
        Vmap=-VmapF(R0,xcen, ycen, nx, ny)
    

    mask = np.asarray(Vmap != 0, dtype='int')
    spec_bin_size = 14.6e-3 # keV
    
    VS = np.zeros((nx,ny,len(spec)))

    for i in range(nx):
        for j in range(ny):
            if mask[i,j] == 1:
                deltaE = Vmap[i,j]/c * Eref
                shift = int(deltaE/spec_bin_size)
                VS[i,j,:] =  np.roll(spec,shift)
                
                
                VS[i,j,:] =VS[i,j,:] /LA.norm(VS[i,j,:] )
    
    return (np.transpose(VS,(2,0,1)))

def get_spec(typeSpec):
    i1, i2 = 378, 514 # 5.5-7.5 keV
    E,spec_Fe=np.genfromtxt('data/line_FeXXV_rebin1.txt',unpack=True)
    E,spec_synch=np.genfromtxt('data/pow_Index2_rebin1.txt',unpack=True)
    E, spec_Fe, spec_synch = E[i1:i2], spec_Fe[i1:i2], spec_synch[i1:i2]
    spec_Fe, spec_synch = spec_Fe/np.sum(spec_Fe), spec_synch/np.sum(spec_synch)
    if typeSpec=='Fe':
        x=spec_Fe
    else:
        x=spec_synch
        
    return(x)