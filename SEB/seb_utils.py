#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Utility functions to help in computation of SEB. 
"""
import numpy as np, pandas as pd
from numba import jit
import scipy.optimize as optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Global vars 
e0=611.0
epsilon=0.622 # Ratio of gas constants for dry/moist air 
rd=287.053 # Gas constant for dry air (J/K/kg)
e0=611.0 # Constant to evaluate vapour pressure in Clasius Clapeyron equation (Pa)
boltz=5.67*10**-8 # Stefan Boltzmann constant

@jit
def SATVP(t):
    
    """
    This function computes saturation vapour pressure for temperature, t. 
    The correct latent heat constant is selected based on t (sublimation 
    if < 0)
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (K)     : temperature 
        
    Out:
        
        - vp (Pa)   : vapour pressure
    """
    t=np.atleast_1d(t)
    LRv=np.ones(len(t))
    LRv[t<=273.15]=6139
    LRv[t>=5423]=5423

    vp=e0*np.exp( LRv * (1./273.15-1./t))
    if len(vp) == 1: vp = vp[0]
    return vp

@jit
def Q2VP(q,p):
    
    """
    This is a very simple function that, given, air pressure, 
    converts specific humidity to vapour pressure. 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - q (kg/kg) : specific humidity
        - p  (Pa)   : air pressure
        
    Out:
        
        - vp (Pa)   : vapour pressure
    """
    
    vp=np.divide(np.multiply(q,p),epsilon)
    
    return vp

@jit
def VP2Q(vp,p):
    
    """
    Given vp and P convert to specific humidity 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - vp (Pa)   : vapour pressure
        - p  (Pa)   : air pressure 
        
    Out:
        
        - q (kg/kg) : specific humidity
    """
    
    q=np.divide(np.multiply(epsilon,vp),p)
    
    return q

@jit
def MIX(p,e):

    """
    This function computes the mixing ratio
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)  : air pressure
        - e (Pa)  : vapour pressure 
        
    Out:
        
        - mr (kg vapour/kg dry air)   : mixing ratio
    """

    mr=epsilon*e/(p-e)
    
    return mr

@jit
def VIRTUAL(t,mr):
    
    """
    This function computes the virtual temperature
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (K)    : air pressure
        - mr (kg vapour/kg dry air)   : mixing ratio
        
    Out:
        
        - tv (K)   : virtual temperature
    """    
    
    tv=t*(1+mr/epsilon)/(1.+mr)
    
    return tv

@jit
def RHO(p,tv):
    
    """
    Computes the air density
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)  : air pressure
        - tv (K)  : virtual temperature
        
    Out:
    
        - rho (kg/m^3) : air density
        
    """    
    
    rho=np.divide(p,np.multiply(rd,tv))
    
    return rho

# returns number of nans in an array
def count_nan(arrin):
    return np.sum(np.isnan(arrin))

""" Below are functions to compute potential (ToA) solar radiation. These
functions are used to figure out day/night for purposes of setting night-time 
sin values to zero. """

# Functions (also now in GF)
@jit
def _decl(lat,doy):
    # Note: approzimate only. Assumes not a leap year
    c=2.*np.pi
    dec=np.radians(23.44)*np.cos(c*(doy-172.)/365.25)

    return dec, np.degrees(dec)

@jit
def _sin_elev(dec,hour,lat,lon):
    c=c=2.*np.pi
    lat=np.radians(lat)
    out=np.sin(lat)*np.sin(dec) - np.cos(lat)*np.cos(dec) * \
    np.cos(c*hour/24.+np.radians(lon))
    
    # Out is the sine of the elevation angle
    return out, np.degrees(out)

@jit
def _sun_dist(doy):
    c=c=2.*np.pi
    m=c*(doy-4.)/365.25
    v=m+0.0333988*np.sin(m)+0.0003486*np.sin(2.*m)+0.0000050*np.sin(3.*m)
    r=149.457*((1.-0.0167**2)/(1+0.0167*np.cos(v)))
    
    return r
    
@jit
def sin_toa(doy,hour,lat,lon):
    dec=_decl(lat,doy)[0]
    sin_elev=_sin_elev(dec,hour,lat,lon)[0]
    r=_sun_dist(doy)
    r_mean=149.6
    s=1366.*np.power((r_mean/r),2)
    toa=sin_elev*s; toa[toa<0]=0.
    
    return toa, r

# Below are functions to estimate LW using the method of Remco de Kok et al. 
#(2019):
# (https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.6249) -- eq.8:
     
#     LW=c1+c2RH+c3*boltz*Tk^4
     
# The set of functions includes optimizations to find the coefficients 

@jit
def sim_lw(rh,tk,lw,toa):
    
    """
    This is the main coordinating function to estimate LW from RH and T. It 
    calls sub routines to optimize coefficients on subsets of data (cloudy/
    clear). These subsets are found with ToA and Rh
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - rh (%)         : relative humidity
        - tk (K)         : air temperature
        - lw (W/m^2)     : measured longwave radiation (to optimize against)
        - toa (W/m^2)    : top-of-atmosphere insolation
     
    Out:
        - lw_out (W/m^2) : modelled lonwave radiation
        
    Note: 
        
        Input series can have NaNs -- we filter them out for fitting
             
    """    
    
    day_idx=toa>0
    night_idx=toa==0
    clear_idx=np.logical_or(np.logical_and(day_idx,rh<60),\
                        np.logical_and(night_idx,rh<80))
    cloudy_idx=np.logical_or(np.logical_and(day_idx,rh>=60),\
                        np.logical_and(night_idx,rh>=80))
    
    # Update those idxs to filter out NaN
    noNaN_idx=np.logical_and(\
              np.logical_and(~np.isnan(lw),~np.isnan(tk)),~np.isnan(rh))
    
    clear_idx_fit=np.logical_and(clear_idx,noNaN_idx)
    cloudy_idx_fit=np.logical_and(cloudy_idx,noNaN_idx)
    
    x0=np.array([1.0,1.0,1.0])
    fit_clear=minimize(lw_rdk,x0,args=(rh[clear_idx_fit],tk[clear_idx_fit],\
                                       lw[clear_idx_fit]))
    x_clear=fit_clear.x # optimized coefs
    fit_cloudy=minimize(lw_rdk,x0,args=(rh[cloudy_idx_fit],tk[cloudy_idx_fit],\
                                        lw[cloudy_idx_fit]))
    x_cloudy=fit_cloudy.x # ditto
    
    lw_mod=np.zeros(len(lw))*np.nan
    lw_mod[clear_idx]=_lw_rdk(x_clear,rh[clear_idx],tk[clear_idx])
    lw_mod[cloudy_idx]=_lw_rdk(x_cloudy,rh[cloudy_idx],tk[cloudy_idx])
    
    return lw_mod

@jit
def lw_rdk(params,rh,tk,lw):
    
    c1=params[0]
    c2=params[1]
    c3=params[2]
    
    lw_mod=_lw_rdk([c1,c2,c3],rh,tk)
    
    err=RMSE(lw_mod,lw)
    
    return err

@jit      
def _lw_rdk(params,rh,tk):
    
    c1=params[0]
    c2=params[1]
    c3=params[2]
    
    lw_mod=c1+c2*rh+c3*boltz*tk**4   
    
    return lw_mod

@jit
def RMSE(obs,sim):
    n=np.float(np.sum(~np.isnan((obs+sim))))
    rmse=np.nansum((obs-sim)**2)/n
    return rmse
