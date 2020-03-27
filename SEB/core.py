#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Core SEB functions
"""
import numpy as np, pandas as pd
from numba import jit
import scipy.optimize as optimize
from scipy.optimize import minimize
import seb_utils

#_____________________________________________________________________________#
# Define constants. 


# NOTE -- some 'legacy' constants may be included here -- called by functions
# below, but not necessrily in the SEB routine used by Matthews et al. (2020)
#_____________________________________________________________________________#

k=0.40 # Von Karman Constant
z0_m = 5*10**-4# Roughness length for snow (m; Oke )
v=1.46*10**-5 # kinematic viscosity of air  *** CHECK
epsilon=0.622 # Ratio of gas constants for dry/moist air 
rd=287.053 # Gas constant for dry air (J/K/kg)
g=9.81 # Acceleration due to gravity
cp=1005.7 # Specific heat capacity of dry air at constant pressure (J/kg)
zu=zq=zt=2. # Measurement height
Le=2.501*10**6 # Latent heat of vaporization (J/kg)
Ls=2.834*10**6 # Latent heat of sublimation (J/kg)
Lf=3.337*10**5 # Latent heat of fusion (J/kg)
Rv=461.0 # Gas constant for water vapour (J/K/kg)
e0=611.0 # Constant to evaluate vapour pressure in Clasius Clapeyron equation (Pa)
maxiter=10 # Max iterations allowed in the shf/MO scheme 
maxerr=1 # Maximum error allowed for the shf/MO scheme to settle (%)
ext=2.5 # Extinction constant to parameterize attenuation of sw radiation in 
# Beer-Lambert function (see Wheler and Flowers, 2011; units = 1/m)
rho_i=910.0 # Density of ice (kg/m^3)
albedo=0.8# Reflectivity of surface (dimensionless). Median from BC-Rover = 0.4
boltz=5.67*10**-8 # Stefan Boltzmann constant
abs_frac=0.36 # % of net solar radiation absorbed at the surface
emiss=0.98 # Thermal emissivity of ice (dimensionless)
ds=120. # time step of the model (seconds)
tg_mean = 273.15-22.2 # Long-term mean temp used to estimate temp of lowest 
# model layer (K)
depth=20 # Depth to extent the sub-surface model to (m)
inc=0.1 # Thickness of each layer in the sub-surface model 
scalar=24*3600./ds #no. to multiply mean melt/sub by to get daily totals  
compute = True # Control whether SEB is calculated
thresh=2.5 # change in q (seb) within Wheler and Flowers routine must be
# less than this (%)
thick=0.1 # Thickness of the upper-most layer in the Wheeler and Flowers 
# sub-surface scheme (m)
cp_i_c=2097 # Specific heat capacity of the surface layer (J/kg/K) - ice - fixed
rho_top=530. # Density of upper-most snow (or ice) layer (kg/m^3)
t_sub_thresh=273.15-40.0 # If temp of top layer goes below this, heat content
# is taken up by a lower, passive layer - see Wheeler ad Flowers (2011)
ratio=True # True=assume z0_t and z0_q are fixed fraction of z0_m
# (otherwise use the expresssions of Andreas (see ROUGHNESS))
roughness_ratio=10 # Divide z0_m by this to find z0_h and z0_q (if ratio=True)


#_____________________________________________________________________________#
# Begn function definitions

# Turbulent heat fluxes
#_____________________________________________________________________________#

@jit
def USTAR(u,zu,lst,z0_m=z0_m):
    
    """
    Computes the friction velocity.
    
    NOTE: this function also returns the stability corrections!
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - u (m/s) : wind speed at height zu
        - zu (m)  : measurement height of wind speed
        - lst (m) : MO length
        - z0_m (m): roughness length of momentum
        
    Out:
    
        - ust (m/s) : friction velocity
        
    """

    corr_m,corr_h,corr_q=STAB((zu-z0_m)/lst) 
    
    ust=k*u/(np.log(zu/z0_m)-corr_m)
    
    return ust,corr_m,corr_h,corr_q

@jit
def STAB(zl):
    
    """
    Computes the stability functions to permit deviations from the neutral 
    logarithmic profile
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - zl (dimensionless): ratio of ~measurement height to the MO length
        
    Out:
    
        - corr_m (dimensionless) : stability correction for momentum
        - corr_h (dimensionless) : stability correction for sensible heat
        - corr_q (dimensionless) : stability correction for vapour
        
    Note: for the stable case (positive lst), the stability functions of 
    Holtslag and de Bruin (1988) are used; for unstable conditions 
    (negative lst), the functions of Dyer (1974) are applied.
   
    
    """
    # Coefs
    a=1.; b=0.666666; c=5; d=0.35
    
    # Stable case
    if zl >0:
        
       corr_m=(zl+b*(zl-c/d)*np.exp(-d*zl)+(b*c/d))*-1
       
       corr_h=corr_q=(np.power((a+b*zl),(1/b)) + \
       b*(zl-c/d)*np.exp(-d*zl)+(b*c)/d-a)*-1
        
    # Unstable
    elif zl<0:
        
        corr_m=np.power((1-16.*zl),-0.25)
        corr_h=corr_q=np.power((1-16.*zl),-0.5)
    
    # Neutral    
    else: corr_m=corr_h=corr_q=0
               
        
    return corr_m, corr_h, corr_q

@jit
def ROUGHNESS(ust,z0_m=z0_m):
    
    """
    Computes the roughness lenghths for heat and vapour
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ust (m/s) : friction velocity
        - z0_m (m): roughness length of momentum
        
    Out:
    
        - z0_h (m) : roughness length for heat
        - z0_q (m) : roughness length for vapour
        
    Note: when stable, 
    """
    # Reynolds roughness number
    re=(ust*z0_m)/v
    
    # Use the fixed ratio method if requested. Then return (so rest of code
    # will be skipped)
    if ratio:
        z0_h=z0_m/roughness_ratio
        z0_q=z0_m/roughness_ratio
        return z0_h, z0_q, re
    
    assert re <= 1000, "Reynolds roughness number >1000. Cannot continue..."
    
    # Equation terms 
    f=np.array([1,np.log(re),np.power(np.log(re),2)])  
       
    # Coefficients
    # heat
    h_coef=np.zeros((3,3))
    h_coef[0,:]=[1.250,0.149,0.317]
    h_coef[1,:]=[0,-0.550,-0.565]
    h_coef[2,:]=[0,0,-0.183]
    # vapour
    q_coef=np.zeros((3,3))
    q_coef[0,:]=[1.610,0.351,0.396]
    q_coef[1,:]=[0,-0.628,-0.512]
    q_coef[2,:]=[0,0,-0.180]    
    
    # Use roughness to determine which column in coefficient array we should 
    # use
    if re<=0.135: col=0
    elif re>0.135 and re<2.5: col=1
    else: col=2
        
    # Compute stability functions with the dot product
    z0_h=np.exp(np.dot(f,h_coef[:,col]))*z0_m
    z0_q=np.exp(np.dot(f,q_coef[:,col]))*z0_m
    
    return z0_h, z0_q, re

@jit
def MO(t,ust,h,rho):
    
    """
    Computes the MO length. 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - t (K)       : air temperature 
        - ust (m/s)   : friction velocity
        - h (W/m^2)   : sensible heat flux
        - rho (kg/m^3): air density 
        
    Out:
    
        - lst (m)    : MO length
        
    
    Note that we use the air temperature, not the virtual temperature 
    (as Hock and Holmgren [2005]). Lst is positive when h is positive -- 
    that is, when shf is toward the surface (t-ts > 0). This is "stable". 
    When lst is negative, shf is away from the surface and the boundary layer
    is "unstable".
        
    """
    
    lst=t*np.power(ust,3) / (k * g * (h/(cp*rho)))
    
    return lst
        
@jit
def SHF(ta,ts,zt,z0_t,ust,rho,corr_h):
    
    """
    Computes the sensible heat flux.     
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)       : air temperaure
        - ts (K)       : surface temperature
        - zt (m)       : measurment height for air temperature
        - z0_h (m)     : roughness length for heat 
        - ust (m/s)    : friction velocity
        - rho (kg/m^3) : air density 
        - corr_h (none): stability correction 
        
    Out:
    
        - shf (W/m^2)  : sensible heat flux
        
    """    
    
    shf = rho * cp * ust * k * (ta-ts) / (np.log(zt/z0_t) - corr_h)
    
    return shf

@jit
def LHF(ts,qa,qs,zq,z0_q,ust,rho,corr_q): 
    
    """
    Computes the latent heat flux
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ts (K)            : surface temperature
        - qa (kg/kg)        : air specific humdity (at zq)
        - qs (kg/kg)        : air specific humidity (immediately above surface)
        - p pressure (Pa)   : air pressure
        - zq (m)            : measurment height for humidity
        - z0_q (m)          : roughness length for water vapour
        - ust (m/s)         : friction velocity
        - rho (kg/m^3)      : air density 
        - corr_q (none)     : stability correction 
        
    Out:
    
        - lhf (W/m^2)   : latent heat flux
    """
    
    # The direction of the latent heat flux determines whether latent heat 
    # of evaporation or sublimation is applied. The former is only used when 
    # flux is directed toward the surface (qa > qs) and when ts = 273.15.
    L=Ls
    #print qa,qs,ts
    if qa > qs and ts == 273.15:
        
        L = Le
        
    lhf= rho * L * k * ust * (qa-qs) / ( np.log(zq/z0_q) - corr_q )
    
    return lhf
@jit
def LHF_HN(ea,es,ts,p,u):
    
    """
    Computes the latent heat flux according to Hock and Noetzli (1997), eq. 4b
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ea (Pa)         : air vaour pressure (at zq)
        - es (Pa)         : air vapour pressure (immediately above surface)
        - ts (K)          : surface temperature
        - p pressure (Pa) : air pressure
        - u (m/s)         : wind speed
        
    Out:
    
        - lhf (W/m^2)   : latent heat flux
    """
    
    L=Ls

    if ea > es and ts == 273.15:
        
        L = Le
        
    alpha=5.7*np.sqrt(u)
    lhf=L*0.623/(p*cp)*alpha*(ea-es)
    
    return lhf
    
@jit    
def NEUTRAL(ta,ts,rho,qa,qs,u,p,z0_m=z0_m):
    
    """
    This is a convenience function that computes the turbulent heat fluxes 
    assuming a netural boundary layer. 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)            : air temperaure
        - ts (K)            : surface temperature
        - rho (kg/m^3)      : air density 
        - qa (kg/kg)        : air specific humdity (at zq)
        - u (m/s)           : wind speed at height zu
        - p (Pa)            : air pressure 
        - z0_m (m)          : roughness length of momentum
        
    Out:
        - shf (W/m^2)   : sensible heat flux
        - lhf (W/m^2)   : latent heat flux
        - qs (kg/kg)    : surface (saturation) specific humidity 
        
    """    
    
    ust = k*u/(np.log(zu/z0_m))
    z0_t, z0_q, re = ROUGHNESS(ust)
    shf = SHF(ta,ts,zt,z0_t,ust,rho,0)
    lhf = LHF(ts,qa,qs,zq,z0_q,ust,rho,0) 

    return ust, shf, lhf

@jit
def ITERATE(ta,ts,qa,qs,rho,u,p,z0_m=z0_m):
    
    """
    This function coordinates the iteration required to solve the circular
    problem of computing MO and shf (which are inter-dependent).
    
    NOTE: if the iteration doesn't converge, the function returns the 
    turbulent heat fluxes under a neutral profile.
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)            : air temperaure
        - ts (K)            : surface temperature
        - qa (kg/kg)        : air specific humdity (at zq)
        - rho (kg/m^3)      : air density
        - u (m/s)           : wind speed (at zu)
        - p pressure (Pa)   : air pressure
        - z0_m (m)          : roughness length of momentum
        
    Out:
    
        - lhf (W/m^2)   : latent heat flux
    """
    
    # Compute the turbulent heat fluxes assuming a neutral profile:
    ust, shf, lhf = NEUTRAL(ta,ts,rho,qa,qs,u,p,z0_m)

    # Iteratively recompute the shf until delta <= maxerr OR niter >= maxiter
    delta=999
    i = 0
    while delta > maxerr and i < maxiter:
        
        # MO (with old ust)
        lst=MO(ta,ust,shf,rho)
        
        # Ust
        ust,corr_m,corr_h,corr_q = USTAR(u,zu,lst,z0_m)
        
        # Roughness
        z0_h, z0_q, re = ROUGHNESS(ust,z0_m)
        
        # SHF (using the stability corrections returned above)
        shf_new = SHF(ta,ts,zt,z0_h,ust,rho,corr_h)
        
        # Difference?
        delta = np.abs(1.-shf/shf_new)*100.; #print i, delta
        
        # Update old 
        shf = shf_new*1
        
        # Increase i
        i+=1
    
    # Loop exited
    if i >= maxiter: # Use fluxes computed under assumption of neutral profile
        return shf, lhf, (i, re, ta-ts, corr_m) 
    
    else: # Compute LHF using the last estimate of z0_q and corr_q
        lhf = LHF(ts,qa,qs,zq,z0_q,ust,rho,corr_q)
        
        return shf_new, lhf, (i, re, ta-ts, corr_m) 

#_____________________________________________________________________________#
# Ground heat flux/Sub-surface temperatures
#_____________________________________________________________________________#
@jit
def INIT(ta,tg,depth,inc):
    
    """
    This function initializes the sub-surface temperature grid, including 
    setting temperatures for all nodes. It does this by setting the surface
    temperature to ta and the bottom temperature to tg; temperatures 
    inbetween are linearly interpolated. Note that the grid should to extend
    to the same depth reached by the seasonal temperature cycle.
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)            : air temperaure
        - tg (K)            : sub-surface temperature at depth m
        - depth (m)         : the depth at which the grid finishes
        - inc (m)           : the distance between grid points/nodes        
    Out:
    
        - sub_temps (K)     : temperatures at the grid nodes
        - z (m)             : coordinates of the grid nodes (m from surface)
    """  
    
    # First two layers are 4 and 5 cm thick (see Wheler & Flowers, 2011)
    # First 3 layers are therefore not simply inc cm apart
    z=np.array([0.04/2.,0.065, 0.065 + inc/2.+0.05/2.])
    # ... rest are:
    z=np.concatenate((z,np.arange(z[-1]+inc,depth,inc)))

    # Interpolate temps
    sub_temps=np.interp(z,[z[0],z[-1]],[ta,tg])
    
    return sub_temps, z

@jit
def CONST(sub_temps):
    
    """
    This function computes the values of physical 'constants' -- namely the
    specific heat capacity and the themal conductivity (and hence thermal
    diffusivity). See Paterson (1994), p. 205.
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - sub_temps (K)     : temperatures at the sub-surface grid nodes
     
    Out:
    
        - cp (J/K/kg)       : specific heat capacity
        - c (W/m/K)         : thermal conductivity
        - k (m^2/s)         : thermal diffusivity
    """    
    
    # Specific heat capacity (J/kg/K)
    cp = 152.5 + 7.122 * sub_temps
    
    # Thermal conductivity (W/m/K)
    c = 9.828*np.exp(-5.7*10**-3*sub_temps) 
    
    # Thermal diffusivity (m^2/s)
    k = c/(rho_i*cp)   
    
    return cp, c, k

@jit
def SOLAR_ABS(q,z):
    
    """
    This function computes the absorption of solar radiation by the 
    sub-surface. See Wheler and Flowers (2011), Eq. 7.
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - q (W/m^2)         : net flux of solar radiation at the surface 
        - z (m)             : depths of sub-surface grid nodes 
     
    Out:
    
        - qz (W/m^2)        : flux of solar radiation at depth z

    """    

    qz = (1-abs_frac) * q * np.exp(-ext*z)
   
    return qz

@jit
def CONDUCT(sub_temps,z,c,cp_i,seb,sw_yes=False,q=None):
    
    """
    This function computes the conductive heat flux (and its convergence).
    If absorption of solar radiation is requested, it includes the dTdt 
    contribution from this, too.
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - sub_temps (K)     : temperatures at the sub-surface grid nodes 
        - z (m)             : depths of sub-surface grid nodes 
        - c (W/m/K)         : thermal conductivity of ice
        - cp_i (J/kg/K)     : specific heat capacity of ice
        - seb (w/m^2)       : surface energy flux 
        - sw_yes (Bool)     : flag to include absorption of solar radiation
        - q (W/m^2)         : net shortwave flux at the surface
     
    Out:
    
        - dTdt (K/s)       : rate of change of temperature in layers
        
        ** NOTE: not used at present; think the attenuation of SW needs 
        attention (should be the derivative of the flux)
    """       
    
    # Note that fluxes are positive for z increasing (i.e. downward)
    
    # Modify conductivity, c, to be an average value between nodes
    c=1/2.*(c[1:]+c[:-1])
    # Preallocate for output
    nT=len(sub_temps)
    dTdt = np.zeros(nT) 
    f=np.zeros(nT)
    div = np.zeros(nT) 
    h=np.zeros(nT)
    
    
    # Begin computations
    dz = np.diff(z) # distance between nodes
    f[1:] = np.diff(sub_temps)/dz * c # conductance. +ve means upward flux
    f[0] = -seb # +ve up, same convention as conductance
    div[:-1] = np.diff(f) # Flux divergence. Not defined for lowest level
    h[1:-1] = 1/2.*(dz[0:-1]+dz[1:]) # thickness of layers (len=nT-2)
    h[0] = z[0]*2. # thickness of top layer is just 2x first mid-point depth
    h[-1] = 1.0 # Bottom layer's thickness isn't defined...
    dTdt[:-1]=div[:-1]/(cp_i[:-1]*rho_i*h[:-1]) # dTdt for inner nodes (inc. upper);
    # Bottom dTdt = 0 
   
    
    # Include solar heating, if requested
    if sw_yes:
        assert q is not None,  "Must provide net shortwave flux at the surface!"
        qz=SOLAR_ABS(q,z); qz[-1]=0 # Because bottom layer doesn't change temp

        dTdt+=qz/(cp_i*rho_i) # Note: no h multiplication because h appears
        # in numerator and denominator (increasing absorption, increasing mass)
    
    # Return the tendency, flux between layers 1-->N (not layer 0), and 
    # flux divergence
    return dTdt, f, div


def NEW_TEMPS(sub_temps,dTdt,ds,z):
    
    """
    This function computes the temperatures of the sub-surface layers. It 
    also returns the temperature of the surface.
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - sub_temps (K)     : temperatures at the sub-surface grid nodes 
        - dTdt (K/s)        : rate of change of temperature for all layers
        - ds (s)            : time-step
        - z (m)

     
    Out:
    
        - sub_temps_new (K) : updated temperatures for sub-surface layers
        - ts (K)            : updated surface temperature
        
        
    """
    # Simple Euler integration
    sub_temps_new = sub_temps + dTdt * ds
    sub_temps_new[sub_temps_new>273.15]=273.15
    
    # Estimate the surface temperature by extrapolating from layers 2 and 1
    ts = np.min([273.15,\
    sub_temps[0]+(sub_temps[0]-sub_temps[1])/(z[0]-z[1])*-z[0]]) # Note...
    # z[0] is depth from surface, so multiply by -this to 'send' temp change
    # to the surface
    
    # return the updated sub-surface tempratures, along with the new surface 
    # temperature.
    return sub_temps_new, ts
    

#_____________________________________________________________________________#
# Radiative fluxes
#_____________________________________________________________________________#

def SW(sin,albedo=albedo):
    
    """
    This function computes the net shortwave flux at the (ice) surface.
    
    Note that, from Wheler and Flowers (2011), we assume abs_frac of the 
    net sw radiation is absorbed by the surface; the rest is absorbed by
    the sub-surface
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - sin (W/m^2)       : incident shorwave flux 
     
    Out:
    
        - swn (W/m^2)       : net shortwave flux absorbed at the surface  
        - sw_i (W/m^2)       : net shortwave flux at the surface  
    """
    
    swn=(1-albedo)*abs_frac*np.max([sin,0])
    sw_i = np.max([(1-albedo) * sin,0])
    return swn, sw_i

def LW(lin,ts):
    
    """
    This function computes the net longwave flux at the (ice) surface.
       
    Inputs/Outputs (units: explanation): 
    
    In:
        - lin (W/m^2)       : incident longwave flux
        - ts (K)            : surface temperature 
     
    Out: 
        - lwn (W/m^2)       : net longwave flux at the surface
        
    """    
    
    lwn = lin - emiss * boltz * np.power(ts,4)
    
    return lwn
# Note that this is not really a summary/coordinating function 
# (despite computing the SEB), because it has its own itertative behaviour
@jit
def SEB_WF(q,ta,qa,rho,u,p,sw_i,lw,cc,ts):
    
    """
    This function computes the SEB/surface temperature according to MacDougall
    and Flowers (2011) [see their Eq. 6 and surrounding text for logic]
        
        dTs = seb / (rho_top * cp_i_c * thick) * ds
       
    Inputs/Outputs (units: explanation): 
    
    In:
        - q (w/m^2)      : surface energy balance 
        - ta (K)         : air temperature at zt 
        - qa (kg/kg)     : specific humidity at zq
        - rho (kg/m^3)   : air density 
        - u (m/s)        : wind speed at zu
        - p pressure (Pa): air pressure
        - sw (W/m^2)     : incident shortwave radiation
        - lw (W/m^2)     : incident longwave radiation
        - cc (W/m^2)     : cold content 
        - ts (K)         : surface temperature from last iteration
     
    Out: 
        - q (W/m^2)      : surface energy balance 
        - ts (k)         : surface temperature
        - shf (W/m^2)    : sensible heat flux
        - lhf (W/m^2)    : latent heat flux
        - swn (W/m^2)    : net shortwave heat flux
        - lwn (W/m^2)    : net longwave heat flux
        - q_melt (W/m)   : surface energy balance, corrected to account for 
                           cold content compensation
      
    Global constants used
    
        - cp_i_c (J/kg)     : specific heat capacity of ice
        - rho_top (kg/m^3)  : density of top layer
        - thick (m)         : thickness of top layer
        - ds (s)            : model time step (seconds)
        
    """      

    # Change in T
    dTs = q / (rho_top * cp_i_c * thick) * ds
    
    # New temp
    ts += dTs
    
    # Assume no melt energy
    q_melt = 0.
    
    # Adjust ts
    if ts > 273.15: # Needs to be reset, and we need to compute melt energy
                
        # Melt energy left after warming up the surface
        q_melt_scratch = (ts-273.15) * thick * cp_i_c * rho_top * 1./ds
                
        # Take ts back to 273.15
        ts = 273.15
                
        # Push heat into passive layer
        if cc != 0:
            
            # Reduce q_melt
            q_melt = np.max([0,q_melt_scratch-cc])
            
            # Erode the cold content 
            cc = np.max([0,cc-q_melt_scratch]) # subtract from cold content
                    
        # No cold content - all energy can be used to melt    
        else: 
            q_melt = q_melt_scratch*1.
            
    # Thin surface layer has gone too cold    
    if ts < t_sub_thresh:
        
        # Add difference to cold content in passive layer 
        cc += (t_sub_thresh-ts) * thick * cp_i_c * rho_top * 1./ds
        
        # Hold to -t_sub_thresh
        ts = t_sub_thresh
        

    return q,q_melt,ts,cc, -(q-q_melt) # Last term is Qg
        
        
@jit          
def SW_WF(sin,albedo=albedo):
    
    """
    This function computes the net shortwave flux at the (ice) surface.
    
    Note that, from Wheler and Flowers (2011) M2, we here assume all solar 
    radiation is absorbed at the surface
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - sin (W/m^2)       : incident shorwave flux 
     
    Out:
    
        - swn (W/m^2)       : net shortwave flux absorbed at the surface  

    """
    
    swn=(1-albedo)*np.max([sin,0])

    return swn

#_____________________________________________________________________________#
# Coordinating/summarising functions
#_____________________________________________________________________________# 
@jit
def SEB(ta,qa,rho,u,p,sw_i,lw,z0_m=z0_m):
    
    """
    This function computes the temperatures of the sub-surface layers. It 
    also returns the temperature of the surface.
    
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ta (K)         : air temperature at zt 
        - qa (kg/kg)     : specific humidity at zq
        - rho (kg/m^3)   : air density 
        - u (m/s)        : wind speed at zu
        - p pressure (Pa): air pressure
        - sw (W/m^2)     : incident shortwave radiation
        - lw (W/m^2)     : incident longwave radiation
        - z0_m (m)       : roughness length of momentum
        
                           
    Out:
        - shf (W/m^2)    : surface sensible heat flux
        - lhf (W/m^2)    : surface latent heat flux
        - swn (W/m^2)    : surface shortwave heat flux
        - lwn (W/m^2)    : surface longwave heat flux
        - seb_log (W/m^2): surface energy balance
        - ts_log (K)     : surface temperature 
        
        
    """
    # Ta
    ts = ta[0] # Starts equal to the air temperature
    
    # Initialise the sub-surface cold content 
    cc=0
    
    # Preallocate for output arrays
    nt=len(ta)
    shf_log=np.zeros(nt)*np.nan
    lhf_log=np.zeros(nt)*np.nan
    swn_log=np.zeros(nt)*np.nan
    lwn_log=np.zeros(nt)*np.nan
    seb_log=np.zeros(nt)*np.nan
    ts_log=np.zeros(nt)*np.nan
    melt_log=np.zeros(nt)*np.nan
    sub_log=np.zeros(nt)*np.nan
    nit_log=np.zeros(nt)*np.nan
    qg_log=np.zeros(nt)*np.nan
    
    # Begin iteration
    for i in range(nt):
        
        # Store ts before it is changed by SEB_WF (below).  Its value
        # now sets behaviour for LW and turb. 
        ts_log[i]=ts  
        
        # Skip all computations if there any NaNs at this time-step [note that
        # means that surface temperature will be unchanged]
        if np.isnan(ta[i]) or np.isnan(sw_i[i]): continue 
        
        # Compute surface vapour pressure (at ts)
        qs = seb_utils.VP2Q(seb_utils.SATVP(ts),p[i])
        
        # Compute the turbulent heat fluxes 
        if u[i] > 0 and (ta[i]-ts!=0):
            shf, lhf, meta = ITERATE(ta[i],ts,qa[i],qs,rho[i],u[i],p[i],\
                                    z0_m=z0_m); 
        else: shf=0; lhf=0
        
        # Compute the radiative heat fluxes
        lwn = LW(lw[i],ts); 
        
        # Swn
        swn=SW_WF(sw_i[i],albedo)
        
        # Initial SEB 
        seb=shf+lhf+swn+lwn
         
        # Compute available melt energy and adjust surface temperature 
        # (also keeps account of sub-surface temperature and cold content)
        seb,seb_melt,ts,cc, qg =\
        SEB_WF(seb,ta[i],qa[i],rho[i],u[i],p[i],sw_i[i],lw[i],cc, ts)

        # Compute melt and sublimation here -- melt energy returned by 
        # SEB_WF is different from the SEB (because cc was accounted for)
        melt_log[i],sub_log[i] = MELT_SUB(ts,lhf,seb_melt,ds)   
               
        # Log ground heat flux 
        qg_log[i]=qg                                     
        # Log seb 
        seb_log[i]=seb # same between WF and non-WF
        # Log shf
        shf_log[i]=shf
        # Log lhf
        lhf_log[i]=lhf
        # Log lwn
        lwn_log[i]=lwn
        # Log lhf
        swn_log[i]=swn        

    return shf_log, lhf_log, swn_log, lwn_log, seb_log, ts_log, \
    melt_log,sub_log,nit_log, qg_log

@jit
def MELT_SUB(ts,lhf,seb,ds):
    
    """
    This function computes mass loss via melt and sublimation. 
    
    Notes: If seb is +ve and ts == 273.15: melt = seb/Lf. Otherwise, melt = 0
           If lhf is -ve: sublimation = -lhf/Ls. Else if lhf is +ve 
           and ts < 273.15: sublimation = -lhf/Le
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ts (K)         : surface temperature
        - lhf (W/m^2)    : latent heat flux
        - seb (W/m^2)    : surface energy balance
        - ds (s)         : model time step
     
    Out:
        - melt (mm we)   : melt 
        - sub (mm we)    : sublimation (+ve); resublimation (-ve)
             
    """    
    
    # Melt
    if ts == 273.15:
        melt=np.max([0,seb/Lf])*ds
    else: melt = 0
    
    # Sublimation
    if lhf <=0:
        sub=-lhf/Ls * ds 
    else:
        if ts <0: # Resublimation
            sub=-lhf/Ls * ds
        else: sub=-lhf/Le # Condensation

    return melt, sub

