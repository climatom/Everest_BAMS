#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Computes the energy balance. 

Notes that computation requires: 
    
    - Temp (in K)
    - Pressure (in Pa)
    - RH (fraction)
    - Ws (m/s)
    - Incident short/longwave radiation (W/m**2)
    
"""
# SEB code repository
import core # Library of SEB computation routines
import seb_utils # Library of helper functions (e.g. convert met vars)
# Public modules
import pandas as pd
import numpy as np
import datetime

# Local functions
# returns number of nans in an array
def count_nan(arrin):
    return np.sum(np.isnan(arrin))

# Options/Parameters
# Note that these data have undergone the light QC described in 
# Matthews et al (2020)
din="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/AWS/"
fs=["south_col_filled.csv","c2_filled.csv","summit_guess.csv"]
outs=["south_col","c2","summit"]
end_date=datetime.datetime(year=2019,month=11,day=1)
odir="/home/lunet/gytm3/Everest2019/Research/BAMS/Data/SEB/"

# Preallocation
out={}
archive={}
input_data={}
        
# Params
# Z0_m is taken as the mean for snow on glacier surfaces in low lats -- from
# Brock et al. (2006)
z0_m = 2.7*1e-3
# Range is 5th and 95th percentiles of that sample
z0s = [0.9*1e-3,z0_m,5.7*1e-3]

# Loop files 
for jj in range(len(fs)):
            
            # Update
            print "Computing SEB for: ",fs[jj]

            # Read in data
            fin=din+fs[jj]
            data=pd.read_csv(fin,sep=",",parse_dates=True, index_col=0,\
                             na_values="NAN")

            # Tuncate to ensure no more recent than 'end_date'
            data=data.loc[data.index<=end_date]
            
            # Archive it (not the interpolated)
            archive[fs[jj].replace(".csv","")]=data
            
            # Resample to higher frequency
            freq="%.0fmin" % (core.ds/60.)
            data=data.resample(freq).interpolate("linear")
                    
            # Assignments (for readability)
            ta=data["T_HMP"].values[:]+273.15
            ta2=data["T_109"].values[:]+273.15
            p=data["PRESS"].values[:]*100.
            rh=data["RH"].values[:]/100.      
            sw=data["SW_IN_AVG"].values[:].astype(np.float)
            lw=data["LW_IN_AVG"].values[:]
            
            # Wind speed names differ between files - deal with that here. 
            try:
                u=(data["WS_AVG"].values[:]+data["WS_AVG_2"].values[:])/2.
            except:
                u=data["WS_AVG"].values[:]
                
            # Met conversions
            vp=np.squeeze(np.array([seb_utils.SATVP(ta[i])*rh[i] for i in range(len(u))]))
            mr=seb_utils.MIX(p,vp)
            tv=seb_utils.VIRTUAL(ta,mr)
            qa=seb_utils.VP2Q(vp,p)
            rho=seb_utils.RHO(p,tv)
                                                           
            # Compute the SEB - with range of z0_m
            out[jj]=[]
            for zi in z0s:
                shf,lhf,swn,lwn,seb,ts,melt,sub,nit_log,qg = \
                core.SEB(ta,qa,rho,u,p,sw,lw,z0_m=zi)
                   
                # Store in DataFrame for easier computation/plotting 
                # indices 0: SCol; 1: Camp II; 2: Summit. For each 
                # out[jj], the energy balance is computed for a n-range
                # set of z0_m -- stored in a list
                scratch=pd.DataFrame({"Melt":melt,"Ablation":melt+sub,"T":ta,\
                              "shf":shf,"lhf":lhf,"sw":swn,"lw":lwn,\
                              "seb":seb,"qg":qg,"qmelt":melt*core.Lf/core.ds,\
                              "wind":u,"rh":rh,"ts":ts,"sub":sub,"press":p},\
                              index=data.index)
                               
                # Write this out
                oname=odir+outs[jj]+"_z0_%.3f.csv"%(zi*1000)
                scratch.to_csv(oname,float_format="%.6f")
                # Summarise stats and write out, too (recycling oname)
                mu=scratch.mean()
                oname=odir+outs[jj]+"_mean_z0_%.3f.csv"%(zi*1000)
                mu.to_csv(oname,float_format="%.6f")
                std=scratch.resample("D").mean().std()
                oname=odir+outs[jj]+"_std_z0_%.3f.csv"%(zi*1000)
                std.to_csv(oname,float_format="%.6f")

                # store out in a list - for plotting
                out[jj].append(scratch)
 
                print("Computed SEB for z0_m = %.5fm"%zi)
    
            input_data[fs[jj].replace(".csv","")]=data
         
            # Update
            print("Computed SEB for file: %s" %fin )

