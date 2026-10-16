import sys
import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset # !! new - building netcdf
import os # file saving 

P0 = 1013. # surface (sea level) pressure (mb)
T0 = 288. # surface temperature (K)
GAMMA = 0.0065 # constant lapse rate (ºC/m)
RHOI = 920 # density of ice (kg/m^3)
EPS = 0.622 # constant for water vapour calculation
PI = 3.14159 # constant pi 
RD = 287.05 # gas constant for dry air (J/kg/K)
RV = 461.51 # gas constant for water vapor (J/kg/K)
TK = 2.32e-2 # thermal conductivity of air (W/m/K) ??
D = 2.11e-5 # molecular diffusivity of water vapor in air (m^2/s) ??
TTR = 273.16 # triple point temperature of water (K) ??
RW = 461.51 # gas constant for water vapor (J/kg/K)
LS = 2.837e6 # latent heat of sublimation (J/kg)
LF = 3.12e5 # latent heat of fusion (J/kg)
LV = 2.5e6 # latent heat of vaporization (J/kg)
NMAX = 500.e6 # 
DIS = 0.25 # dispersion 
GRAV = 9.81 # gravity
ETA = 1.67e-5 # dynamic viscosity of air (Pa*s) ?
CPW = 4.27e3 # specific heat of water (J/kg/K)
nsh = 0
nnu = 0
lwc = 0
rad = 0
nre = 0

""" 
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 defining sub-functions 
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
"""
def vapour(t):
    """
    Goff-Gratch formula for water saturation vapour pressure.
    
    Args:
        t (float): Temperature.
    Returns:
        float: Saturation vapour pressure.
    """
    arg1 = 11.344 * (1. - t / 373.16)
    arg2 = 3.49149 * (1. - 373.16 / t)
    e = (-7.90298 * (373.16 / t - 1.) + 5.02808 * math.log10(373.16 / t)
         - 1.3816e-7 * (10. ** arg1 - 1.)
         + 8.1328e-3 * (10. ** arg2 - 1.))
    v = 1013.246 * 10. ** e
    return v
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def gr(tempr):
    """
    The Growth Rates and Densities of Ice Crystals between −3°C and −21°C (Ryan et al., 1976)
    dr/dt values from Ryan et al's lab expts for diffusional growth of ice.
    
    Args:
        tempr (float): Temperature.
    Returns:
        float: Growth rate.
    """
    if tempr >= -3.5:
        return 0.2
    if -4.5 <= tempr < -3.5:
        return 0.5
    if -5.5 <= tempr < -4.5:
        return 1.0
    if -6.5 <= tempr < -5.5:
        return 1.25
    if -7.5 <= tempr < -6.5:
        return 0.75
    if -8.5 <= tempr < -7.5:
        return 0.5
    if -9.5 <= tempr < -8.5:
        return 0.3
    if -10.5 <= tempr < -9.5:
        return 0.3
    if -11.5 <= tempr < -10.5:
        return 0.35
    if -12.5 <= tempr < -11.5:
        return 0.5
    if -13.5 <= tempr < -12.5:
        return 0.6
    if -14.5 <= tempr < -13.5:
        return 1.3
    if -15.5 <= tempr < -14.5:
        return 1.8
    if -16.5 <= tempr < -15.5:
        return 1.3
    if -17.5 <= tempr < -16.5:
        return 0.8
    if -18.5 <= tempr < -17.5:
        return 0.7
    if -19.5 <= tempr < -18.5:
        return 0.6
    if -20.5 <= tempr < -19.5:
        return 0.5
    if tempr < -20.5:
        return 0.4
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def trev(pbase_, tbase_, p):
    """
    Calculate adiabatic temperature and liquid water content at pressure p.
    The air parcel starts at cloud base with pressure pbase_ and temperature tbase_.

    Args:
        pbase_ (float): Cloud-base pressure (mb).
        tbase_ (float): Cloud-base temperature (°C).
        p (float): Pressure at the level being calculated (mb).

    Returns:
        tuple: Adiabatic temperature (°C) and liquid water content (g/m³).
    """
    eps = 0.622
    cpd = 1.0042e3
    cw  = 4.218e3
    rd  = 287.05
    alhv = 2.501e6

    tk = tbase_ + 273.15
    e  = vapour(tk)
    r  = eps * e / (pbase_ - e)
    cpt = cpd + r * cw
    thetaq = tk * (1000.0 / (pbase_ - e))**(rd / cpt) * math.exp(alhv * r / (cpt * tk))

    # 1st approximate t
    t1 = tk
    e = vapour(t1)
    rv = eps * e / (p - e)
    t1 = thetaq / ((1000.0 / (p - e))**(rd / cpt) * math.exp(alhv * rv / (cpt * t1)))

    # Successive approximations
    for _ in range(10):
        e  = vapour(t1)
        rv = eps * e / (p - e)
        t1 = (thetaq / ((1000.0 / (p - e))**(rd / cpt) * math.exp(alhv * rv / (cpt * t1))) + t1) / 2.0
    
    t_c = t1 - 273.15
    # LWC from mixing ratio difference
    e  = vapour(t1)
    rv = eps * e / (p - e)
    tw = r - rv  # difference in mixing ratios
    # Convert to g m^-3 (matches trev.c line)
    alwc_gm3 = tw * p * 28.9644 / (8.314e7 * t1) * 1.0e9
    return t_c, alwc_gm3
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def ztp(zz):
    """
    Calculate pressure at a given altitude. (Uses a constant lapse rate)

    Args:
        zz (float): Altitude (m).
    Returns:
        float: Pressure (mb) at altitude zz.
    """    
    p = P0 * ((T0 - GAMMA * zz) / T0) ** (GRAV / (RD * GAMMA))
    return p
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def ptz(p):
    """
    Calculate altitude at a given pressure. (Uses a constant lapse rate)

    Args:
        p (float): Pressure (mb).
    Returns:
        float: Altitude (m) at pressure p.
    """
    ex = (RD * GAMMA) / GRAV
    ptzr = T0 * (1.0 - (p / P0) ** ex) / GAMMA
    return ptzr
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def drag(r, rhog, rhoa_):
    """
    Calculate the Reynolds number and drag coefficient for a particle.

    Args:
        r (float): particle radius. (m)?
        rhog (float): gruapel particle density.
        rhoa_ (float): air density.
    Returns:
        float: Drag coefficient.
    Note:
        The Reynolds number is stored in the global variable `nre`.
    """

    global nre # Reynolds number

    # Best (or Davies) number for current pressure level
    xd = 32. * rhog * rhoa_ * GRAV * r ** 3 / (3. * ETA * ETA)

    if xd < 1.09e4:
        a = 0.0688
        b = 0.769
    elif xd < 6.58e5:
        a = 0.347
        b = 0.595
    else:
        a = 3.6184
        b = 0.420

    nre = a * xd ** b
    cd = 8. * nre ** (-0.27)
    return cd
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def tsurf(kap, tsk):
    """
    Calculate the surface temperature of the growing graupel particle,
    using the relationships derived in Pflaum and Pruppacher (1979).

    Args:
        kap (float): collection kernal for the graupel particle ???
        tsk (float): Temperature of the environment/ambient air (K).
    Returns:
        float: Surface temperature of the graupel particle (K).
    Note:
        The calculation iterates until the change in surface temperature
        is less than 0.10 K, or until the iteration limit is reached.
    """
    global es # water vapour pressure

    cpv = 1.87e3
    ci = 2.031e3
    mu = 1.667e-5
    to = 273.15
    ik = 0

    # mixing ratio
    es = 100. * vapour(tsk)
    rve = es / (RV * tsk)

    # Schmidt number - not used
    #nsc = mu / (rhoa * D)
    # nsh2 = 2. + 0.6 * nsc ** 0.333 * nre ** 0.5

    # Prandtl number - not used
    #npr = mu * cpv / TK
    # nnu2 = 2. + 0.6 * npr ** 0.333 * nre ** 0.5

    tm1 = kap * lwc / (rad * PI)

    # calculate surface temperature (iteratively)
    tsr = tsk
    tdiff = 5.
    while tdiff > 0.10:
        ik += 1
        tsold = tsr
        esi_local = 100. * vapour(tsr)
        rvs_local = esi_local / (RV * tsr)
        ts1 = tm1 * (LF + CPW * (tsk - to) + ci * to)
        ts2 = 2. * D * LS * nsh * (rve - rvs_local) + 2. * TK * tsk * nnu
        tsr = (ts1 + ts2) / (2. * TK * nnu + tm1 * ci)
        tdiff = abs(tsr - tsold)
        if ik > 100 and tsr > 273.15:
            tsr = 273.15
            break

    return tsr
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def make_nc_variable(nc_group, data, name, dimensions, units, long_name, data_type="f4", fill_value_f=np.nan):
    """
    Create a NetCDF variable in the specified group with the given attributes.
    
    Args:
        nc_group: The NetCDF group to create the variable in.
        data: The data to be stored in the variable.
        name: The name of the variable.
        dimensions: The dimensions of the variable.
        units: The units of the variable.
        long_name: The long name of the variable.
        data_type: The data type of the variable, default is "f4".

    Returns:
        netCDF4.Variable: The created NetCDF variable.
    """
    var = nc_group.createVariable(name, data_type, dimensions, fill_value=fill_value_f)
    var.units = units
    var.long_name = long_name
    var[:] = data
    return var

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# main helper functions - for thermal position
def find_grid_index(height_km, alt):
    if not np.isfinite(height_km):
        return None

    alt_km = alt / 1000.

    for ii in range(1, len(alt_km)):
        if alt_km[ii - 1] <= height_km < alt_km[ii]:
            return ii

    return None

def boundary_velocity(index, vv):
    if index is None:
        return 0.
    return vv[index] / 2.