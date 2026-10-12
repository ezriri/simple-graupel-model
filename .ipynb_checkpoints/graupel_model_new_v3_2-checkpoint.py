## very very different from v1 -> include secondary ice - whole of graupel() is changed
## v3 netcdf + namelist added. _1 -> made changes to how simulate thermals
## also save inital conditions - so +1 to final time 
## also have particle concentration through time - using the fixed distribution

"""
Python translation of graupel_thermals_ezri.c

precip + icemult -- A wee model to predict the development of precipitation + HM processes too.

  Design:  a) updraught assigned based on combination of radar
              and aircraft observations; vertical size constant
           b) terminal velocity of particles based on particle
              size and observations of particle types in KA87 project
           c) liquid water content profile based on aircraft observations
           d) growth by vapour diffusion and riming according to graupel model
           e) ice crystal growth from Ryan et al. according to T.
           f) initial spectrum from limited aircraft observations in KA87
              project and various sailplane measurements; start at cloud
              top at various temperatures
           g) position of particles relative to ground and to the top of
              the thermal calculated
           h) output the particle size distribution and particle positions
              in the vertical
           i) note concentrations based on exponential fit and doesn't change.
           j) lwc depleted above 8 km due to conversion to ice

  Ice multiplication design:
           a) the Hallett-Mossop zone is divided up into 1 deg C sub-zones
           b) splinters are produced in each of these sub-zones at the rate
              given by the formula Nx = rime * Ng * H; rime being the amount
              of rime gathered in the time interval, and H being the rate
              of splinter production per mg of rime collected
              (Other approaches will be tried later - 15/8/94.)
           
TRANSLATION NOTES
-----------------
1. `trev()` (adiabatic temperature/LWC vs pressure) is *declared* in the C
   source (`float trev();`) and called, but its body is not in this file --
   it must live in another .c file that wasn't supplied. A stub is provided
   below; replace it with the real implementation before running.

2. `vapour()` and `gr()` *are* fully defined in this file (unlike the
   previous graupel-only file you gave me, where they were only declared),
   so those are translated faithfully below.

3. `scale()` is called once (to pick y-axis limits for the "f vs time"
   plot) but is not defined in this file -- it's presumably another
   library routine from the same custom graphics package as `line()`,
   `axes()`, etc. A simple equivalent (`scale_axis()`) is provided below;
   it picks integer-rounded min/max bounds from the data, which is
   consistent with how its result (`nticks = maxy - miny + 1`) is used.

   
4. `BAD` is a sentinel value used throughout the C code to flag "missing/
   invalid" data points, but it's a macro from "cdfhdr.h" which was not
   supplied. Given the code checks `value < 99999.` to decide whether a
   value is valid, BAD is assumed here to be 999999.0. Adjust
   `BAD` below if your header defines it differently.

5. The C program draws all its plots using a custom, in-house graphics
   library (`gopen`, `gclear`, `window`, `axes`, `line`, `label`, `mark`,
   `conto`, `grid`, `fill`, `gpause`, `gclose`, `xtick`, `ytick`, `badset`).
   None of these are standard C library calls and their implementations
   were not supplied (they live in "cdfhdr.h" / a linked graphics library).
   It is not possible to reproduce their exact visual behaviour, so each
   plotting block has been translated into an equivalent, clearly-labelled
   matplotlib figure that plots the same data. The *numerical* content
   (the environment profiles, thermal tracking, particle growth, and
   reflectivity calculations) is translated exactly.

6. All C globals shared between functions are kept as Python module-level
   globals (via `global` statements) to preserve the original data flow --
   this is particularly important here since `graupel()` is called
   repeatedly (once per initial particle, then once per Hallett-Mossop
   splinter group -- and the splinter-group loop itself grows dynamically
   as `graupel()` spawns new splinter groups, exactly as in the C `for`
   loop whose bound `hm - 1` is re-evaluated on every iteration).
"""

import sys
import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset # !! new - building netcdf
import os # file saving 

# ---------------------------------------------------------------------------
# # namelist inputs
# ---------------------------------------------------------------------------
pyfile_name = sys.argv[0] # python file name
namelist_txt = sys.argv[1] # namelist file name

with open(namelist_txt, "r") as f:
    namelist_contents = f.read()
namelist = {} ## all the namelist inputs are stored in this dictionary
exec(namelist_contents, {"np": np}, namelist)

## saving nc file
namelist_file = namelist_txt.split("/")[-1]
nlsit_split = namelist_file[:-4].split("_")
if len(nlsit_split) > 1:
    save_name = "_".join(nlsit_split[1:])
else:
    save_name = 'ctrl'

diam = namelist['diam']
numt = int(namelist['runtime'] * 60. / namelist['DELTIM']) ### number of simulation time steps (s)
numt_0 = numt + 1 # including initial time step (t=0)
MAX1 = len(diam)   # number of particle size categories

# ---------------------------------------------------------------------------
# #define constants
# ---------------------------------------------------------------------------
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


# the whole environment is initialised with empty arrays defined by MAX values
lw = np.zeros(namelist['MAX2'])                              # LWC (at all heights)
db = np.zeros(namelist['MAX2'])                              # dbar ??
vv = np.zeros(namelist['MAX2'])                              # vertical velocity
alt = np.zeros(namelist['MAX2'])                             # altitude (m?)
ps = np.zeros(namelist['MAX2'])                              # pressure (mb?)
tr = np.zeros(namelist['MAX2'])                              # temperature (Kelvin)
conc = np.zeros(MAX1)                            # concentration of particles in each size category

radi = [d/2 for d in diam]  #np.zeros(MAX1)      # inital radius
stime = [] # time of splinter production
xhm = [] # splinter position (horizontal)
zhm = [] # splinter position (vertical)
nhm = [] # number (conc) of splinters in each group for each zone
hmi = [] # splinter time index for each group

''' multiple thermals defined '''
thtop1 = np.zeros(numt_0)                          # top alt of thermal 1
thbase1 = np.zeros(numt_0)                         # bottom alt of thermal 1
thtop2 = np.zeros(numt_0)                          # top alt of thermal 2
thbase2 = np.zeros(numt_0)                         # bottom alt of thermal 2
cldtop = np.zeros(numt_0)                          # top of cloud top debis
downbase = np.zeros(numt_0)                        # bottom of cloud top debris


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


""" 
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 main functions 
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
"""
def graupel():
    """
    if splinter == 0:
    Grow 1 graupel particle (channel `j`) in the environment, through all time steps.
    !! OR !!
    if splinter == 1:
    Grows a splinter particle (a HM splinter)

    j is a reference loc for all variables
    
    ! there is a lot happening in this function !
    2 sub loops in this function:
    loop through each time step
        loop through cloud drop size spectrum (24 channels) - for riming
    I have broken it up into 11 sections - seperating out specific processes

    ** going through each time step, the following is calculated **
    1. starting altitude of particle (in environment)
    2. where is the particle with respect to the thermal (updraft / downdraught)
    3. calculate drag and terminal velocity for the particle
    4. calculate rate of diffusional growth of ice crystals
    5. calculate vapour pressure
    6. calculate rimming ** looping through cloud drop size spectrum 6.1-6.3 **
        6.1 Collection kernel calculation - for each channel
        6.2 Impact velocity of cloud drops calculation
        6.3 Density of newly formed rime calculation
        6.4 Surface temperature calculation
        6.5 Fraction of unfrozen water shed
    7. calculate change in radius of graupel particle (from mass / diffusional growth)
    8. changes in mass and bulk density
    9. Hallett-Mossop time
    10. Move the particle

    """
    global j, hm, nre, nsh, nnu
    global vtsave, ddsave, drtrans 
    global lwc, rad

    # initialise local variables (mainly particle / environment)
    z = x = pres = temp = tempc = dbar = 0.0 # z = vertical position of particle (m)// x = horizontal position of particle (km?) //  pressure (mb) // temp = temperature (K) // tempc = temperature (C) // dbar = mean cloud droplet diameter
    zold = xold = 0.0 # previous position of particle (m, km)
    zkm = 0.0 # vertical position of particle (km)
    zdown = 0.0 # altitude in which particle enters downdraught (km)
    hw = 0.0 # m/s horizontal wind
    lwc7 = 0.0 # lwc at 7km

    tflag = 0 # tflag == 0 -> frozen growth // tflag == 1 -> warm growth
    one = 0 # initialisation of particle (0 = first time step, 1 = subsequent time steps)
    indx = 0
    downflag = 0 # is particle in downdraught? (0 = no - normal / thermal envi, 1 = yes)
    topflag_local = 0
    graup = 0 # dictate graupel growth occuring (0 = no, 1 = yes) - changes in transition period
    zflag = 0 # control LWC depletion above 7km
    hmrime = 0. # rime mass accumulated while moving through HM zone
    hmzone1 = hmzone2 = hmzone3 = hmzone4 = hmzone5 = 0 # different HM zones for splinter growth

    # physical quantities 
    drtrans = 0.
    dr300 = None
    delrd = 0.
    ts = None
    mass = None
    rhog = None
    kapb = None
    rhor = None
    delmr = 0.
    time = 0.
    nx = 0.

    ## all the particle variables we are saving
    diameter_s = [] # diameter of particle (m)
    density_s = [] # density of particle (kg/m^3)
    mass_s = [] # mass of particle (kg)
    terminal_velocity_s = [] # terminal velocity of particle (m/s)
    surface_temp_s = [] # surface temperature of particle (K)
    altitude_s = [] # altitude of particle (m)
    horizontal_position_s = [] # horizontal position of particle (km)
    horizontal_wind_s = [] # horizontal wind (m/s)
    conc_s = [] # concentration of particles (n/m3)

    # are we growing a splinter particle or a graupel particle?
    # splinter == 0 -> graupel particle // splinter == 1 -> splinter particle
    if splinter == 1:
        j = MAX1 + hmj
        istart = hmi[hmj] # doesnt start at timestep 0 - dynamically create new particles during the simulation
        
        # if splinter particle, need to fill in time before created with empty nan values, so all arrays are same size
        empty = [np.nan] * istart
        diameter_s.extend(empty)
        density_s.extend(empty)
        mass_s.extend(empty)
        terminal_velocity_s.extend(empty)
        surface_temp_s.extend(empty)
        altitude_s.extend(empty)
        horizontal_position_s.extend(empty)
        horizontal_wind_s.extend(empty)
        conc_s.extend(empty)
    else:
        istart = 0
    pos = 0

    # time-step growth loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # this is the main part, go through each time step and calculate growth of particle
    
    #for i in range(istart, numt - 1):
    for i in range(istart, numt):
        # we do -1 as we save the initial conditions of the particle in the first loop
        ### 1. starting altitude of particle (in environment) - - - - - - - - - - - - - - 
        if one == 0:
            # first round - get initial conditions
            if splinter != 0: # if particle is a splinter particle, get initial conditions from the graupel particle that created it
                z = 1000 * zhm[hmj]
                x = xhm[hmj]
                time = stime[hmj]
            else: # graupel particle
                z = namelist['zinit']
                x = 0.
                time = 0.
            zold = z
            xold = x

            # ** add inital conditions **
            altitude_s.append(z) # [0]
            horizontal_position_s.append(x) # [0]
			

        # loop for graupel environment values -- get pointer to initial height
        for ii in range(namelist['MAX2']):
            if ii > 0:
                if z < alt[ii] and z >= alt[ii - 1]: # what alt level is particle at?
                    pos = ii
        pres = ps[pos] * 100.
        zkm = z / 1000.

        # 2. where is the particle with respect to the thermal? * * * * * * * * * * * * * * * * * * * * * * * * * *
        # this modifies the environmental conditions -> alter particle growth and movement
        hw = 0 # horizontal wind (m/s) # not persistent between time steps
        if downflag == 0: # particle is not in downdraft
            if ((zkm <= thtop1[i] - namelist['ZCT'] and zkm >= thbase1[i]) or (zkm <= thtop2[i] - namelist['ZCT'] and zkm >= thbase2[i])):
                # particle is still in thermal (can be thermal 1 or thermal 2)
                temp = tr[pos] - 1. # thermal envi is 1ºC cooler than the environment
                lwc = namelist['RATTH'] * lw[pos]
                dbar = db[pos] # drop size
                wi = vv[pos] # vertical velocity - based on the vertical velocity profile of the thermal

            # in cloud top region (top of thermal 1)
            if zkm <= thtop1[i] and zkm >= thtop1[i] - namelist['ZCT']:
                temp = tr[pos] - 1.
                lwc = namelist['RATCT'] * lw[pos]
                dbar = db[pos]
                hw = umax
                # work out smooth vertical velocity in cloud top regions
                wi = vv[pos] - vv[pos] * (zkm - (thtop1[i] - namelist['ZCT'])) / (2. * namelist['ZCT']) # vertical velocity change with height

            # out of thermal -> in debris under thermal // above thermal (if at start)
            if ((zkm > thtop1[i] or zkm < thbase1[i]) and (zkm > thtop2[i] or zkm < thbase2[i]) and (zkm < downbase[i])) or (zkm > thtop1[i] and zkm > thtop2[i]):
                temp = tr[pos] - 1.
                lwc = namelist['RATDEB'] * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = 0.0

            # above thermal 1, within cloud top, in descending thermal remnants
            if zkm > thtop1[i] and zkm <= cldtop[i] and zkm >= downbase[i]:
                temp = tr[pos] - 2.
                lwc = namelist['RATDEB'] * lw[pos]
                dbar = db[pos]
                hw = 0.
                if cldtop[i] > namelist['GENCTOP']:
                    # cloud top above general cloud top -> prescribed verical velocity
                    wi = namelist['WCTDEB']
                else:
                    wi = 0.

            # horizontal position in km
            if x >= namelist['TWID']:
                # particle has left the thermal, into downdraught
                temp = tr[pos] - 2.
                lwc = namelist['RATDOWN'] * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = -5.
                downflag = 1 # in downdraft
                zdown = zkm

        else:
            # particle is in downdraft
            temp = tr[pos] - 2.
            lwc = namelist['RATDOWN'] * lw[pos]
            dbar = db[pos]
            hw = 0.
            wi = -5.
            # make downdraughts no deeper than DDEPTH km
            if abs(zdown - zkm) >= namelist['DDEPTH']:
                # reset particle to normal environment
                downflag = 0
                xold = 0
                x = 0.
                zdown = 0.

        # overshoot cloud top?
        if zkm >= namelist['CTOP'] or zkm > cldtop[i]:
            zkm = cldtop[i]

        # deplete the lwc if above 7 km (due to conversion to ice)
        if zkm >= 7:
            if zflag == 0:
                lwc7 = lwc
                zflag = 1
            if zflag > 0:
                lwc = lwc7 - (zkm - 7.) * lwc7 / (10. - 7.)
            if zkm >= 10.:
                lwc = 0.

        # density of air
        rhoa = pres / (RD * temp)
        # kinematic viscosity (not used)
        nu = ETA / rhoa
        tempc = temp - 273.15 # temp ºC

        # only on first iteration of loop
        if one == 0:
            # initial radius in m and density kg m^-3
            if splinter < 1:
                rad = radi[j] * 1.0e-3 # in m
            else:
                # if splinter particle
                rad = namelist['HMRINIT']
            rhog = 900. # initial ice density (kg m^-3)
            # initial mass
            mass = 4. * PI * rad ** 3 * rhog / 3.
            '''
            # calculate vt and delmr for a 300 um diameter particle
            # Assume density of 100 g/cm^3 just now as a quick fix - 7/7/94
            # and use simple formula for kapb
            cd = drag(0.15e-3, 100., rhoa) ## ?? this isnt used anywhere
            
            # construct 200-300um transition for vt and delmr
            #if rad * 2.0e3 < 0.2:
            vt300 = 0.6 * 0.3
            # force dr300 to slightly larger than largest value of dr
            # by diffusion
            dr300 = 2.0e-6
            '''
            # calculate drag coefficient for 300 um particle
            cd = drag(0.15e-3, rhog, rhoa) ## ?? this isnt used anywhere

            vt300 = 8. * 0.15 * 1.0e-3 * 220. * GRAV / (3. * rhoa * cd) # terminal velocity for 300um particle
            dmr300 = PI * 0.15e-3 ** 2 * vt300 * lwc * namelist['DELTIM'] # riming mass change for 300um particle
            dr300 = dmr300 / (4. * PI * 0.15e-3 ** 2 * 220.) # radius change for 300um particle
            # these values ^^ are used for smooth transition for riming between 200-300 um diameter particles
            


        # 3. calculate drag and terminal velocity for the particle * * * * * * * * * * * * * * * * * *
        cd = drag(rad, rhog, rhoa)

        # terminal velocity is taken as a function of temperature according
        # to the graph of Fukuta et al. 1982, Chicago conference.
        # If initial diam is greater than 200 um, go straight to vt for graupel
        dmm = rad * 2.0e3
        if dmm <= 0.30:
            # for small particles, use empirical relationship, based on temp
            if dmm <= 0.2:
                if (-6.5 < tempc <= -4.) or (-16 < tempc <= -14.):
                    vt = 100. * (0.395 * dmm - 0.177 * dmm ** 2 + 0.073 * dmm ** 3 - 0.0153 * dmm ** 4)
                if (-14. < tempc <= -12) or (-8 < tempc <= -6.5) or tempc > -4:
                    vt = 40. * dmm
                if -12 < tempc <= -8:
                    vt = 51. * dmm
                if tempc <= -16:
                    vt = 60. * dmm
                vt *= 0.01
                vtsave = vt
            else:
                if graup == 0:
                    vt200 = vtsave ## unsure of what is happening here ?? - terminal velocity for 200um particle
                    vt = vt200 + (vt300 - vt200) * (dmm - 0.2) / 0.1
        else:
            vt = 8. * rad * rhog * GRAV / (3. * rhoa * cd)
            vt = vt ** 0.5

        # 4. calculate rate of diffusional growth of ice crystals * * * * * * * * * * * * * * * * * *
        # need to know surface temp
        # On first round, use temp + 2.4 for surface temp (not sure why ??)
        if one == 0:
            one = 1 # so next time step, use main growth loop
            ts = temp + 2.4

            # # ** add inital conditions **
            diameter_s.append(rad * 2) # [0]
            density_s.append(rhog) # [0]
            mass_s.append(mass) # [0]
            terminal_velocity_s.append(vt) # [0] 
            surface_temp_s.append(ts) # [0]
            horizontal_wind_s.append(0) # [0] ? idc


        # dm for diffusion only for Diameter < 300 um
        if dmm <= 0.30:
            if dmm <= 0.20:
                ts = temp
                delrd = 1.0e-6 * gr(tempc)
                ddsave = delrd
            else:
                # for particles between 200-300 um - smooth transition between
                if graup == 0:
                    dd200 = ddsave
                    graup = 1
                drtrans = dd200 + (dr300 - dd200) * (dmm - 0.2) / 0.1

        # 5. calculate vapour pressure * * * * * * * * * * * * * * * * * * * * * * * * * * 
        esi = 100. * vapour(TTR) * (math.exp((ts - TTR) * LS / (RW * ts * TTR)))
        rvs = esi / (RV * ts)
        es = 100. * vapour(temp)
        re = es / (RV * temp)
        # Sherwood number as per Mason (1971) - mass transfer 
        nsh = 0.58 * nre ** 0.5
        # Nusselt number as per Mason (1971) - heat transfer
        nnu = nsh
        # use lab values for vapour growth
        delrd = 1.0e-6 * gr(tempc)
        ## vapour growth of crystals 
        delmd = 2. * PI * rad * D * nsh * (re - rvs) * namelist['DELTIM'] if dmm > 0.2 else 0. # diffusional growth increment mass (kg) for particles > 200 um
        

        # 6. calculate rimming * * * * * * * * * * * * * * * * * * * * * * * * * * 
        # dm for riming - only for D > 300 um
        delmr = 0.
        if dmm > 0.30:
            # droplet spectrum -- Gaussian distribution matched to LWC
            stdv = DIS * dbar / 2.
            sumkap = 0.
            sumrh = 0.
            sumy = 0.
            if ts < 273.15:
                mom = mass * vt * 1.0e5
                beta = 0.738
                # the cloud is represented by a Gaussian distribution of droplet sizes, with mean dbar and std dev stdv
                # Start: loop through 24 channels of droplet spectrum + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +                
                for k in range(24):
                    # 6.1 Collection kernel calculation - for each channel * * * * * * * * * * * * * * * * * * * * * * * * * *
                    xi = (26. - (k + 1.)) * 1.0e-6
                    hxa = (xi * 1.0e6 - dbar / 2.) / stdv
                    hxa = hxa ** 2 / 2.
                    yi = NMAX * math.exp(-hxa) # concentration of droplets in channel k
                    sumy = yi + sumy
                    # collection kernel - formulae derived from B & G relationships
                    # ^ how effectively the graupel collects drops
                    if k == 23:
                        kappa = 1.09 * 1.0e-6 * mom ** beta
                    if k == 22:
                        kappa = 3.56 * 1.0e-6 * mom ** beta
                    if k == 21:
                        kappa = 5.41 * 1.0e-6 * mom ** beta
                    if k == 20:
                        kappa = 6.80 * 1.0e-6 * mom ** beta
                    if k == 19:
                        kappa = 7.75 * 1.0e-6 * mom ** beta
                    if k == 18:
                        kappa = 8.37 * 1.0e-6 * mom ** beta
                    if k == 17:
                        kappa = 8.80 * 1.0e-6 * mom ** beta
                    if k == 16:
                        kappa = 9.13 * 1.0e-6 * mom ** beta
                    if k == 15:
                        kappa = 9.38 * 1.0e-6 * mom ** beta
                    if k == 14:
                        kappa = 9.58 * 1.0e-6 * mom ** beta
                    if k == 13:
                        kappa = 9.75 * 1.0e-6 * mom ** beta
                    if k == 12:
                        kappa = 9.87 * 1.0e-6 * mom ** beta
                    if k == 11:
                        kappa = 9.97 * 1.0e-6 * mom ** beta
                    if k == 10:
                        kappa = 10.07 * 1.0e-6 * mom ** beta
                    if k == 9:
                        kappa = 10.17 * 1.0e-6 * mom ** beta
                    if k == 8:
                        kappa = 10.22 * 1.0e-6 * mom ** beta
                    if k == 7:
                        kappa = 10.30 * 1.0e-6 * mom ** beta
                    if k == 6:
                        kappa = 10.37 * 1.0e-6 * mom ** beta
                    if k <= 5:
                        kappa = 10.45 * 1.0e-6 * mom ** beta
                    sumkap = sumkap + kappa * yi # weight collection kernel by droplet concentration

                    # 6.2 Impact velocity of cloud drops calculation * * * * * * * * * * * * * * * * * * * * * * * * * *
                    # impact velocity of impinging cloud drops from R & H
                    ns = 2. * vt * xi * xi * 1000. / (9. * ETA * rad)
                    w = math.log10(ns)
                    w2 = w * w
                    w3 = w ** 3
                    w4 = w ** 4
                    if nre <= 20.:
                        if 0.4 <= ns <= 10.:
                            vimp = (0.1701 + 0.7246 * w + 0.2257 * w2 - 1.13 * w3 + 0.5756 * w4)
                        if ns < 0.4:
                            vimp = 0
                        if ns > 10.0:
                            vimp = 0.57
                    if 20. < nre <= 65.:
                        if 0.2 <= ns <= 10.:
                            vimp = (0.2927 + 0.5085 * w - 0.03453 * w2 - 0.2184 * w3 + 0.03595 * w4)
                        if ns < 0.2:
                            vimp = 0.0
                        if ns > 10.0:
                            vimp = 0.59
                    if 65. < nre <= 200.:
                        if 0.2 <= ns <= 10.0:
                            vimp = (0.3272 + 0.4907 * w - 0.09452 * w2 - 0.1906 * w3 + 0.07105 * w4)
                        if ns < 0.2:
                            vimp = 0.0
                        if ns > 10.0:
                            vimp = 0.61
                    if nre > 200.:
                        if 0.2 <= ns <= 10.0:
                            vimp = (0.356 + 0.4738 * w - 0.1233 * w2 - 0.1618 * w3 + 0.08087 * w4)
                        if ns < 0.2:
                            vimp = 0.0
                        if ns > 10.0:
                            vimp = 0.63
                    vimp = vimp * vt # this is the impact velocity 

                    # 6.3 Density of newly formed rime calculation * * * * * * * * * * * * * * * * * * * * * * * * * *
                    # density of the newly formed rime -- equation from H & P
                    tsc = ts - 273.15
                    arg = -dbar * vimp / (2. * tsc)
                    arg2 = arg ** 2
                    arg3 = arg ** 3
                    if tsc <= -5. or arg >= -1.6:
                        rhor = 0.30 * arg ** 0.44
                    else:
                        rhor = math.exp(-0.03115 - 1.7030 * arg + 0.9116 * arg2 - 0.1224 * arg3)
                    # rhor -> density of new rime
                    sumrh = rhor * yi + sumrh ## total rime (from contribution of all droplet channels) - accumulates through loop
                # End: loop through 24 channels of droplet spectrum + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

                # adjust to kg/m3
                rhor = sumrh * 1.0e3 / sumy # this is effective rime density - depend on total drop conc
                # collection kernel
                kapb = sumkap / sumy # final collection kernel (how much liquid water is collected) - depend on total drop conc
            else:
                #(ts >= 273.15 --> warm growth) 
                rhor = 900.
                kapb = PI * rad ** 2 * vt

            # 6.4 surface temperature calculation * * * * * * * * * * * * * * * * * * * * * * * * * *
            # surface temperature of graupel particle
            if tflag == 0:
                # in frozen growth regime - droplets freeze on impact (but depend on amount of rime)
                # only go to sub tsurf if kapb is significant
                if kapb >= 0.2e-10:
                    ts = tsurf(kapb, temp)
                else:
                    # if collection kernel is small, assume surface temp = ambient temp
                    ts = temp
                if ts >= 273.:
                    # if surface temp reach 0 -> into warm growth regime
                    tflag = 1
                    ts = 273.15

            # increase in mass due to riming
            delmr = kapb * lwc * namelist['DELTIM'] # mass of rime accreted (kg)

            # 6.5 fraction of unfrozen water shed * * * * * * * * * * * * * * * * * * * * * * * * * * 
            # If warm growth, assume unfrozen water is shed
            if tflag == 1:
                es = 100. * vapour(ts)
                rvs = es / (RV * ts)
                re = es / (RV * temp)
                frac1 = TK * (temp - ts)
                frac2 = LV * D * (rvs - re)
                frac = -PI * 2. * rad * nnu * (frac1 - frac2)
                frac = frac - delmr * CPW * (temp - ts)
                frac = frac / (delmr * LF)
                # increase frac artificially - 29 Sept. 88 ??
                frac = 1.
                delmr = frac * delmr # so no water shed ??
        # end of rimming loop (for D > 300 um) * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        # 7. calculate change in radius of graupel particle (from mass / diffusional growth) * * * * * * * * * * * * * * * * * * * * * * * * * *
        # delmr = riming mass increment // delmd = diffusional mass increment 

        # dR and Rnew
        if rad * 2.e3 <= 0.2:
            # small particles (<200 um) - only diffusional growth
            delrr = 0.
        if rad * 2.e3 <= 0.3 and rad * 2.e3 > 0.2:
            # for particles between 200-300 um - smooth transition between diffusional and riming growth
            delrr = drtrans
        if rad * 2.e3 > 0.3:
            # for particles > 300 um - only riming growth
            delrr = delmr / (4. * PI * rad * rad * rhor)
            """ this has been added in"""
            delrd = abs(delmd) / (4. * PI * rad * rad * RHOI)
            if delmd < 0.: # not sure i understand this - if delmd is negative, then delrd is negative? - so subtract from delr?
                delrd = -delrd

        delr = delrr + delrd
        rad = rad + delr

        # 8. changes in mass and bulk density * * * * * * * * * * * * * * * * * * * * * * * * * *
        # new bulk density of the graupel particle
        mass = mass + delmr
        rhog = mass * 3. / (4. * PI * rad ** 3)
        if rhog < 100.:
            rhog = 100.
        if rhog > 900.:
            rhog = 900.

        # 9. Hallett-Mossop time * * * * * * * * * * * * * * * * * * * * * * * * * *
        # particle in one of the Hallett-Mossop zones? Store x,z position, start splinter with a diameter of 20 um, give splinter a number, and calculate the concentration of splinters produced.
        # Note, only one calculation per zone; zhm is calculated to be at the centre of the zone. Assume temp is close to the temp at the top of the
        # sub-zone and calculate the mid-point using the current altitude and 6.7 deg -> 1 km. The value of i (which gives the time) is stored for each new splinter group.

        if namelist['hmflag'] == 1: #HM flag is switched on -> splinter production
            if rad * 2e3 >= 0.3 and tempc >= -9.:
            # if radius > 0.3 mm and temperature > -9ºC, then check for HM zones
                if -8. <= tempc < -7.:
                    # if temperature is between -8 and -7ºC, then in HM zone 1
                    hmrime += delmr # delmr = mass of new rime accumulated in this zone
                    hmzone1 = 1
                if hmzone1 == 1 and (tempc < -8. or tempc >= -7.):
                    nx = conc[j] * namelist['HMFACT'] * hmrime if splinter == 0 else nhm[hmj] * namelist['HMFACT'] * hmrime
                    # nx = number of splinters produced in this zone. 
                    # conc[j] = conc of orginal graupel particles - so if second gen of splinter - still based on inital graupel
                    # HMFACT = HM splinter production factor (0.2) 
                    # hmrime = mass of rime accumulated in this zone
                    hmi.append(i) # time step of splinter production - for plotting, got rid o
                    nhm.append(nx) # store splinter conc for this zone
                    xhm.append(x) # horizontal position of splinter
                    zhm.append(zkm - 0.5 / 6.7) # vertical position of splinter
                    stime.append(time) # time of splinter production

                    hm += 1 # increment splinter counter
                    hmrime = 0. # reset acccumulated rime mass for next zone
                    hmzone1 = 0 # reset HM zone 1 flag for next zone

                ### ^^ this is then repeated for the other HM zones (2-5) - same logic, just different temperature ranges
                ## particle moves through each zone seperately

                if -7. <= tempc < -6.:
                    # if temperature is between -7 and -6ºC, then in HM zone 2
                    hmrime += delmr
                    hmzone2 = 1
                if hmzone2 == 1 and (tempc < -7. or tempc >= -6.):
                    nx = conc[j] * namelist['HMFACT'] * hmrime if splinter == 0 else nhm[hmj] * namelist['HMFACT'] * hmrime
                    nhm.append(nx)
                    xhm.append(x)
                    zhm.append(zkm - 0.5 / 6.7)
                    stime.append(time)
                    hmi.append(i)
                    hm += 1
                    hmrime = 0.
                    hmzone2 = 0

                if -6. <= tempc < -5.:
                    # if temperature is between -6 and -5ºC, then in HM zone 3
                    hmrime += delmr
                    hmzone3 = 1
                if hmzone3 == 1 and (tempc < -6. or tempc >= -5.):
                    nx = conc[j] * namelist['HMFACT'] * hmrime if splinter == 0 else nhm[hmj] * namelist['HMFACT'] * hmrime
                    nhm.append(nx)
                    xhm.append(x)
                    zhm.append(zkm - 0.5 / 6.7)
                    stime.append(time)
                    hmi.append(i)
                    hm += 1
                    hmrime = 0.
                    hmzone3 = 0

                if -5. <= tempc < -4.:
                    # if temperature is between -5 and -4ºC, then in HM zone 4
                    hmrime += delmr
                    hmzone4 = 1
                if hmzone4 == 1 and (tempc < -5. or tempc >= -4.):
                    nx = conc[j] * namelist['HMFACT'] * hmrime if splinter == 0 else nhm[hmj] * namelist['HMFACT'] * hmrime
                    nhm.append(nx)
                    xhm.append(x)
                    zhm.append(zkm - 0.5 / 6.7)
                    stime.append(time)
                    hmi.append(i)
                    hm += 1
                    hmrime = 0.
                    hmzone4 = 0

                if -4. <= tempc < -3.:
                    # if temperature is between -4 and -3ºC, then in HM zone 5
                    hmrime += delmr
                    hmzone5 = 1
                if hmzone5 == 1 and (tempc < -4. or tempc >= -3.):
                    nx = conc[j] * namelist['HMFACT'] * hmrime if splinter == 0 else nhm[hmj] * namelist['HMFACT'] * hmrime
                    nhm.append(nx)
                    xhm.append(x)
                    zhm.append(zkm - 0.5 / 6.7)
                    stime.append(time)
                    hmi.append(i)
                    hm += 1
                    hmrime = 0.
                    hmzone5 = 0

        time_m = time + namelist['DELTIM'] / 60. # time in minutes

        # 10. move the particle * * * * * * * * * * * * * * * * * * * * * * * * * *
        # particle velocity: vt is positive down
        wpcle = wi - vt # wi = environmental air velocity (positive up) // vt = particle terminal velocity (positive down)

        # vertical and horizontal position
        z = zold + wpcle * namelist['DELTIM']
        x = xold + hw * namelist['DELTIM'] / 1000.
		if dmm <= 0.3:
            # diam is in mm
            conc = 10. ** 4. * math.exp(-39.47 * dmm) # calculating concentration of particles
        else:
            conc = 10. ** 0.7 * math.exp(-1.3 * dmm)

        # save this time step !
        diameter_s.append(rad * 2)
        density_s.append(rhog) 
        mass_s.append(mass)
        terminal_velocity_s.append(vt)
        surface_temp_s.append(ts)
        altitude_s.append(z)
        horizontal_position_s.append(x)
        horizontal_wind_s.append(hw)
        # if time > run_time, env temp is warmer than 0, or z < zbase (alt of cloud base), then break
        if time_m >= namelist['runtime'] or temp > 273.15 or z <= zbase:
            #num[j] = tt_ - 1
            break

        rhoa = pres / (RD * temp)
        zold = z
        xold = x

        # back for more growth -- loop on i
        nx = 0
        indx += 1

    # end of time-step growth loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    data_dict = {
        "diameter": diameter_s,
        "density": density_s,
        "mass": mass_s,
        "terminal_velocity": terminal_velocity_s,
        "surface_temperature": surface_temp_s,
        "altitude": altitude_s,
        "horizontal_position": horizontal_position_s,
        "horizontal_wind": horizontal_wind_s}
    return data_dict

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


def main():
    """
    executing graupel function and other stuff - again lots of steps
    outlined below:
        1. set up saving variables
        2. set up particle properties
            2a. initialise arrays and variables
            2b. create initial particle size distribution (spectrum)
            2c. define cloud properties: pressure / temperature / depth /vertical velocity profile
        3. set up environmental profile
        4. set up thermals
            4a. setting up first thermal
            4b. setting up second thermal
            4c. find environmental grid location of the thermals
            4d. simulate thermals through time
        5. growing all particles in graupel function
            5a. grow graupel particles in graupel()
            5b. grow HM splinter particles in graupel()
        6. making netcdf!
        
    ! grow and develop all thermals through time - this is recorded, then passed to graupel !
    """

    global zbase, cdepth, umax
    global hm, hmj, hmrime, splinter, j
    global thtop1, thbase1, thtop2, thbase2, cldtop, downbase
    global time, alt

    # 1. set up saving variables
    ## save tings babe! - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    os.makedirs(namelist['save_loc'], exist_ok=True) # if make folder, if already exist, do nothing
    # setting up time and height for model environment
    time = np.zeros(numt)
    alt[:] = np.arange(namelist['MAX2']) * 10. # height array
    
    # 2. set up particle properties
    # 2a. initialise arrays and variables - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # umax from continuity
    umax = namelist['UFACT'] * namelist['maxup'] / 2. # calculate horizontal wind

    # 2b. create initial particle size distribution (spectrum) - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #totnum = 0
    for jj in range(MAX1):
        #radi[jj] = diam[jj] / 2. 
        # two exponential fits - spectrum (10/8/94)
        if diam[jj] <= 0.3:
			# diam is in mm
            conc[jj] = 10. ** 4. * math.exp(-39.47 * diam[jj]) # calculating concentration of particles
        else:
            conc[jj] = 10. ** 0.7 * math.exp(-1.3 * diam[jj])
        #totnum += conc[jj] # total intitial particle concentration (m-3)
    

    # 2c. define cloud properties: pressure / temperature / depth /vertical velocity profile - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    zbase = ptz(namelist['pbase']) # height of cloud base (m)
    #print(f"cloud base: p = {namelist['pbase']:5.1f} mb, T = {namelist['tbase']:4.1f} C, "f"z = {zbase / 1000.:3.1f} km", file=sys.stderr)
    cdepth = namelist['CTOP'] - zbase / 1000. # depth of cloud (km)

    # vertical velocity change with alt -> slope and intercept of (linear) vertical velocity profile
    slope = (namelist['maxup'] - namelist['wbase']) / (namelist['zmax'] - zbase / 1000.)
    intcpt = namelist['wbase'] - slope * zbase / 1000.

    # 3. set up environmental profile
    # build environment profile - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for ii in range(namelist['MAX2']):
        # whole column properties
        alt[ii] = 10. * ii # each level seperated by 10 m
        ps[ii] = ztp(alt[ii]) # pressure
        at_, alwc_ = trev(namelist['pbase'], namelist['tbase'], ps[ii]) # temp + LWC
        lw[ii] = alwc_ * 1.0e-3 # lwc through column
        tr[ii] = at_ + 273.15 # temp through column
        db[ii] = namelist['DBAR'] + ii / 125. # mean cloud drop diameter (not sure why / 125.)

        # environmental updraft
        # vv = vertical velocity defined at each height
        if alt[ii] < zbase:
            # below cloud
            vv[ii] = 1.
        elif alt[ii] / 1000. <= namelist['zmax']:
            # in cloud + alt < max updraft alt
            # increasing updraft to zmax
            vv[ii] = slope * alt[ii] / 1000. + intcpt
        else:
            # in cloud + alt >= zmax
            # decreasing updraft above this
            vv[ii] = (namelist['maxup'] / ((namelist['CTOP'] + 2.) - namelist['zmax'])) * ((namelist['CTOP'] + 2.) - alt[ii] / 1000.0)

    # 4. set up thermals
    # 4a. setting up first thermal - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # initial positions of thermals
    thtop1[0] = namelist['zinit'] / 1000. # top of thermal initally at the particles initial height

    if thtop1[0] > namelist['CTOP']:
        thtop1[0] = namelist['CTOP'] # top of thermal cannot exceed cloud top
    cldtop[0] = thtop1[0] # cloud top = thermal top
    thbase1[0] = thtop1[0] - namelist['TDEPTH'] # base of thermal
    
    if thbase1[0] < 0.:
        thbase1[0] = 0. # base of thermal cannot be below ground

    # 4b. setting up second thermal - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    thtop2[0] = thbase1[0] - namelist['TGAP'] # set second thermal below first thermal by TGAP
    if thtop2[0] < 0.:
        thtop2[0] = np.nan # top of second thermal cannot be below ground
    thbase2[0] = thtop2[0] - namelist['TDEPTH']
    if thbase2[0] < 0.:
        thbase2[0] = np.nan # base of second thermal cannot be below ground
    downbase[0] = np.nan  # bottom of cloud top debris

    # 4c. find environmental grid location of the thermals - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    post1 = posb1 = post2 = posb2 = 0 
    post1 = find_grid_index(thtop1[0], alt)
    posb1 = find_grid_index(thbase1[0], alt)
    post2 = find_grid_index(thtop2[0], alt)
    posb2 = find_grid_index(thbase2[0], alt)

    wtht1 = boundary_velocity(post1, vv)
    wthb1 = boundary_velocity(posb1, vv)
    wtht2 = boundary_velocity(post2, vv)
    wthb2 = boundary_velocity(posb2, vv)


    # 4d. simulate thermals through time - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # positions of thermals (in km) with time
    topflag = tht2flag = thb2flag = 0

    # loop through each time step
    for i in range(1, numt_0):
        # move thermal 1 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # deal with thermal 1 top
        thtop1[i] = thtop1[i - 1] + (wtht1 * namelist['DELTIM']) / 1000. # new height = old height + velocity × time
        if thtop1[i] >= namelist['CTOP']:
            thtop1[i] = namelist['CTOP']

        # deal with thermal 1 base
        thbase1[i] = thtop1[i] - namelist['TDEPTH']
        if thbase1[i] > 0.:
            # base of thermal 1 is above ground, so it does exist
            thbase1[i] = thbase1[i - 1] + (wthb1 * namelist['DELTIM']) / 1000.
        if thbase1[i] < 0.:
            thbase1[i] = 0. # base of thermal cannot be below ground

        # create / move thermal 2 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if tht2flag != 1:
            # if thermal top 2 has not been initilised 
            thtop2[i] = thbase1[i] - namelist['TGAP']
            if thtop2[i] > 0.: # if thermal top 2 is above ground, then initialise it
                tht2flag = 1
            if thtop2[i] < 0.: # if thermal top 2 is below ground, np.nan it
                thtop2[i] = np.nan
        else:
            # if thermal top 2 has been initialised, then move it
            thtop2[i] = thtop2[i - 1] + (wtht2 * namelist['DELTIM']) / 1000.

        if thb2flag != 1:
            # if thermal base 2 has not been initilised
            thbase2[i] = thtop2[i] - namelist['TDEPTH'] # initalise thermal 2 base
            if thbase2[i] > 0.:
                thb2flag = 1
            if thbase2[i] < 0.: # if thermal base 2 is below ground, np.nan it
                thbase2[i] = np.nan
        else:
            # thermal 2 base above ground, so it does exist
            thbase2[i] = thbase2[i - 1] + (wthb2 * namelist['DELTIM']) / 1000.

        # deal with cloud top region * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if thtop1[i] >= namelist['CTOP'] and thbase1[i] >= (namelist['CTOP'] - namelist['TDEPTH'] / 4.) and topflag == 0:
            # thermal is approaching maximum cloud top -> stopping growth 
            topflag = 1 # switch on processes
            thtop1[i] = thtop2[i]
            thbase1[i] = thbase2[i]
            thtop2[i] = thbase1[i] - namelist['TGAP']
            thbase2[i] = thtop2[i] - namelist['TDEPTH']
            tht2flag = thb2flag = 0
            cldtop[i - 1] = namelist['CTOP'] # new?

        if topflag == 1:
            # thermal has reached cloud top -> thermal stopped growing, now become region of descending material 
            cldtop[i] = cldtop[i - 1] + (namelist['WCTDEB'] * namelist['DELTIM']) / 1000. # cloud top region grows
            #if cldtop[i] <= namelist['GENCTOP']:
            #    cldtop[i] = namelist['GENCTOP']
            
            downbase[i] = cldtop[i] - namelist['TDEPTH'] / 4. # base of cloud top region 
            if downbase[i] <= thtop1[i]:
                downbase[i] = thtop1[i]
            if cldtop[i] < thtop1[i]:
                topflag = 0
        else:
            # thermal is not near max cloud top -> still growing
            cldtop[i] = thtop1[i]
            downbase[i] = np.nan

        # again find environmental grid location of the thermals * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        post1 = posb1 = post2 = posb2 = 0
        post1 = find_grid_index(thtop1[i], alt)
        posb1 = find_grid_index(thbase1[i], alt)
        post2 = find_grid_index(thtop2[i], alt)
        posb2 = find_grid_index(thbase2[i], alt)

        wtht1 = boundary_velocity(post1, vv)
        wthb1 = boundary_velocity(posb1, vv)
        wtht2 = boundary_velocity(post2, vv)
        wthb2 = boundary_velocity(posb2, vv)
    # end of simulating thermals through time - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # 5. growing all particles in graupel function - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # using pre-defined evolution of thermals
    # Note particles produced in the H-M zones in graupel() are splinters.
    # Their growth is dealt with after the initial set of trajectories, by number.

    # 5a. grow graupel particles in graupel()
    #wi = 0. # updraft m/s
    hm = 0
    hmj = 0
    hmrime = 0.
    splinter = 0 # first simulating graupel particle growth
    particle_data = {
        "diameter": [],
        "density": [],
        "mass": [],
        "terminal_velocity": [],
        "surface_temperature": [],
        "altitude": [],
        "horizontal_position": [],
        "horizontal_wind": []}

    for jj in range(MAX1):
        j = jj # j = current particle index
        particle_grow = graupel() # makes a dict of saved values
        # then save each particle iteration
        for name in particle_data:
            particle_data[name].append(particle_grow[name])  

    # some of particles finish early - need to make sure all same size
    particle_data_t = {}

    for name, trajectories in particle_data.items():
        max_len = max(len(x) for x in trajectories)
        for x in trajectories:
            x.extend([np.nan] * (max_len - len(x)))

        # this is combining across all time and particles
        particle_data_t[name] = np.stack(trajectories)

    if hm > 0:
        # 5b. grow HM splinter particles in graupel()
        splinter_data = {
            "diameter": [],
            "density": [],
            "mass": [],
            "surface_temperature": [],
            "terminal_velocity": [],
            "altitude": [],
            "horizontal_position": [],
            "horizontal_wind": []}
        
        splinter = 1
        hmj = 0
        while hmj < hm - 1:
            # for every HM splinter group created -> run graupel to simulate growth and movement
            particle_grow = graupel()
            # save the results for this splinter
            for name in splinter_data:
                splinter_data[name].append(particle_grow[name])
            hmj += 1 # loop through index og 

        splinter_data_t = {}
        for name, trajectories in splinter_data.items():
            max_len = max(len(x) for x in trajectories)
            for x in trajectories:
                x.extend([np.nan] * (max_len - len(x)))

            # this is combining across all time and splinters
            splinter_data_t[name] = np.stack(trajectories)

    ## 6. making netcdf! + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
    fname = os.path.join(namelist['save_loc'], f"{save_name}.nc")
    nc = Dataset(fname, "w", format="NETCDF4")
    # dimensions - - - - - -
    nc.createDimension("time", numt_0)
    nc.createDimension("height", namelist['MAX2'])
    nc.createDimension("particle", MAX1) ## we are having particles as their own dimension
    if hm > 0:
        nc.createDimension("splinter_particle", splinter_data_t["diameter"].shape[0]) ## splinter dimension - number of splinters varies depending on set up

    # core variables - - - - - -
    make_nc_variable(nc, np.arange(numt_0) * namelist['DELTIM'], "time", ("time",), f"seconds since simulation start (dt={namelist['DELTIM']}s)", "simulation time", fill_value_f = None)
    make_nc_variable(nc, alt, "height", ("height",), "m", "height above ground", fill_value_f = None)
    make_nc_variable(nc, diam, "particle", ("particle",), "dimensionless", "particle index", data_type="i4", fill_value_f = None)
    if hm > 0:
        make_nc_variable(nc, np.arange(splinter_data_t["diameter"].shape[0]), "splinter_particle", ("splinter_particle",), "dimensionless", "splinter particle index", data_type="i4", fill_value_f = None)

    ##### deal with environment variables #############
    ## envi variables varying with height:
    env_nc = nc.createGroup("environment")
    make_nc_variable(env_nc, ps, "pressure", ("height",), "hPa", "environmental pressure")
    make_nc_variable(env_nc, tr, "temperature", ("height",), "K", "environmental temperature")
    make_nc_variable(env_nc, lw, "lwc", ("height",), "kg m-3", "environmental liquid water content")
    make_nc_variable(env_nc, db, "cloud_drop_diameter", ("height",), "um", "mean cloud drop diameter")
    make_nc_variable(env_nc, vv, "vertical_velocity", ("height",), "m s-1", "environmental vertical velocity")
    ## envi variables varying with time:
    make_nc_variable(env_nc, np.where(thtop1[:numt_0] == np.nan, np.nan, thtop1[:numt_0]), "thermal_top_1", ("time",), "km", "top altitude of thermal 1")
    make_nc_variable(env_nc, np.where(thbase1[:numt_0] == np.nan, np.nan, thbase1[:numt_0]), "thermal_base_1", ("time",), "km", "base altitude of thermal 1")
    make_nc_variable(env_nc, np.where(thtop2[:numt_0] == np.nan, np.nan, thtop2[:numt_0]), "thermal_top_2", ("time",), "km", "top altitude of thermal 2")
    make_nc_variable(env_nc, np.where(thbase2[:numt_0] == np.nan, np.nan, thbase2[:numt_0]), "thermal_base_2", ("time",), "km", "base altitude of thermal 2")
    make_nc_variable(env_nc, np.where(cldtop[:numt_0] == np.nan, np.nan, cldtop[:numt_0]), "cloud_top", ("time",), "km", "cloud top altitude")
    make_nc_variable(env_nc, np.where(downbase[:numt_0] == np.nan, np.nan, downbase[:numt_0]), "debris_base", ("time",), "km", "base altitude of descending cloud-top debris")

    ##### deal with ice particle variables #############
    ice_nc = nc.createGroup("ice_particles")
    make_nc_variable(ice_nc, particle_data["diameter"], "diameter", ("particle", "time"), "m", "graupel particle diameter")
    make_nc_variable(ice_nc, particle_data["density"], "density", ("particle", "time"), "kg m-3", "graupel particle bulk density")
    make_nc_variable(ice_nc, particle_data["mass"], "mass", ("particle", "time"), "kg", "graupel particle mass")
    make_nc_variable(ice_nc, particle_data["surface_temperature"], "surface_temperature", ("particle", "time"), "K", "graupel particle surface temperature")
    make_nc_variable(ice_nc, particle_data["terminal_velocity"], "terminal_velocity", ("particle", "time"), "m s-1", "graupel particle terminal velocity")
    make_nc_variable(ice_nc, particle_data["altitude"], "altitude", ("particle", "time"), "m", "graupel particle altitude")
    make_nc_variable(ice_nc, particle_data["horizontal_position"], "horizontal_position", ("particle", "time"), "km", "graupel particle horizontal position")
    make_nc_variable(ice_nc, particle_data["horizontal_wind"], "horizontal_wind", ("particle", "time"), "m s-1", "graupel particle horizontal wind")

    # only make splinter group if there are splinters
    if hm > 0:
        ##### deal with ice splinter variables #############
        splinter_nc = nc.createGroup("splinter_particles")
        make_nc_variable(splinter_nc, splinter_data["diameter"], "diameter", ("splinter_particle", "time"), "m", "graupel splinter diameter")
        make_nc_variable(splinter_nc, splinter_data["density"], "density", ("splinter_particle", "time"), "kg m-3", "graupel splinter bulk density")
        make_nc_variable(splinter_nc, splinter_data["mass"], "mass", ("splinter_particle", "time"), "kg", "graupel splinter mass")
        make_nc_variable(splinter_nc, splinter_data["surface_temperature"], "surface_temperature", ("splinter_particle", "time"), "K", "graupel splinter surface temperature")
        make_nc_variable(splinter_nc, splinter_data["terminal_velocity"], "terminal_velocity", ("splinter_particle", "time"), "m s-1", "graupel splinter terminal velocity")
        make_nc_variable(splinter_nc, splinter_data["altitude"], "altitude", ("splinter_particle", "time"), "m", "graupel splinter altitude")
        make_nc_variable(splinter_nc, splinter_data["horizontal_position"], "horizontal_position", ("splinter_particle", "time"), "km", "graupel splinter horizontal position")
        make_nc_variable(splinter_nc, splinter_data["horizontal_wind"], "horizontal_wind", ("splinter_particle", "time"), "m s-1", "graupel splinter horizontal wind")

    # add metadata about what version of py file used + what namelist used
    nc.pyfile_name = sys.argv[0]
    nc.namelist_file = sys.argv[1]
    nc.close()
    ## + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 


if __name__ == "__main__":
    main()

print('ran')