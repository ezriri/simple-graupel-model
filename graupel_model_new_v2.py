## very very different from v1 -> include secondary ice - whole of graupel() is changed

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
# #define constants
# ---------------------------------------------------------------------------
diam = np.array([.05, .1, .15, .2, .3, .4,
                  .5, .6, .7, .8, .9, 1., 1.1, 1.2,
                  1.4, 1.6, 1.8, 2.0, 2.2, 2.4]) ## i assume initial particle diameters (mm) ??


MAX1 = len(diam)   # number of particle size categories
MAX2 = 1500        # number of altitude values 
MAX3 = 9000        # number of time steps
MAX4 = 24          # number of grid points in vertical
MAX5 = 2000        # number of H-M splinter groups
MAX6 = 2010        # number of original size categories + H-M splinter groups
MAX8 = 500         # number of points for plotting

TBASE = 10.6        # temperature at cloud base
PBASE = 705.        # pressure at cloud base
DBAR = 20.          # mean cloud droplet diameter
RATTH = 0.7         # L/La in thermal
RATCT = 0.3         # L/La in uppermost cloud top region
RATDEB = 0.1        # L/La in debris
RATDOWN = 0.1       # L/La in downdraught
WCTDEB = -2.        # w in descending cloud top region
UFACT = 1.0         # mult factor for umax
DDEPTH = 1.0        # depth that downdraught descends
CTOP = 11.0         # altitude of maximum cloud top
TDEPTH = 3.0        # depth of thermal
TGAP = 2.0          # vertical distance between thermals
TWID = TDEPTH       # width of thermal incl downdraught
DOWN = 0.4          # width of downdraught
UMAX = 2. # max updraft (m/s) ??
ZCT = TDEPTH / 20.  # depth of overall cloud top region
ZCT1 = ZCT / 2.     # depth of cloud top region 1
ZCT2 = ZCT / 2.     # depth of cloud top region 2
GENCTOP = CTOP - 0.5  # height of general cloud top

HMRINIT = 10.0e-6   # initial radius of HM splinter
HMFACT = 5.0e7      # number per kg of rime

PFACT = 5           # reduction in pts for plotting

P0 = 1013. # surface (sea level) pressure (mb)
T0 = 288. # surface temperature (K)
GAMMA = 0.0065 # constant lapse rate (ºC/m)
RHOI = 920 # density of ice (kg/m^3)
EPS = 0.622 # constant for water vapour calculation
PI = 3.14159 # constant pi 
DELTIM = 5. # time step interval (s)
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

# see note 3 above -- best-effort guess at the value of BAD
BAD = np.nan

# declared in the C source but not referenced anywhere in main() -- kept
# here for fidelity, unused
diamf = np.array([.05, .15, .25, .35, .5, .75, 1.1, 1.55, 2.1, 2.75,
                   3.45, 4.15, 4.85, 5.55, 6.25])
wid_arr = np.array([.1, .1, .1, .1, .2, .3, .4, .5, .6, .7,
                     .7, .7, .7, .7, .7])

# ---------------------------------------------------------------------------
# Module-level globals (mirrors C file/global-scope variables shared between
# main(), graupel(), drag(), tsurf())
# ---------------------------------------------------------------------------
# the whole environment is initialised with empty arrays defined by MAX values
lw = np.zeros(MAX2)                              # LWC (at all heights)
db = np.zeros(MAX2)                              # dbar ??
vv = np.zeros(MAX2)                              # vertical velocity
alt = np.zeros(MAX2)                             # altitude (m?)
ps = np.zeros(MAX2)                              # pressure (mb?)
tr = np.zeros(MAX2)                              # temperature (Kelvin)
wid = 0.0                                        # width of channels
conc = np.zeros(MAX1)                            # concentration of particles in each size category
radi = np.zeros(MAX1)
spec = np.zeros(MAX1)
speci = np.zeros(MAX1)
tothm = np.zeros(MAX3)
cumtot = np.zeros(MAX3)

diaml = np.zeros(MAX1)                           # diameter spectrum at requested level
concl = np.zeros(MAX1)                           # concentration spectrum at requested level
radi = np.zeros(MAX1)                            # inital radius
diamh = np.zeros((MAX4, MAX1, MAX3))             # diameter for particular height
diamx = np.zeros(MAX1)                           # diameter in window coords ??
diamp = np.zeros((MAX6, MAX8))                   # diameter for straight plotting
vtp = np.zeros((MAX6, MAX8))                     # terminal velocity for plotting
deltsp = np.zeros((MAX6, MAX8))
effp = np.zeros((MAX6, MAX8))
timep = np.zeros((MAX6, MAX8))                   # time stored for plotting
rhop = np.zeros((MAX6, MAX8))                    # density of particle
zp = np.zeros((MAX6, MAX8))                      # altitude of each size
zptop = np.zeros((MAX1, MAX3))                   # rel to ground and cloud top
xp = np.zeros((MAX6, MAX8))                      # pos of particle relative to centre of thermal
stime = np.zeros(MAX5)
xhm = np.zeros(MAX5)
zhm = np.zeros(MAX5)
nhm = np.zeros(MAX5)

''' multiple thermals defined '''
thtop1 = np.zeros(MAX3)                          # top alt of thermal 1
thbase1 = np.zeros(MAX3)                         # bottom alt of thermal 1
thtop2 = np.zeros(MAX3)                          # top alt of thermal 2
thbase2 = np.zeros(MAX3)                         # bottom alt of thermal 2
cldtop = np.zeros(MAX3)                          # top of cloud top debis
downbase = np.zeros(MAX3)                        # bottom of cloud top debris

wtht1 = wthb1 = wtht2 = wthb2 = 0.0              # velocity of top and bottom of thermals
rad = 0.0                                        # current radius of graupel particles
wi = 0.0                                         # updraft (environmental air velocity) (m/s)
lwc = 0.0                                        # LWC - but selecting from lw at a specific height
rhoa = 0.0                                       # density of air (kg/m^3?)
reft = np.zeros(MAX3)                            # reflectivity with time
refg = np.zeros((MAX3, MAX4))                    # reflectivity with time and height
zinit = 0.0                                      # starting altitude of ice particles (m)
cf = 0.0                                         # # used to calculate reflectivity
sumd = np.zeros(MAX3)                            # used to calculate reflectivity
sumr = np.zeros((MAX4, MAX3))                    # used to calculate reflectivity
pbase = tbase = 0.0                              # pressure (mb) and temp (c) at cloud base
zbase = cdepth = 0.0                             # alt of cloud base and cloud depth
time = 0.0                                       # time in minutes
time_ = 0.0
at = alwc = 0.0                                  # adiabatic temp and adiabatic LWC
zgrid = np.zeros(MAX4)                           # vertical grid
xx = np.zeros(MAX3)                              # d.v. for plotting (dummy variable)
yy = np.zeros(MAX3)                              # d.v. for plotting
c1 = 0.0                                         # offset for framing spectra
spec = np.zeros(MAX1)                            # d.v. conc
specy = np.zeros(MAX1)                           # conc only for valid diameters
xinit = yinit = 0.0                              # initilisation pts for windows
x1 = x2 = wy1 = wy2 = 0.0                        # window coordinates (for plotting??)
maxup = 0.0                                      # max updraft
wbase = zmax = 0.0                               # updraft at cloud base and max altitude of updraft
slope = intcpt = 0.0                             # variables to calculate vertical velocity profile
runtime = 0.0                                    # length of run in (mins)
es = esi = 0.0                                   # water and ice vapour pressure
nre = nsh = nnu = 0.0                            # Reynolds, Sherwood and Nusslet no
xmax = 0.0                                       # maximum value on x-axis
level = 0.0                                      # level used for spectrum
totnum = 0.0
level1 = level2 = level3 = 0.0
hmrime = 0.0
zm8 = zm3 = 0.0

numt = 0                                         # number of time points for plotting
num = np.zeros(MAX6, dtype=int)                  # number of time points for plotting
hmi = np.zeros(MAX5, dtype=int)
hm = 0 # number / index of H-M splinter groups
hmj = 0 # index of H-M splinter group ?? 
j = 0                                            # global channel counter
tt = pp = 0
nxtick = 0                                       # number of ticks on x-axis
topflag = tht2flag = thb2flag = 0                # flag set if base of thermal at top

post1 = posb1 = post2 = posb2 = 0                # index for thermal velocities
splinter = 0
atlevel1 = atlevel2 = atlevel3 = 0
totpts = 0
hmflag = 0

pname = ""
lab1 = lab2 = lab3 = lab4 = lab5 = lab6 = lab7 = lab8 = lab9 = lab10 = ""
lab11 = lab12 = lab13 = lab14 = lab15 = lab16 = lab17 = lab18 = lab19 = ""
lab20 = lab21 = ""

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

def scale_axis(values, step=1.0):
    """
    Best-effort equivalent of the (unsupplied) library routine `scale()`.
    Picks axis min/max, rounded outward to the nearest `step`, from a set
    of (possibly BAD-flagged) data values. Used only for choosing y-axis
    bounds on the "f vs time" plot, where the caller then does
    `nticks = maxy - miny + 1`, i.e. it expects integer-ish bounds.
    """
    valid = values[values < BAD / 10.]
    # filter our bad data
    if valid.size == 0:
        return 0.0, 1.0
    miny = math.floor(valid.min() / step) * step
    maxy = math.ceil(valid.max() / step) * step
    if maxy <= miny:
        maxy = miny + step
    return maxy, miny
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def drag(r, rhog, rhoa_):
    """
    Calculate suitable minimum and maximum bounds for an axis.

    Args:
        values (np.ndarray): Array of data values used to determine the axis
            limits. Values greater than or equal to `BAD / 10` are treated as
            invalid and ignored.
        step (float): Step size to which the axis limits are rounded.
            Defaults to 1.0.

    Returns:
        tuple[float, float]: The maximum and minimum axis bounds, respectively.
            If no valid values are provided, returns (0.0, 1.0).

    Note:
        The minimum bound is rounded down to the nearest multiple of `step`,
        while the maximum bound is rounded up. If the resulting bounds are
        equal or reversed, the maximum is increased by `step`.
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

def _mask_bad(arr):
    """
        Replace BAD-sentinel values with NaN for plotting.

    Args:
        arr (array-like): Input array containing numerical values. Values
            greater than or equal to 99999 are treated as BAD sentinels.

    Returns:
        numpy.ndarray: Float array with BAD-sentinel values replaced by NaN.
    """
    arr = np.asarray(arr, dtype=float)
    return np.where((arr >= 99999.) | np.isnan(arr), np.nan, arr)
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
    1. loop through each time step
        2. loop through cloud drop size spectrum (24 channels)

    I have broken it up into 11 sections - seperating out specific processes     
    """
    global j, hm, hmrime, rad, wi, lwc, rhoa, es, esi, nre, nsh, nnu
    global num # store number of time points for plotting
    global vtsave
    global ddsave 
    global drtrans
    global delmd 
    global delmr
    global delrd

    # initialise local variables (mainly particle / environment)
    z = x = pres = temp = tempc = dbar = 0.0 # z = vertical position of particle (m)// x = horizontal position of particle (km?) //  pressure (mb) // temp = temperature (K) // tempc = temperature (C) // dbar = mean cloud droplet diameter
    zold = xold = 0.0 # previous position of particle (m, km)
    zkm = 0.0 # vertical position of particle (km)
    zdown = 0.0 # altitude in which particle enters downdraught (km)
    hw = 0.0 # m/s horizontal wind
    lwc7 = 0.0 # lwc at 7km

    tflag = 0 # tflag == 0 -> frozen growth // tflag == 1 -> warm growth
    one = 0 # initialisation of particle (0 = first time step, 1 = subsequent time steps)
    pp_ = 0
    tt_ = 0
    wi = 0.2
    indx = 0
    downflag = 0 # is particle in downdraught? (0 = no - normal / thermal envi, 1 = yes)
    topflag_local = 0
    graup = 0 # dictate graupel growth occuring (0 = no, 1 = yes) - changes in transition period
    zflag = 0 # control LWC depletion above 7km
    hmrime = 0. # rime mass accumulated while moving through HM zone
    hmzone1 = hmzone2 = hmzone3 = hmzone4 = hmzone5 = 0 # different HM zones for splinter growth

    # physical quantities 
    #vtsave = vt200 = vt300 = None
    #ddsave = dd200 = None
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

    # calculate vt and delmr for a 300 um diameter particle
    # Assume density of 220 g/cm^3 just now as a quick fix - 7/7/94
    # and use simple formula for kapb
    #cd = drag(0.15e-3, 220., rhoa) # drag coefficient for 300um (diameter) particle (function uses radius)
    #vt300 = 8. * 0.15 * 1.0e-3 * 220. * GRAV / (3. * rhoa * cd) # terminal velocity for 300um particle
    #dmr300 = PI * 0.15e-3 ** 2 * vt300 * lwc * DELTIM # riming mass change for 300um particle
    #dr300 = dmr300 / (4. * PI * 0.15e-3 ** 2 * 220.) # radius change for 300um particle
    # these values ^^ are used for smooth transition for riming between 200-300 um diameter particles

    # are we growing a splinter particle or a graupel particle?
    # splinter == 0 -> graupel particle // splinter == 1 -> splinter particle
    if splinter == 1:
        j = MAX1 + hmj
        istart = hmi[hmj] # doesnt start at timestep 0 - dynamically create new particles during the simulation
    else:
        istart = 0
    pos = 0

    # time-step growth loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # this is the main part, go through each time step and calculate growth of particle
    
    for i in range(istart, numt):
        ### 1. starting altitude of  particle (in environment) - - - - - - - - - - - - - - 
        if one == 0:
            # first round - get initial conditions
            if splinter > 0: # if particle is a splinter particle, get initial conditions from the graupel particle that created it
                z = 1000 * zhm[hmj]
                x = xhm[hmj]
                time = stime[hmj]
            else: # graupel particle
                z = zinit
                x = 0.
                time = 0.
            zold = z
            xold = x

        # loop for graupel environment values -- get pointer to initial height
        for ii in range(MAX2):
            if ii > 0:
                if z < alt[ii] and z >= alt[ii - 1]: # what alt level is particle at?
                    pos = ii
        pres = ps[pos] * 100.
        zkm = z / 1000.

        # 2. where is the particle with respect to the thermal? * * * * * * * * * * * * * * * * * * * * * * * * * *
        # this modifies the environmental conditions -> alter particle growth and movement
        
        hw = 0 # horizontal wind (m/s) # not persistent between time steps
        if downflag == 0: # particle is not in downdraft
            if ((zkm <= thtop1[i] - ZCT and zkm >= thbase1[i]) or (zkm <= thtop2[i] - ZCT and zkm >= thbase2[i])):
                # particle is still in thermal (can be thermal 1 or thermal 2)
                temp = tr[pos] - 1. # thermal envi is 1ºC cooler than the environment
                lwc = RATTH * lw[pos]
                dbar = db[pos] # drop size
                wi = vv[pos] # vertical velocity - based on the vertical velocity profile of the thermal

            # in cloud top region (top of thermal 1)
            if zkm <= thtop1[i] and zkm >= thtop1[i] - ZCT:
                temp = tr[pos] - 1.
                lwc = RATCT * lw[pos]
                dbar = db[pos]
                hw = umax
                # work out smooth vertical velocity in cloud top regions
                wi = vv[pos] - vv[pos] * (zkm - (thtop1[i] - ZCT)) / (2. * ZCT) # vertical velocity change with height

            # out of thermal -> in debris under thermal // above thermal (if at start)
            if ((zkm > thtop1[i] or zkm < thbase1[i]) and (zkm > thtop2[i] or zkm < thbase2[i]) and (zkm < downbase[i])) or (zkm > thtop1[i] and zkm > thtop2[i]):
                temp = tr[pos] - 1.
                lwc = RATDEB * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = 0.0

            # above thermal 1, within cloud top, in descending thermal remnants
            if zkm > thtop1[i] and zkm <= cldtop[i] and zkm >= downbase[i]:
                temp = tr[pos] - 2.
                lwc = RATDEB * lw[pos]
                dbar = db[pos]
                hw = 0.
                if cldtop[i] > GENCTOP:
                    # cloud top above general cloud top -> prescribed verical velocity
                    wi = WCTDEB
                else:
                    wi = 0.

            # horizontal position in km
            if x >= TWID:
                # particle has left the thermal, into downdraught
                temp = tr[pos] - 2.
                lwc = RATDOWN * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = -5.
                downflag = 1 # in downdraft
                zdown = zkm

        else:
            # particle is in downdraft
            temp = tr[pos] - 2.
            lwc = RATDOWN * lw[pos]
            dbar = db[pos]
            hw = 0.
            wi = -5.
            # make downdraughts no deeper than DDEPTH km
            if abs(zdown - zkm) >= DDEPTH:
                # reset particle to normal environment
                downflag = 0
                xold = 0
                x = 0.
                zdown = 0.

        # overshoot cloud top?
        if zkm >= CTOP or zkm > cldtop[i]:
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
                rad = HMRINIT
            rhog = 900. # initial density (kg m^-3)
            # initial mass
            mass = 4. * PI * rad ** 3 * rhog / 3.

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

        # 3. calculate drag for particle * * * * * * * * * * * * * * * * * *
        cd = drag(rad, rhog, rhoa)

        # 2. calculate terminal velocity * * * * * * * * * * * * * * * * * *
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
            one = 1
            ts = temp + 2.4

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
                            vimp = (0.1701 + 0.7246 * w + 0.2257 * w2 - 1.13 * w3
                                    + 0.5756 * w4)
                        if ns < 0.4:
                            vimp = 0
                        if ns > 10.0:
                            vimp = 0.57
                    if 20. < nre <= 65.:
                        if 0.2 <= ns <= 10.:
                            vimp = (0.2927 + 0.5085 * w - 0.03453 * w2 - 0.2184 * w3
                                    + 0.03595 * w4)
                        if ns < 0.2:
                            vimp = 0.0
                        if ns > 10.0:
                            vimp = 0.59
                    if 65. < nre <= 200.:
                        if 0.2 <= ns <= 10.0:
                            vimp = (0.3272 + 0.4907 * w - 0.09452 * w2 - 0.1906 * w3
                                    + 0.07105 * w4)
                        if ns < 0.2:
                            vimp = 0.0
                        if ns > 10.0:
                            vimp = 0.61
                    if nre > 200.:
                        if 0.2 <= ns <= 10.0:
                            vimp = (0.356 + 0.4738 * w - 0.1233 * w2 - 0.1618 * w3
                                    + 0.08087 * w4)
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
                        rhor = math.exp(-0.03115 - 1.7030 * arg + 0.9116 * arg2
                                         - 0.1224 * arg3)
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
            delmr = kapb * lwc * DELTIM # mass of rime accreted (kg)

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
        # particle in one of the Hallett-Mossop zones? Store x,z position,
        # start splinter with a diameter of 20 um, give splinter a number,
        # and calculate the concentration of splinters produced. Note, only
        # one calculation per zone; zhm is calculated to be at the centre of
        # the zone. Assume temp is close to the temp at the top of the
        # sub-zone and calculate the mid-point using the current altitude
        # and 6.7 deg -> 1 km. The value of i (which gives the time) is
        # stored for each new splinter group.

        if hmflag == 1: #HM flag is switched on 
            if rad * 2e3 >= 0.3 and tempc >= -9.:
            # if radius > 0.3 mm and temperature > -9ºC, then check for HM zones
                if -8. <= tempc < -7.:
                    # if temperature is between -8 and -7ºC, then in HM zone 1
                    hmrime += delmr # delmr = mass of new rime accumulated in this zone
                    hmzone1 = 1
                if hmzone1 == 1 and (tempc < -8. or tempc >= -7.):
                    nx = conc[j] * HMFACT * hmrime if splinter == 0 else nhm[hmj] * HMFACT * hmrime
                    # nx = number of splinters produced in this zone. 
                    # conc[j] = conc of orginal graupel particles - so if second gen of splinter - still based on inital graupel
                    # HMFACT = HM splinter production factor (0.2) 
                    # hmrime = mass of rime accumulated in this zone
                    nhm[hm] = nx  # storing splinter conc for this zone
                    tothm[i] += nx # total HM production in this time step 
                    xhm[hm] = x # horizontal position of splinter
                    zhm[hm] = zkm - 0.5 / 6.7 # vertical position of splinter
                    stime[hm] = time # time of splinter production
                    hmi[hm] = i # time step of splinter production
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
                    nx = conc[j] * HMFACT * hmrime if splinter == 0 else nhm[hmj] * HMFACT * hmrime
                    nhm[hm] = nx
                    tothm[i] += nx
                    xhm[hm] = x
                    zhm[hm] = zkm - 0.5 / 6.7
                    stime[hm] = time
                    hmi[hm] = i
                    hm += 1
                    hmrime = 0.
                    hmzone2 = 0

                if -6. <= tempc < -5.:
                    # if temperature is between -6 and -5ºC, then in HM zone 3
                    hmrime += delmr
                    hmzone3 = 1
                if hmzone3 == 1 and (tempc < -6. or tempc >= -5.):
                    nx = conc[j] * HMFACT * hmrime if splinter == 0 else nhm[hmj] * HMFACT * hmrime
                    nhm[hm] = nx
                    tothm[i] += nx
                    xhm[hm] = x
                    zhm[hm] = zkm - 0.5 / 6.7
                    stime[hm] = time
                    hmi[hm] = i
                    hm += 1
                    hmrime = 0.
                    hmzone3 = 0

                if -5. <= tempc < -4.:
                    # if temperature is between -5 and -4ºC, then in HM zone 4
                    hmrime += delmr
                    hmzone4 = 1
                if hmzone4 == 1 and (tempc < -5. or tempc >= -4.):
                    nx = conc[j] * HMFACT * hmrime if splinter == 0 else nhm[hmj] * HMFACT * hmrime
                    nhm[hm] = nx
                    tothm[i] += nx
                    xhm[hm] = x
                    zhm[hm] = zkm - 0.5 / 6.7
                    stime[hm] = time
                    hmi[hm] = i
                    hm += 1
                    hmrime = 0.
                    hmzone4 = 0

                if -4. <= tempc < -3.:
                    # if temperature is between -4 and -3ºC, then in HM zone 5
                    hmrime += delmr
                    hmzone5 = 1
                if hmzone5 == 1 and (tempc < -4. or tempc >= -3.):
                    nx = conc[j] * HMFACT * hmrime if splinter == 0 else nhm[hmj] * HMFACT * hmrime
                    nhm[hm] = nx
                    tothm[i] += nx
                    xhm[hm] = x
                    zhm[hm] = zkm - 0.5 / 6.7
                    stime[hm] = time
                    hmi[hm] = i
                    hm += 1
                    hmrime = 0.
                    hmzone5 = 0

        # for plotting (in main program) -- only every PFACT-th step is kept (not saving every time step)
        pp_ += 1
        if pp_ == PFACT:
            diamp[j][tt_] = rad * 2.e3 # diameter in mm
            rhop[j][tt_] = rhog # density in kg/m3
            #massp[j][tt_] = mass * 1.0e6 # mass in mg
            #surf_temp[j][tt_] = ts # surface temp in K
            timep[j][tt_] = time # time in minutes
            zp[j][tt_] = zkm # altitude in km
            xp[j][tt_] = x # horizontal position in km
            vtp[j][tt_] = vt # terminal velocity in m/s
            effp[j][tt_] = kapb / (PI * rad ** 2 * vt) if kapb is not None else BAD # collection efficiency 
            deltsp[j][tt_] = ts - temp # difference between surface and ambient temperature
            tt_ += 1
            pp_ = 0        

        time = time + DELTIM / 60. # time in minutes

        # 10. move the particle * * * * * * * * * * * * * * * * * * * * * * * * * *
        # particle velocity: vt is positive down
        wpcle = wi - vt # wi = environmental air velocity (positive up) // vt = particle terminal velocity (positive down)

        # vertical and horizontal position
        z = zold + wpcle * DELTIM
        x = xold + hw * DELTIM / 1000.

        # if time > run_time, env temp is warmer than 0, or z < zbase, then break
        if time >= runtime or temp > 273.15 or z <= zbase:
            num[j] = tt_ - 1
            break

        rhoa = pres / (RD * temp)
        zold = z
        xold = x

        # back for more growth -- loop on i
        nx = 0
        indx += 1
    # end of time-step growth loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # NOTE: unlike graupel_thermals_ezri.c, this file's growth loop only
    # sets num[j] inside the break above -- there is no unconditional
    # assignment after the loop. If the loop runs to completion without
    # tripping the break condition, num[j] is left at whatever it was
    # before this call (0 by default, since `num` is a zero-initialised
    # global/static array in the C source). This is translated faithfully.

    return

#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def main():
    """
    executing graupel function and other stuff - again lots of steps

    ! grow and develop all thermals through time - this is recorded, then passed to graupel !
    """

    global zinit, maxup, wbase, zmax, runtime, umax
    global pbase, tbase, zbase, cdepth, slope, intcpt
    global numt, wi, hm, hmj, hmrime, splinter, hmflag, pname, j
    global thtop1, thbase1, thtop2, thbase2, cldtop, downbase
    global wtht1, wthb1, wtht2, wthb2
    global topflag, thb1flag, tht2flag, thb2flag, post1, posb1, post2, posb2
    global totnum, zm8, zm3
    global lab1, lab2, lab3, lab4, lab5, lab6, lab7, lab8, lab9, lab10
    global lab11, lab12, lab13, lab14, lab15, lab16, lab17, lab18, lab19, lab20, lab21


    global particle_diameter, particle_density, particle_mass, particle_surface_temperature
    global particle_terminal_velocity, particle_altitude, particle_horizontal_position, particle_vertical_position
    global save_loc
    global time, alt
    


    # check command line arguments - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pname = sys.argv[0] # python script name
    args = sys.argv[1:]
    hmflag = 0
    if args and args[0] == "-h":
        hmflag = 1
        args = args[1:]
    if len(args) != 5:
        print(f"Usage: {pname} [-h] zi (km) wmax wbase (m/s)  z@wmax (km), "
              f"run_time (min) ", file=sys.stderr)
        sys.exit(0)

    zinit = 1000. * float(args[0]) # initial height of particle (m)
    maxup = float(args[1]) # maximum updraft velocity (m/s)
    wbase = float(args[2]) # updraft velocity at cloud base (m/s)
    zmax = float(args[3]) # height of maximum updraft (km)
    runtime = float(args[4]) # simulation run time (min)

    ## save tings babe! - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    save_loc = "./model_output/" # location to save output files
    os.makedirs(save_loc, exist_ok=True) # if make folder, if already exist, do nothing
    save_name = 'test_output'
    # setting up time and height for model environment
    numt = int(runtime * 60. / DELTIM) ### number of simulation time steps (s)
    time = np.zeros(numt) # empty time array
    time[:] = np.arange(numt) * DELTIM # time array
    #alt = np.zeros(MAX2)
    time = np.zeros(numt)
    alt[:] = np.arange(MAX2) * 10. # height array

    """ for saving ice particle properties """
    particle_diameter = np.full((MAX1, numt), np.nan)
    particle_density = np.full((MAX1, numt), np.nan)
    particle_mass = np.full((MAX1, numt), np.nan)
    particle_surface_temperature = np.full((MAX1, numt), np.nan)
    particle_terminal_velocity = np.full((MAX1, numt), np.nan)
    particle_altitude = np.full((MAX1, numt), np.nan)
    particle_horizontal_position = np.full((MAX1, numt), np.nan)
    particle_vertical_position = np.full((MAX1, numt), np.nan)
    ## save tings babe! - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    
    # initialise arrays and variables - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # umax from continuity
    umax = UFACT * maxup / 2. # calculate horizontal wind

    # zero arrays (numpy arrays are already zero-initialised, but mirror
    # the explicit zeroing from the C source in case this is re-run)
    # to be filled later
    sumd[:] = 0. # quantity to calculate total reflectivity
    concl[:] = 0. # particle concentration at requested level
    diaml[:] = 0. # particle diameter at requested level
    tothm[:] = 0. # total number of splinters produced in each time step
    #nhm[:] = 0. # number of splinters produced in each HM zone

    # create initial particle size distribution (spectrum) - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    totnum = 0
    for jj in range(MAX1):
        radi[jj] = diam[jj] / 2.
        # two exponential fits - spectrum (10/8/94)
        if diam[jj] <= 0.3:
            conc[jj] = 10. ** 4. * math.exp(-39.47 * diam[jj]) # calculating concentration of particles
        else:
            conc[jj] = 10. ** 0.7 * math.exp(-1.3 * diam[jj])
        totnum += conc[jj] # total intitial particle concentration (m-3)
    

    # define cloud properties: pressure / temperature / depth /vertical velocity profile - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pbase = PBASE # pressure at cloud base (mb)
    tbase = TBASE # temperature at cloud base (ºC)
    zbase = ptz(pbase) # height of cloud base (m)
    print(f"cloud base: p = {pbase:5.1f} mb, T = {tbase:4.1f} C, "f"z = {zbase / 1000.:3.1f} km", file=sys.stderr)
    cdepth = CTOP - zbase / 1000. # depth of cloud (km)

    # vertical velocity change with alt -> slope and intercept of (linear) vertical velocity profile
    slope = (maxup - wbase) / (zmax - zbase / 1000.)
    intcpt = wbase - slope * zbase / 1000.

    # build environment profile - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for ii in range(MAX2):
        ps[ii] = pbase - ii / 2.
        alt[ii] = ptz(ps[ii]) # height (m)
        at_, alwc_ = trev(pbase, tbase, ps[ii]) # temp + LWC
        lw[ii] = alwc_ * 1.0e-3
        tr[ii] = at_ + 273.15
        db[ii] = DBAR + ii / 125. # mean cloud drop diameter

        # environmental updraft
        # vv = vertical velocity defined at each height
        if alt[ii] < zbase:
            # below cloud
            vv[ii] = 1.
        elif alt[ii] / 1000. <= zmax:
            # in cloud + alt < max updraft alt
            # increasing updraft to zmax
            vv[ii] = slope * alt[ii] / 1000. + intcpt
        else:
            # in cloud + alt >= zmax
            # decreasing updraft above this
            vv[ii] = (maxup / ((CTOP + 2.) - zmax)) * ((CTOP + 2.) - alt[ii] / 1000.0)

    # number of time points
    numt = 1 + int(runtime * 60. / DELTIM)
    print(f"numt = {numt}", file=sys.stderr)

    # setting up first thermal - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # initial positions of thermals
    thtop1[0] = zinit / 1000. + 0.0 # top of thermal initally at the particles initial height
    cldtop[0] = thtop1[0] # cloud top = thermal top
    thbase1[0] = thtop1[0] - TDEPTH # base of thermal

    if thtop1[0] > CTOP:
        thtop1[0] = CTOP # top of thermal cannot exceed cloud top
    if thbase1[0] < 0.:
        thbase1[0] = 0. # base of thermal cannot be below ground

    # setting up second thermal - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    thtop2[0] = thbase1[0] - TGAP # set second thermal below first thermal by TGAP
    if thtop2[0] < 0.:
        thtop2[0] = 0. # top of second thermal cannot be below ground
    thbase2[0] = thtop2[0] - TDEPTH
    if thbase2[0] < 0.:
        thbase2[0] = 0. # base of second thermal cannot be below ground
    downbase[0] = BAD  # bottom of cloud top debris

    # find environmental grid location of the thermals - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    post1 = posb1 = post2 = posb2 = 0 
    for ii in range(MAX2):
        # find grid locations for thermals
        if ii > 0:
            if thtop1[0] < alt[ii] / 1000. and thtop1[0] >= alt[ii - 1] / 1000.:
                post1 = ii # thermal 1 top
            if thbase1[0] < alt[ii] / 1000. and thbase1[0] >= alt[ii - 1] / 1000.:
                posb1 = ii # thermal 1 base
            if thtop2[0] < alt[ii] / 1000. and thtop2[0] >= alt[ii - 1] / 1000.:
                post2 = ii # thermal 2 top
            if thbase2[0] < alt[ii] / 1000. and thbase2[0] >= alt[ii - 1] / 1000.:
                posb2 = ii # thermal 2 base
    # get vertical velocity at each thermal boundary
    wtht1 = vv[post1] / 2.
    wthb1 = vv[posb1] / 2.
    wtht2 = vv[post2] / 2.
    wthb2 = vv[posb2] / 2.

    # simulate thermals through time - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # positions of thermals (in km) with time
    topflag = thb1flag = tht2flag = thb2flag = 0

    # loop through each time step
    for i in range(1, numt):
        # move thermal 1 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        thtop1[i] = thtop1[i - 1] + (wtht1 * DELTIM) / 1000.
        if thtop1[i] >= CTOP:
            thtop1[i] = CTOP

        thbase1[i] = thtop1[i] - TDEPTH
        if thbase1[i] > 0.:
            thbase1[i] = thbase1[i - 1] + (wthb1 * DELTIM) / 1000.

        # create / move thermal 2 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if tht2flag != 1:
            # if thermal top 2 has not been initilised 
            thtop2[i] = thbase1[i] - TGAP
            if thtop2[i] > 0.: # if thermal top 2 is above ground, then initialise it
                tht2flag = 1
            if thtop2[i] < 0.: # if thermal top 2 is below ground, then set it to ground level
                thtop2[i] = 0.
        else:
            # if thermal top 2 has been initialised, then move it
            thtop2[i] = thtop2[i - 1] + (wtht2 * DELTIM) / 1000.

        thbase2[i] = thtop2[i] - TDEPTH # initalise thermal 2 base
        if thbase2[i] > 0.:
            # thermal 2 base above ground, so it does exist
            thbase2[i] = thbase2[i - 1] + (wthb2 * DELTIM) / 1000.

        # deal with cloud top region * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if thtop1[i] >= CTOP and thbase1[i] >= (CTOP - TDEPTH / 4.) and topflag == 0:
            # thermal is approaching maximum cloud top -> stopping growth 
            topflag = 1 # switch on processes
            thtop1[i] = thtop2[i]
            thbase1[i] = thbase2[i]
            thtop2[i] = thbase1[i] - TGAP
            thbase2[i] = thtop2[i] - TDEPTH
            thb1flag = tht2flag = thb2flag = 0
            cldtop[i - 1] = CTOP

        if topflag == 1:
            # thermal has reached cloud top -> thermal stopped growing, now become region of descending material 
            cldtop[i] = cldtop[i - 1] + (WCTDEB * DELTIM) / 1000. # cloud top region grows
            if cldtop[i] <= GENCTOP:
                cldtop[i] = GENCTOP
            downbase[i] = cldtop[i] - TDEPTH / 4. # base of cloud top region 
            if downbase[i] <= thtop1[i]:
                downbase[i] = thtop1[i]
            if cldtop[i] < thtop1[i]:
                topflag = 0

        else:
            # thermal is not near max cloud top -> still growing
            cldtop[i] = thtop1[i]
            downbase[i] = BAD

        # again find environmental grid location of the thermals * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        post1 = posb1 = post2 = posb2 = 0
        for ii in range(MAX2):
            if ii > 0:
                if thtop1[i] < alt[ii] / 1000. and thtop1[i] >= alt[ii - 1] / 1000.:
                    post1 = ii
                if thbase1[i] < alt[ii] / 1000. and thbase1[i] >= alt[ii - 1] / 1000.:
                    posb1 = ii
                if thtop2[i] < alt[ii] / 1000. and thtop2[i] >= alt[ii - 1] / 1000.:
                    post2 = ii
                if thbase2[i] < alt[ii] / 1000. and thbase2[i] >= alt[ii - 1] / 1000.:
                    posb2 = ii

                # altitude of H-M zone -- note temperature is the same in
                # each zone except for the downdraughts
                if 266.5 < tr[ii] < 266.75:
                    zm8 = alt[ii] / 1000.
                if 272.0 < tr[ii] < 272.25:
                    zm3 = alt[ii] / 1000.

        # again get vertical velocity at each thermal boundary
        wtht1 = vv[post1] / 2.
        wthb1 = vv[posb1] / 2.
        wtht2 = vv[post2] / 2.
        wthb2 = vv[posb2] / 2.
    # end of simulating thermals through time - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # growing all particles in graupel function - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # using pre-defined evolution of thermals
    # Note particles produced in the H-M zones in graupel() are splinters.
    # Their growth is dealt with after the initial set of trajectories, by number.
    wi = 0. # updraft m/s
    hm = 0
    hmj = 0
    hmrime = 0.
    splinter = 0 # first simulating graupel particle growth
    for jj in range(MAX1):
        j = jj # j = current particle index
        graupel()

    if hmflag == 0:
        hm = 1

    if hm > 0:
        splinter = 1
        hmj = 0
        # NOTE: `hm` can grow inside graupel() (new splinter groups spawn
        # more splinter groups), so this loop bound is re-checked every
        # iteration, exactly like the C `for (hmj = 0; hmj < hm - 1; hmj++)`.
        while hmj < hm - 1:
            # for every HM splinter group created -> run graupel to simulate growth and movement
            graupel()
            hmj += 1

    # total number of h-m particles produced versus time / number of ice
    # pcles present at the beginning - this gives value of f in our paper
    for i in range(numt):
        cumtot[i] = 1

    for i in range(numt):
        if i > 0:
            cumtot[i] = cumtot[i - 1] + tothm[i] / totnum

    plot_results()

    ## making netcdf! + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
    fname = os.path.join(save_loc, f"{save_name}.nc")
    nc = Dataset(fname, "w", format="NETCDF4")
    # dimensions - - - - - -
    nc.createDimension("time", numt)
    nc.createDimension("height", MAX2)
    nc.createDimension("particle", MAX1) ## we are having particles as their own dimension

    # variables - - - - - -
    time_var = nc.createVariable("time", "f4", ("time",))
    time_var.units = f"seconds since simulation start (dt={DELTIM}s)"
    time_var.long_name = "simulation time"
    time_var[:] = np.arange(numt) * DELTIM

    height_var = nc.createVariable("height", "f4", ("height",))
    height_var.units = "m"
    height_var.long_name = "height above ground"
    height_var[:] = np.asarray(alt)

    particle_var = nc.createVariable("particle", "i4", ("particle",))
    particle_var.units = "dimensionless"
    particle_var.long_name = "particle index"
    particle_var[:] = diam #np.arange(MAX1)

    ##### deal with environment variables #############
    ## envi variables varying with height:
    env_nc = nc.createGroup("environment")
    pressure_var = env_nc.createVariable("pressure", "f4", ("height",), fill_value=np.nan)
    pressure_var.units = "hPa"
    pressure_var.long_name = "environmental pressure"
    pressure_var[:] = ps

    temperature_var = env_nc.createVariable("temperature", "f4", ("height",), fill_value=np.nan)
    temperature_var.units = "K"
    temperature_var.long_name = "environmental temperature"
    temperature_var[:] = tr

    lwc_var = env_nc.createVariable("lwc", "f4", ("height",), fill_value=np.nan)
    lwc_var.units = "kg m-3"
    lwc_var.long_name = "environmental liquid water content"
    lwc_var[:] = lw

    drop_diameter_var = env_nc.createVariable("cloud_drop_diameter", "f4", ("height",), fill_value=np.nan)
    drop_diameter_var.units = "um"
    drop_diameter_var.long_name = "mean cloud drop diameter"
    drop_diameter_var[:] = db

    velocity_var = env_nc.createVariable("vertical_velocity", "f4", ("height",), fill_value=np.nan)
    velocity_var.units = "m s-1"
    velocity_var.long_name = "environmental vertical velocity"
    velocity_var[:] = vv

    ## envi variables varying with time:
    thermal_top_1 = env_nc.createVariable("thermal_top_1", "f4", ("time",), fill_value=np.nan)
    thermal_top_1.units = "km"
    thermal_top_1.long_name = "top altitude of thermal 1"
    thermal_top_1[:] = np.where(thtop1[:numt] == BAD, np.nan, thtop1[:numt])

    thermal_base_1 = env_nc.createVariable("thermal_base_1", "f4", ("time",), fill_value=np.nan)
    thermal_base_1.units = "km"
    thermal_base_1.long_name = "base altitude of thermal 1"
    thermal_base_1[:] = np.where(thbase1[:numt] == BAD, np.nan, thbase1[:numt])

    thermal_top_2 = env_nc.createVariable("thermal_top_2", "f4", ("time",), fill_value=np.nan)
    thermal_top_2.units = "km"
    thermal_top_2.long_name = "top altitude of thermal 2"
    thermal_top_2[:] = np.where(thtop2[:numt] == BAD, np.nan, thtop2[:numt])

    thermal_base_2 = env_nc.createVariable("thermal_base_2", "f4", ("time",), fill_value=np.nan)
    thermal_base_2.units = "km"
    thermal_base_2.long_name = "base altitude of thermal 2"
    thermal_base_2[:] = np.where(thbase2[:numt] == BAD, np.nan, thbase2[:numt])

    cloud_top = env_nc.createVariable("cloud_top", "f4", ("time",), fill_value=np.nan)
    cloud_top.units = "km"
    cloud_top.long_name = "cloud top altitude"
    cloud_top[:] = np.where(cldtop[:numt] == BAD, np.nan, cldtop[:numt])

    debris_base = env_nc.createVariable("debris_base", "f4", ("time",), fill_value=np.nan)
    debris_base.units = "km"
    debris_base.long_name = "base altitude of descending cloud-top debris"
    debris_base[:] = np.where(downbase[:numt] == BAD, np.nan, downbase[:numt])

    ##### deal with ice particle variables #############
    ice_nc = nc.createGroup("ice_particles")

    diameter_var = ice_nc.createVariable("diameter", "f4", ("particle", "time"), fill_value=np.nan)
    diameter_var.units = "m"
    diameter_var.long_name = "graupel particle diameter"
    diameter_var[:] = particle_diameter

    density_var = ice_nc.createVariable("density", "f4", ("particle", "time"), fill_value=np.nan)
    density_var.units = "kg m-3"
    density_var.long_name = "graupel particle bulk density"
    density_var[:] = particle_density

    mass_var = ice_nc.createVariable("mass", "f4", ("particle", "time"), fill_value=np.nan)
    mass_var.units = "kg"
    mass_var.long_name = "graupel particle mass"
    mass_var[:] = particle_mass

    surface_temperature_var = ice_nc.createVariable("surface_temperature", "f4", ("particle", "time"), fill_value=np.nan)
    surface_temperature_var.units = "K"
    surface_temperature_var.long_name = "graupel particle surface temperature"
    surface_temperature_var[:] = particle_surface_temperature

    terminal_velocity_var = ice_nc.createVariable("terminal_velocity", "f4", ("particle", "time"), fill_value=np.nan)
    terminal_velocity_var.units = "m s-1"
    terminal_velocity_var.long_name = "graupel particle terminal velocity"
    terminal_velocity_var[:] = particle_terminal_velocity

    altitude_var = ice_nc.createVariable("altitude", "f4", ("particle", "time"), fill_value=np.nan)
    altitude_var.units = "m"
    altitude_var.long_name = "graupel particle altitude"
    altitude_var[:] = particle_altitude

    horizontal_position_var = ice_nc.createVariable("horizontal_position", "f4", ("particle", "time"), fill_value=np.nan)
    horizontal_position_var.units = "km"
    horizontal_position_var.long_name = "graupel particle horizontal position"
    horizontal_position_var[:] = particle_horizontal_position


    nc.close()
    ## + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 

    
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def plot_results():
    """
    Re-creates each of the original plotting blocks using matplotlib, since
    the original custom graphics library (gopen/window/axes/line/label/
    mark/conto/grid/fill/gpause/gclose/xtick/ytick/badset) is not available.
    Each figure below corresponds to one "window(...)/gpause()/gclear()"
    block in the C source, in the same order, and plots the same data.
    """
    realdate = datetime.now().strftime("%H:%M:%S %a %d %b %Y")
    lab1_ = f"{pname}: {realdate}"
    lab2_ = f"Cloud base: T = {TBASE:4.1f} C; P = {PBASE:5.1f} mb"
    lab3_ = f"TDEPTH = {TDEPTH:3.1f} km"
    lab4_ = f"TGAP = {TGAP:3.1f} km"
    lab5_ = f"Lmax = {RATTH:4.2f} Lad"
    lab6_ = f"wmax = {maxup:4.1f} m/s"
    lab7_ = f"Lct = {RATCT:4.2f} Lad"
    lab8_ = f"Ldeb = {RATDEB:4.2f} Lad"
    lab11_ = f"largest initial diameter = {diam[9]:3.1f} mm"
    footer_lines = [lab1_, lab2_, lab3_ + "   " + lab4_,
                    lab5_ + "   " + lab6_, lab7_ + "   " + lab8_, lab11_]
    if umax > 0:
        lab9_ = f"umax = {umax:4.1f} m/s"
        lab10_ = f"DDEPTH = {DDEPTH:4.1f} km"
        lab18_ = f"Ldown = {RATDOWN:4.2f} Lad"
        footer_lines += [lab9_ + "   " + lab10_, lab18_]
    footer = "\n".join(footer_lines)

    def add_footer(fig):
        fig.text(0.01, 0.01, footer, fontsize=7, va="bottom")

    t_axis = np.zeros(numt)
    for i in range(1, numt):
        t_axis[i] = t_axis[i - 1] + DELTIM / 60.

    # thermal-boundary traces used in the height plots -- once a boundary
    # array value decreases (thermal cycle restarts), mask the rest as BAD,
    # matching the C source's one-off masking loop
    thtop1_plot = thtop1[:numt].copy()
    thtop2_plot = thtop2[:numt].copy()
    thbase1_plot = thbase1[:numt].copy()
    thbase2_plot = thbase2[:numt].copy()
    for i in range(1, numt):
        if thtop1_plot[i - 1] < BAD and thtop1_plot[i] < thtop1_plot[i - 1]:
            thtop1_plot[i] = BAD
        if thtop2_plot[i - 1] < BAD and thtop2_plot[i] < thtop2_plot[i - 1]:
            thtop2_plot[i] = BAD
        if thbase1_plot[i - 1] < BAD and thbase1_plot[i] < thbase1_plot[i - 1]:
            thbase1_plot[i] = BAD
        if thbase2_plot[i - 1] < BAD and thbase2_plot[i] < thbase2_plot[i - 1]:
            thbase2_plot[i] = BAD

    # 1) height plot (particle trajectories + thermal boundaries + H-M zone)
    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1 + hm):
        n = num[jj]
        tvals = _mask_bad(np.where(timep[jj, :n] > 0., timep[jj, :n], np.nan))
        zvals = _mask_bad(np.where(zp[jj, :n] > 0., zp[jj, :n], np.nan))
        style = "-" if jj < MAX1 else "--"
        ax.plot(tvals, zvals, style, linewidth=0.8)
    ax.plot(t_axis, _mask_bad(cldtop[:numt]), "k-", label="cldtop")
    ax.plot(t_axis, _mask_bad(thtop1_plot), "b-", label="thtop1")
    ax.plot(t_axis, _mask_bad(thbase1_plot), "b--", label="thbase1")
    ax.plot(t_axis, _mask_bad(downbase[:numt]), "g--", label="downbase")
    ax.plot(t_axis, _mask_bad(thtop2_plot), "r-", label="thtop2")
    ax.plot(t_axis, _mask_bad(thbase2_plot), "r--", label="thbase2")
    ax.axhline(zm8, color="0.5", linestyle=":", linewidth=1)
    ax.axhline(zm3, color="0.5", linestyle=":", linewidth=1)
    ax.set_xlabel("t (min)")
    ax.set_ylabel("z (km)")
    ax.set_xlim(0., 90.)
    ax.set_ylim(3., 12.)
    ax.legend(fontsize=7)
    add_footer(fig)
    plt.savefig(f'{save_loc}height_plt_trajectories_boundaries.png')

    fig.tight_layout()

    # 2) efficiency plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1 + hm - 1):
        n = num[jj]
        dvals = _mask_bad(np.where(diamp[jj, :n] > 0., diamp[jj, :n], np.nan))
        evals = _mask_bad(np.where(effp[jj, :n] > 0., effp[jj, :n], np.nan))
        ax.plot(dvals, evals, linewidth=0.8)
    ax.set_xlabel("diam (m)")
    ax.set_ylabel("Coll. Eff.")
    ax.set_xlim(0., 8.)
    ax.set_ylim(0., 1.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig(f'{save_loc}efficiency.png')


    # 3) surface temperature excess
    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1 + hm - 1):
        n = num[jj]
        dvals = _mask_bad(np.where(diamp[jj, :n] > 0., diamp[jj, :n], np.nan))
        tvals = _mask_bad(np.where(deltsp[jj, :n] > -5., deltsp[jj, :n], np.nan))
        ax.plot(dvals, tvals, linewidth=0.8)
    ax.set_xlabel("diam (m)")
    ax.set_ylabel("Ts-Ta (deg)")
    ax.set_xlim(0., 8.)
    ax.set_ylim(-5., 5.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig(f'{save_loc}surface_temperature_excess.png')

    # 4) multiplication factor f vs time
    f_xx = np.zeros(numt)
    f_yy = np.zeros(numt)
    time_labels = []
    for i in range(1, numt):
        f_xx[i] = f_xx[i - 1] + DELTIM / 60.
        for target in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            if target - 0.05 < f_xx[i] < target + 0.05:
                msg = f"Time = {f_xx[i]:4.1f} min, f = {cumtot[i]:5.2e}"
                print(msg, file=sys.stderr)
                time_labels.append(msg)
        f_yy[i] = math.log10(cumtot[i]) if cumtot[i] > 0 else BAD

    maxy, miny = scale_axis(f_yy[:numt])
    nticks = int(maxy - miny + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(f_xx, _mask_bad(f_yy))
    ax.set_xlabel("t (min)")
    ax.set_ylabel("log f")
    ax.set_xlim(0., 90.)
    ax.set_ylim(miny, maxy)
    ax.set_yticks(np.linspace(miny, maxy, max(nticks, 2)))
    label_footer = footer + "\n" + "\n".join(time_labels)
    fig.text(0.01, 0.01, label_footer, fontsize=6, va="bottom")
    fig.tight_layout()
    plt.savefig(f'{save_loc}multiplication_factor.png')

    # 5) initial spectrum
    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1):
        speci[jj] = math.log10(conc[jj]) if conc[jj] > 0. else BAD
    ax.plot(diam, _mask_bad(speci), "o-")
    ax.set_xlabel("D (mm)")
    ax.set_ylabel("N (m^-3 mm^-1)")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-1.0, 6.0)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig(f'{save_loc}initial_spectrum.png')

    # 6) diameter vs height
    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1 + hm - 1):
        n = num[jj]
        dvals = _mask_bad(np.where(diamp[jj, :n] > 0., diamp[jj, :n], np.nan))
        zvals = _mask_bad(np.where(zp[jj, :n] > 0., zp[jj, :n], np.nan))
        ax.plot(dvals, zvals, linewidth=0.8)
    ax.set_xlabel("diam (mm)")
    ax.set_ylabel("Height (km)")
    ax.set_xlim(0., 6.)
    ax.set_ylim(3., 13.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig(f'{save_loc}diameter_vs_height.png')

if __name__ == "__main__":
    main()

print('ran')