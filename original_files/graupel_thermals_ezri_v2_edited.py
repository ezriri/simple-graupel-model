"""
Python translation of graupel_thermals_ezri.c

precip -- A wee model to predict the development of precipitation.

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

TRANSLATION NOTES
-----------------
1. `trev()` (adiabatic temperature/LWC vs pressure) is *declared* in the C
   source (`float trev();`) and called, but its body is not in this file --
   it must live in another .c file that wasn't supplied. A stub is provided
   below; replace it with the real implementation before running.

2. `vapour()` and `gr()` *are* fully defined in this file (unlike the
   previous graupel-only file you gave me, where they were only declared),
   so those are translated faithfully below.

3. `BAD` is a sentinel value used throughout the C code to flag "missing/
   invalid" data points, but it's a macro from "cdfhdr.h" which was not
   supplied. Given the code checks `value < 99999.` to decide whether a
   value is valid, BAD is assumed here to be 999999.0. Adjust
   `BAD` below if your header defines it differently.

4. The C program draws all its plots using a custom, in-house graphics
   library (`gopen`, `gclear`, `window`, `axes`, `line`, `label`, `mark`,
   `conto`, `grid`, `fill`, `gpause`, `gclose`, `xtick`, `ytick`, `badset`).
   None of these are standard C library calls and their implementations
   were not supplied (they live in "cdfhdr.h" / a linked graphics library).
   It is not possible to reproduce their exact visual behaviour, so each
   plotting block has been translated into an equivalent, clearly-labelled
   matplotlib figure that plots the same data. The *numerical* content
   (the environment profiles, thermal tracking, particle growth, and
   reflectivity calculations) is translated exactly.

5. All C globals that are shared between functions (because they were not
   declared as local variables where used) are kept as Python module-level
   globals, accessed with `global` statements, exactly mirroring the
   original data flow.
"""

import sys
import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# #define constants
# ---------------------------------------------------------------------------
diam = np.array([.05, .1, .15, .2, .3, .4,
                  .5, .6, .7, .8, .9, 1., 1.1, 1.2,
                  1.4, 1.6, 1.8, 2.0, 2.2, 2.4]) ## i assume initial particle diameters (mm)

MAX1 = len(diam)   # number of particle size categories
MAX2 = 1500        # number of altitude values 
MAX3 = 9000        # number of time steps
MAX4 = 24          # number of grid points in vertical

DBAR = 20
RATTH = 0.1
RATCT = 0.1
RATDEB = 0.05
RATDOWN = 0.05
WCTDEB = -3.
CTOP = 8.
TDEPTH = 2. # thermal depth (km)
TGAP = 1. # gap between thermals ??
TWID = 2.
DOWN = 0.4
UMAX = 2.
ZCT = TDEPTH / 6.
ZCT1 = ZCT / 3.
ZCT2 = 2. * ZCT / 3.

P0 = 1013. # surface (sea level) pressure (mb)
T0 = 288. # surface temperature (K)
GAMMA = 0.0065                              # constant lapse rate (ºC/m)
RHOI = 920
EPS = 0.622
PI = 3.14159                              # constant pi 
DELTIM = 5. # time step interval (s)??
RD = 287.05 # gas constant for dry air (J/kg/K)
RV = 461.51 # gas constant for water vapor (J/kg/K)
TK = 2.32e-2 # thermal conductivity of air (W/m/K) ??
D = 2.11e-5 # molecular diffusivity of water vapor in air (m^2/s) ??
TTR = 273.16 # triple point temperature of water (K) ??
RW = 461.51
LS = 2.837e6
LF = 3.12e5
LV = 2.5e6
NMAX = 500.e6
DIS = 0.25
GRAV = 9.81                             # gravity
ETA = 1.67e-5
CPW = 4.27e3

# see note 3 above -- best-effort guess at the value of BAD
BAD = np.nan

# alternative spectrum (commented out in the original):
# diam = np.array([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3,
#                   0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.])

# ---------------------------------------------------------------------------
# Module-level globals (mirrors C file/global-scope variables shared between
# main(), graupel(), drag(), tsurf())
# ---------------------------------------------------------------------------
# the whole environment is initialised with empty arrays defined by MAX values
lw = np.zeros(MAX2)                              # LWC ??
db = np.zeros(MAX2)                              # dbar ??
vv = np.zeros(MAX2)                              # vertical velocity
alt = np.zeros(MAX2)                             # altitude (m?)
ps = np.zeros(MAX2)                              # pressure (mb?)
tr = np.zeros(MAX2)                              # temperature (Kelvin)
wid = 0.0                                        # width of channels
conc = np.zeros(MAX1)                            # concentration of particles in each size category
diaml = np.zeros(MAX1)                           # diameter spectrum at requested level
concl = np.zeros(MAX1)                           # concentration spectrum at requested level
radi = np.zeros(MAX1)                            # inital radius
diamh = np.zeros((MAX4, MAX1, MAX3))             # diameter for particular height
diamx = np.zeros(MAX1)                           # diameter in window coords ??
diamp = np.zeros((MAX1, MAX3))                   # diameter for straight plotting
vtp = np.zeros((MAX1, MAX3))                     # terminal velocity for plotting
timep = np.zeros((MAX1, MAX3))                   # time stored for plotting
rhop = np.zeros((MAX1, MAX3))                    # density of particle
zp = np.zeros((MAX1, MAX3))                      # altitude of each size
zptop = np.zeros((MAX1, MAX3))                   # rel to ground and cloud top
xp = np.zeros((MAX1, MAX3))                      # pos of particle relative to centre of thermal
''' multiple thermals defined '''
thtop1 = np.zeros(MAX3)                          # top alt of thermal 1
thbase1 = np.zeros(MAX3)                         # bottom alt of thermal 1
thtop2 = np.zeros(MAX3)                          # top alt of thermal 2
thbase2 = np.zeros(MAX3)                         # bottom alt of thermal 2
cldtop = np.zeros(MAX3)                          # top of cloud top debis
downbase = np.zeros(MAX3)                        # bottom of cloud top debris
wtht1 = wthb1 = wtht2 = wthb2 = 0.0              # velocity of top and bottom of thermals
rad = 0.0                                        # radius of graupel particles
wi = 0.0                                         # updraft (environmental air velocity) (m/s)
lwc = 0.0                                        # LWC ?? why lwc defined twice
rhoa = 0.0                                       # density of air (kg/m^3?)
reft = np.zeros(MAX3)                            # reflectivity with time
refg = np.zeros((MAX3, MAX4))                    # reflectivity with time and height
zinit = 0.0                                      # initial z altitude ?? - starting altitude of graupel particle
cf = 0.0                                         # # used to calculate reflectivity
sumd = np.zeros(MAX3)                            # used to calculate reflectivity
sumr = np.zeros((MAX4, MAX3))                    # used to calculate reflectivity
pbase = tbase = 0.0                              # pressure (mb) and temp (c) at cloud base
zbase = cdepth = 0.0                             # alt of cloud base and cloud depth
time = 0.0                                       # time in minutes
at = alwc = 0.0                                  # adiabatic temp and adiabatic LWC
zgrid = np.zeros(MAX4)                           # vertical grid
xx = np.zeros(MAX3)                              # d.v. for plotting
yy = np.zeros(MAX3)                              # d.v. for plotting
c1 = 0.0                                         # offset for framing spectra
spec = np.zeros(MAX1)                            # d.v. conc
specy = np.zeros(MAX1)                           # conc only for valid diameters
xinit = yinit = 0.0                              # initilisation pts for windows
x1 = x2 = wy1 = wy2 = 0.0                        # window coordinates (for plotting??)
maxup = 0.0                                      # max updraft
wbase = zmax = 0.0                               # updraft at cloud base and max altitude of updraft
slope = intcpt = 0.0                             # variables to calculate vertical velocity profile
runtime = 0.0                                    # length of run in mins ?? - why 0 
es = esi = 0.0                                   # water and ice vapour pressure
nre = nsh = nnu = 0.0                            # Reynolds, Sherwood and Nusslet no
xmax = 0.0                                       # maximum value on x-axis
level = 0.0                                      # level used for spectrum

numt = 0                                         # number of time points for plotting
num = np.zeros(MAX1, dtype=int)                  # number of time points for plotting
j = 0                                            # global channel counter
nxtick = 0                                       # number of ticks on x-axis
topflag = tht2flag = thb2flag = 0                # flag set if base of thermal at top

post1 = posb1 = post2 = posb2 = 0                # index for thermal velocities

pname = ""
lab1 = lab2 = lab3 = lab4 = lab5 = lab6 = ""     # for labels

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
    return np.where(arr >= 99999., np.nan, arr)
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 


""" 
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 main functions 
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
"""
def graupel(j):
    """
    Grow 1 graupel particle (channel `j`) in the environment, through all time steps.
    j is a reference loc for all variables
    
    ! there is a lot happening in this function !
    
    2 sub loops in this function:
    1. loop through each time step
        2. loop through cloud drop size spectrum (24 channels)

    I have broken it up into 11 sections - seperating out specific processes     
    """
    global rhoa, rad, lwc, es, esi, nre, nsh, nnu, wi
    global num

    ### starting altitude of graupel particle (in environment) - - - - - - - - - - - - - - 
    pos = 0
    # loop for graupel environment values -- get pointer to initial height
    for ii in range(MAX2):
        if ii > 0:
            if zinit < alt[ii] and zinit >= alt[ii - 1]:
                pos = ii
    z = alt[pos]
    zold = z # starting altitude of graupel particle - - - - - - - - - - - - - -

    # initial environment values - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pres = ps[pos] * 100. # pressure in Pa
    temp = tr[pos] # temperature in K
    lwc = lw[pos] # liquid water content in g/m^3
    dbar = db[pos] # cloud drop diameter ?? atmospheric pressure?
    timep[j][0] = 0. # time in seconds 
    zp[j][0] = z / 1000. # altitude in km
    xp[j][0] = 0. # horizontal position of particle 
    # density of air
    rhoa = pres / (RD * temp)
    # kinematic viscosity (not used)
    nu = ETA / rhoa
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   
    # creating graupel particle! - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # initial radius, diameter, density and mass of particle
    # radi has been assigned diam/2 in main() function
    rad = radi[j] * 1.0e-3 # initial radius (m)
    diamp[j][0] = radi[j] * 2. # initial diameter (m)
    rhog = 900. # prescribed density (kg/m^3)
    rhop[j][0] = rhog # initial density - 900 (kg/m^3)
    # initial mass
    mass = 4. * PI * rad ** 3 * rhog / 3. # assume spherical particle (kg)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # set counters and things - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ptime = 0. # current time in minutes
    tflag = 0 # warm growth flag - if become 1 -> surface temp is >=0 -> unfrozen water is shed
    hw = 0. # horizontal wind (m/s) ??
    wi = 0. # environmental air vertical velocity
    indx = 0
    downflag = 0 # particle is not in downdraft -> can change later. if = 1 -> particle in downdraft
    graup = 0 # has ice particle surpassed 200-300 um threshold to start riming

    # calculate vt and delmr for a 300 um diameter particle
    # Assume density of 220 g/cm^3 just now as a quick fix - 7/7/94
    # and use simple formula for kapb
    cd = drag(0.15e-3, 220., rhoa) # drag coefficient for 300um (diameter) particle (function uses radius)
    vt300 = 8. * 0.15 * 1.0e-3 * 220. * GRAV / (3. * rhoa * cd) # terminal velocity for 300um particle
    dmr300 = PI * 0.15e-3 ** 2 * vt300 * lwc * DELTIM # riming mass change for 300um particle
    dr300 = dmr300 / (4. * PI * 0.15e-3 ** 2 * 220.) # radius change for 300um particle
    # these values ^^ are used for smooth transition for riming between 200-300 um diameter particles

    global vtsave
    global ddsave 
    global drtrans
    global delmd 
    global delmr
    global delrd
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # time-step growth loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # this is the main part, go through each time step and calculate growth of graupel particle
    for i in range(1, MAX3):

        # 1. calculate drag for particle * * * * * * * * * * * * * * * * * *
        cd = drag(rad, rhog, rhoa)

        tempc = temp - 273.15 # temperature in celsius
        dmm = rad * 2.0e3 # diameter in mm 

        # 2. calculate terminal velocity * * * * * * * * * * * * * * * * * *
        if dmm <= 0.30:
            # for small particles, use empirical relationship, based on temp
            if dmm <= 0.2:
                if (-6.5 < tempc <= -4.) or (-16 < tempc <= -14.):
                    vt = (-0.014 + 0.395 * dmm - 0.177 * dmm ** 2 + 0.073 * dmm ** 3 - 0.0153 * dmm ** 4)
                elif (-14. < tempc <= -12.) or (-8 < tempc <= -6.5) or tempc > -4:
                    vt = 40. * dmm
                elif -12. < tempc <= -8:
                    vt = 51. * dmm
                elif tempc <= -16:
                    vt = 60. * dmm
                vt *= 0.01
                vtsave = vt
            else:
                if graup == 0:
                    vt200 = vtsave ## unsure of what is happening here ??
                    vt = vt200 + (vt300 - vt200) * (dmm - 0.2) / 0.1
        else:
            # gravity and drag for larger particles
            vt = 8. * rad * rhog * GRAV / (3. * rhoa * cd)
            vt = vt ** 0.5 
        ## importantly vt -> downwards is positive

        # 3. calculate rate of diffusional growth of ice crystals * * * * * * * * * * * * * * * * * *
        # need to know surface temp
        # On first round, use temp + 2.4 for surface temp (not sure why ??)
        if i == 1:
            ts = temp + 2.4

        # dm for diffusion - only for Diameter < 300 um ?? - i think this bit is a bit weird
        if dmm <= 0.30:
            if dmm <= 0.20:
                # for particles < 200 um - emperical growth based on temp
                ts = temp
                delrd = 1.0e-6 * gr(tempc)
                ddsave = delrd
            else:
                # for particles between 200-300 um - smooth transition between
                if graup == 0:
                    dd200 = ddsave
                    graup = 1
                drtrans = dd200 + (dr300 - dd200) * (dmm - 0.2) / 0.1


        # 4. calculate vapour pressure * * * * * * * * * * * * * * * * * * * * * * * * * * 
        esi = 100. * vapour(TTR) * (math.exp((ts - TTR) * LS / (RW * ts * TTR))) # ice vapour pressure at surface temp
        rvs = esi / (RV * ts)
        es = 100. * vapour(temp) # water vapour pressure at ambient temp
        re = es / (RV * temp)
        # Sherwood number as per Mason (1971) - mass transfer 
        nsh = 0.58 * nre ** 0.5
        # Nusselt number as per Mason (1971) - heat transfer
        nnu = nsh
        ## vapour growth of crystals
        delmd = 2. * PI * rad * D * nsh * (re - rvs) * DELTIM if dmm > 0.2 else 0. # diffusional growth increment mass (kg) for particles > 200 um

        # 5. calculate rimming * * * * * * * * * * * * * * * * * * * * * * * * * * 
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
                    # 5.1 Collection kernel calculation - for each channel * * * * * * * * * * * * * * * * * * * * * * * * * *
                    xi = (26. - (k + 1.)) * 1.0e-6 # droplet radius
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

                    # 5.2 Impact velocity of cloud drops calculation * * * * * * * * * * * * * * * * * * * * * * * * * *
                    # impact velocity of impinging cloud drops from R & H
                    ns = 2. * vt * xi * xi * 1000. / (9. * ETA * rad) # Stokes parameter
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

                    # 5.3 Density of newly formed rime calculation * * * * * * * * * * * * * * * * * * * * * * * * * *
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

            # 5.4 surface temperature calculation * * * * * * * * * * * * * * * * * * * * * * * * * *
            # surface temperature of graupel particle
            if tflag == 0:
                # in frozen growth regime - droplets freeze on impact (but depend on amount of rime)
                if kapb >= 0.2:
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

            # 5.5 fraction of unfrozen water shed * * * * * * * * * * * * * * * * * * * * * * * * * * 
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


        # 6. calculate change in radius of graupel particle (from mass / diffusional growth) * * * * * * * * * * * * * * * * * * * * * * * * * *
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
            delrd = abs(delmd) / (4. * PI * rad * rad * RHOI)
            if delmd < 0.:
                delrd = -delrd
        delr = delrr + delrd
        rad = rad + delr

        # 7. changes in mass and bulk density * * * * * * * * * * * * * * * * * * * * * * * * * *
        # new bulk density of the graupel particle
        mass = mass + delmr
        rhog = mass * 3. / (4. * PI * rad ** 3)
        if rhog < 100.:
            rhog = 100.
        if rhog > 900.:
            rhog = 900.

        ptime = ptime + DELTIM / 60. # current time in minutes

        # 8. move the particle * * * * * * * * * * * * * * * * * * * * * * * * * *
        # particle velocity: vt is positive down
        wpcle = wi - vt # wi = environmental air velocity (positive up) // vt = particle terminal velocity (positive down)

        # vertical and horizontal position
        z = zold + wpcle * DELTIM # new position of particle
        xp[j][i] = xp[j][i - 1] + hw * DELTIM / 1000. # move particle horizontally

        # check and change environmental conditions - where in environment has particle moved to?
        for ii in range(MAX2):
            if ii > 0:
                if z < alt[ii] and z >= alt[ii - 1]:
                    pos = ii
        pres = ps[pos] * 100. # what pressure level at?
        zkm = z / 1000. # what altitude particle at? (km)

        # 9. where is the particle with respect to the thermal? * * * * * * * * * * * * * * * * * * * * * * * * * *
        # this modifies the environmental conditions -> alter graupel growth and movement

        hw = 0
    
        if downflag == 0: # particle is not in downdraft
            if ((zkm <= thtop1[i] - ZCT and zkm >= thbase1[i]) or (zkm <= thtop2[i] - ZCT and zkm >= thbase2[i])):
                # particle still in thermal
                temp = tr[pos] - 2.
                lwc = RATTH * lw[pos]
                dbar = db[pos]
                wi = vv[pos]

            # overshoot cloud top?
            if zkm >= CTOP:
                zkm = CTOP

            # in cloud top region (top of thermal 1)
            if zkm <= thtop1[i] and zkm > thtop1[i] - ZCT1:
                temp = tr[pos] - 4.
                lwc = RATCT * lw[pos]
                dbar = db[pos]
                hw = UMAX
                wi = vv[pos] / 2.
            if zkm <= thtop1[i] - ZCT1 and zkm > thtop1[i] - ZCT2:
                temp = tr[pos] - 3.
                lwc = RATCT * lw[pos]
                dbar = db[pos]
                hw = 2. * UMAX / 3.
                wi = vv[pos] / 1.7
            if zkm <= thtop1[i] - ZCT2 and zkm > thtop1[i] - ZCT:
                temp = tr[pos] - 3.
                lwc = RATCT * lw[pos]
                dbar = db[pos]
                hw = UMAX / 3.
                wi = vv[pos] / 1.3

            # out of thermal, into debris under thermal
            if ((zkm > thtop1[i] or zkm < thbase1[i]) and (zkm > thtop2[i] or zkm < thbase2[i]) and (zkm < downbase[i])) or (zkm > thtop1[i] and zkm > thtop2[i]):
                temp = tr[pos] - 4.
                lwc = RATDEB * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = 0.

            # in descending thermal remnants
            if zkm > thtop1[i] and zkm <= cldtop[i] and zkm >= downbase[i]:
                temp = tr[pos] - 4.
                lwc = RATDEB * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = WCTDEB

            # horizontal position in km 
            if xp[j][i] >= TWID - DOWN:
                # particle has moved out of thermal horizontally - in downdraft now
                temp = tr[pos] - 6.
                lwc = RATDOWN * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = -3.
                downflag = 1

        # 10. recording data * * * * * * * * * * * * * * * * * * * * * * * * * *
        # for plotting (in main program)
        ## what about size distribution of cloud drops?
        diamp[j][i] = rad * 2.e3 # particle diameter in mm
        rhop[j][i] = rhog # particle density in kg/m^3
        timep[j][i] = ptime # time in minutes ??
        zp[j][i] = zkm # altitude in km
        vtp[j][i - 1] = vt # teminal velocity of particle in m/s

        # 11. decision if to continue to next time step * * * * * * * * * * * * * * * * * * * * * * * * * *
        # if time > run_time, env temp is warmer than 0, or z < zbase, then break
        if ptime > runtime or temp > 273.15 or z <= zbase:
            num[j] = indx
            break
        rhoa = pres / (RD * temp)
        zold = z

        indx += 1
    # end of time-step growth loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # i dont think this else statement needed?? no if to match it
    #else:
    #    num[j] = indx - 1

    return 
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def main():
    """
    executing graupel function and other stuff - again lots of steps

    ! grow and develop all thermals through time - this is recorded, then passed to graupel !
    """

    global zinit, maxup, wbase, zmax, runtime, level
    global pbase, tbase, zbase, cdepth, slope, intcpt
    global numt, wi, cf, pname, j
    global thtop1, thbase1, thtop2, thbase2, cldtop, downbase
    global wtht1, wthb1, wtht2, wthb2
    global topflag, tht2flag, thb2flag, post1, posb1, post2, posb2
    global concl, diaml, conc, radi
    global lab1, lab2, lab3, lab4, lab5, lab6
    global c1, xmax, nxtick

    # check command line arguments - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pname = sys.argv[0] # python script name
    if len(sys.argv) != 7:
        print(
            f"Usage: {pname} zi (km) wmax wbase (m/s) z@wmax (km) "
            f"run_time (min) zspec (km)",
            file=sys.stderr,
        )
        sys.exit(0)

    zinit = 1000. * float(sys.argv[1]) # initial height of particle (m)
    maxup = float(sys.argv[2]) # maximum updraft velocity (m/s)
    wbase = float(sys.argv[3]) # updraft velocity at cloud base (m/s)
    zmax = float(sys.argv[4]) # height of maximum updraft (km)
    runtime = float(sys.argv[5]) # simulation run time (min)
    level = float(sys.argv[6]) # height of cloud drop size spectrum (km) - for plotting

    # initialise arrays and variables - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # zero arrays (numpy arrays are already zero-initialised, but mirror
    # the explicit zeroing from the C source in case this is re-run)
    # to be filled later
    sumd[:] = 0. # quantity to calculate total reflectivity
    concl[:] = 0. # particle concentration at requested level
    diaml[:] = 0. # particle diameter at requested level

    # create initial particle size distribution (spectrum) - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for jj in range(MAX1):
        radi[jj] = diam[jj] / 2.
        # two exponential fits (only the first is active in the original)
        conc[jj] = 10. ** 5. * math.exp(-39.47 * diam[jj]) # calculating concentration of particles
        # conc[jj] = 10.**3.7 * math.exp(-6.3 * diam[jj])
        # from Jim Dye's paper - see p 35 of book:
        # conc[jj] = 250. * math.exp(-3.00 * diam[jj])

    # define cloud properties: pressure / temperature / depth /vertical velocity profile - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pbase = 672. # pressure mb
    tbase = 8.2 # temperature ºC
    #pbase = 970.
    #tbase = 25.2
    zbase = ptz(pbase)
    print(zbase)
    cdepth = CTOP - zbase / 1000. # depth of cloud (km)

    # vertical velocity change with alt -> slope and intercept of (linear) vertical velocity profile
    slope = (maxup - wbase) / (zmax - zbase / 1000.)
    intcpt = wbase - slope * zbase / 1000.

    # build environment profile - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for ii in range(MAX2):
        # create atmosphere environment at every height
        # then at each level, define atmospheric properties
        alt[ii] = 10. * ii # each level seperated by 10 m

        if alt[ii] >= zbase:
            ps[ii] = ztp(alt[ii]) # pressure
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
            vv[ii] = maxup - (alt[ii] / 1000 - zmax) * 1.75

    # number of simulation time steps - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # number of time points
    numt = int(runtime * 60. / DELTIM)

    # setting up first thermal - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # initial positions of thermals
    thtop1[0] = zinit / 1000. # top of thermal initally at the particles initial height
    cldtop[0] = thtop1[0] # cloud top = thermal top
    thbase1[0] = thtop1[0] - TDEPTH # base of thermal

    # setting up second thermal - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # can another thermal fit below first one? - with a 3km gap??
    if thbase1[0] - TGAP >= 3.: 
        thtop2[0] = thbase1[0] - TGAP # yes - make thermal top
    else:
        thtop2[0] = BAD

    # can thermal base fit below? 
    if thtop2[0] - TDEPTH >= 3.:
        thbase2[0] = thtop2[0] - TDEPTH
    else:
        thbase2[0] = BAD
    downbase[0] = BAD # bottom of cloud top debris

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
    wtht1 = vv[post1]
    wthb1 = vv[posb1]
    wtht2 = vv[post2]
    wthb2 = vv[posb2]

    # simulate thermals through time - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # positions of thermals (in km) with time
    topflag = tht2flag = thb2flag = 0 # flags for if thermal 2 is created 1 = yes it is ??

    # loop through each time step
    for i in range(1, numt):
        # move thermal 1 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        thtop1[i] = thtop1[i - 1] + (wtht1 * DELTIM) / 1000. # new height = old height + velocity × time
        if thtop1[i] >= CTOP:
            thtop1[i] = CTOP
        thbase1[i] = thbase1[i - 1] + (wthb1 * DELTIM) / 1000. # likewise move base

        # create / move thermal 2 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if tht2flag != 1:
            # if thermal top 2 has not been initilised ??
            if thbase1[i] - TGAP >= 3.:
                # if there is enough space for second thermal to exist
                thtop2[i] = thbase1[i] - TGAP
                tht2flag = 1
            else:
                thtop2[i] = BAD
        else:
            # thermal 2 top does exist
            thtop2[i] = thtop2[i - 1] + (wtht2 * DELTIM) / 1000. # move updawds

        if thb2flag != 1:
            # if thermal base 2 has not been initilised ??
            if thtop2[i] - TDEPTH >= 3.:
                thbase2[i] = thtop2[i] - TDEPTH
                thb2flag = 1
            else:
                thbase2[i] = BAD
        else:
            # thermal 2 base does exist
            thbase2[i] = thbase2[i - 1] + (wthb2 * DELTIM) / 1000.

        # deal with cloud top region * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        if thbase1[i] >= CTOP - TDEPTH / 2. and topflag == 0:
            # thermal is approaching maximum cloud top -> stopping growth 
            topflag = 1 # switch on processes
            thtop1[i] = thtop2[i]
            thbase1[i] = thbase2[i]
            thtop2[i] = thbase1[i] - TGAP
            thbase2[i] = thtop2[i] - TDEPTH
            tht2flag = thb2flag = 0
        
        if topflag == 1:
            # thermal has reached cloud top -> thermal stopped growing, now become region of descending material 
            cldtop[i] = cldtop[i - 1] + (WCTDEB * DELTIM) / 1000. # cloud top region grows
            downbase[i] = cldtop[i] - TDEPTH / 2. # base of cloud top region 
            if downbase[i] <= thtop1[i]:
                downbase[i] = thtop1[i]
            if cldtop[i] < thtop1[i]:
                topflag = 0
        else:
            # thermal is not near max cloud top -> still growing
            cldtop[i] = thtop1[i]
            downbase[i] = BAD

        # again find environmental grid location of the thermals * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
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
        # again get vertical velocity at each thermal boundary
        wtht1 = vv[post1]
        wthb1 = vv[posb1]
        wtht2 = vv[post2]
        wthb2 = vv[posb2]
    # end of simulating thermals through time - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # growing all particles in graupel function - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # using pre-defined evolution of thermals
    wi = 0. # updraft m/s
    for jj in range(MAX1):
        j = jj
        graupel(j)

    # calculate radar reflectivity - - - - - - - - - - - - - - - - - - - - - - - - - - - - ?? just per size bin per time step?
    cf = 0.23 / 0.93 # ??
    for jj in range(MAX1):
        # go through all inital particle sizes
        for i in range(num[jj]):
            sumd[i] += (rhop[jj][i] * 1.e-3) ** 2 * conc[jj] * diamp[jj][i] ** 6
            # some calculation which accumulates contributions from all sized particles
    for i in range(numt):
        reft[i] = 10. * math.log10(cf * sumd[i]) if sumd[i] > 0. else BAD
        # from previous calculation - calculate reflectivity ??
        # reflectivity as function of time

    # calculate radar reflectivity at different heights - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # size distribution and reflectivity on a 500 m vertical "grid"
    for k in range(MAX4):
        zgrid[k] = 3. + k / 2. # create vertical grid (start at 3 km) 500m vertical spacing
        for i in range(MAX3):
            sumr[k][i] = 0.
    for k in range(MAX4):
        # ^ loop through each vertical level
        for jj in range(MAX1):
            # ^ loop through each particle size
            for i in range(numt):
                # ^ loop through each time step
                diamh[k][jj][i] = BAD
            for i in range(num[jj]):
                # ^ loop through each time point for plotting (same as number of time steps)
                # i.e. every vertical level, at each particle size at every time step
                if zp[jj][i] < zgrid[k] + .25 and zp[jj][i] >= zgrid[k] - .25:
                    # is this particle within 500m vertical level, yes -> added to sumr and
                    sumr[k][i] += (rhop[jj][i] * 1.e-3) ** 2 * conc[jj] * diamp[jj][i] ** 6 
                    diamh[k][jj][i] = diamp[jj][i]
        for i in range(numt):
            # ^ loop through time steps
            refg[i][k] = 10. * math.log10(cf * sumr[k][i]) if sumr[k][i] > 0. else BAD
            refg[i][k] = refg[i][k] if refg[i][k] > -10. else BAD
            # this is reflectivity as function of time and height

    # particle spectrum at specific height - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # this is for Alan's plotting 
    # does particle pass through the requested level? - more for plotting
    for jj in range(MAX1):
        for i in range(1, num[jj]):
            # check every particle trajectory - has it passed through this specified level?
            # bit of buffer - +/- 0.1km and particle is moving downward and is not in downdraft region
            if ((zp[jj][i] < level + 0.1 and zp[jj][i] > level - 0.1) and (zp[jj][i] < zp[jj][i - 1]) and (xp[jj][i] < TWID - DOWN)):
                concl[jj] = conc[jj] # initial conc of particle size
                diaml[jj] = diamp[jj][i] # grown diameter at requested level

    plot_results()
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
    lab2_ = f"zi = {zinit / 1000.:4.1f} km, TDEPTH = {TDEPTH:3.1f} km"
    lab3_ = f"Lmax = {RATTH:4.2f} Lad, wmax = {maxup:4.1f} m/s"
    lab4_ = f"Lct = {RATCT:4.2f} Lad, Ldeb = {RATDEB:4.2f} Lad"
    lab5_ = f"umax = {UMAX:4.1f} m/s"
    footer = "\n".join([lab1_, lab2_, lab3_, lab4_, lab5_])

    def add_footer(fig):
        fig.text(0.01, 0.01, footer, fontsize=7, va="bottom")

    # 1) altitude of each particle size vs time, with thermal boundaries
    
    with open("particle_altitude_vs_time.csv","w") as f:
             for jj in range(MAX1):
                 n = num[jj]
                 row = ",".join(map(str, zp[jj, :n]))
                 f.write(f"{jj},{row}\n")
    
    # position of thtop1, thbase1, downbase, thtop2, thbase2 in
    # separate files
    with open("thtop1.csv", "w") as f:
        for i in range(numt):
            f.write(f"{i},{thtop1[i]}\n")

    with open("thbase1.csv", "w") as f:
        for i in range(numt):
            f.write(f"{i},{thbase1[i]}\n")

    with open("downbase.csv", "w") as f:
        for i in range(numt):
            f.write(f"{i},{downbase[i]}\n")

    with open("thtop2.csv", "w") as f:
        for i in range(numt):
            f.write(f"{i},{thtop2[i]}\n")

    with open("thbase2.csv", "w") as f:
        for i in range(numt):
            f.write(f"{i},{thbase2[i]}\n")

    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1):
        n = num[jj]
        ax.plot(timep[jj, :n], zp[jj, :n])
    t_axis = timep[0, :numt]
    #ax.plot(t_axis, _mask_bad(cldtop[:numt]), "k--", label="cldtop")
    ax.plot(t_axis, _mask_bad(thtop1[:numt]), "k-", label="thtop1")
    ax.plot(t_axis, _mask_bad(thbase1[:numt]), "b-", label="thbase1")
    ax.plot(t_axis, _mask_bad(downbase[:numt]), "b--", label="downbase")
    #ax.plot(t_axis, _mask_bad(thtop2[:numt]), "g-", label="thtop2")
    ax.plot(t_axis, (thtop2[:numt]), "g-", label="thtop2")
    #ax.plot(t_axis, _mask_bad(thbase2[:numt]), "r-", label="thbase2")
    ax.plot(t_axis, (thbase2[:numt]), "r-", label="thbase2")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("Height (km)")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 9.)
    ax.legend(fontsize=7)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/thermals_particles.png')
    #plt.show()

    # 2) particle size distribution at requested level
    fig, ax = plt.subplots(figsize=(8, 6))
    # also write out file 
    with open("psd.csv","w") as f:
    
        for jj in range(MAX1):
            wid_ = diam[jj] if jj == 0 else diam[jj] - diam[jj - 1]
            spec[jj] = math.log10(concl[jj] / wid_) if concl[jj] > 0. else BAD
            f.write(f"{diaml[jj]},{spec[jj]}\n")
    ax.plot(diaml, _mask_bad(spec), "o-")
    ax.set_xlabel("D (mm)")
    ax.set_ylabel("N (m^-3 mm^-1)")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-1.0, 6.0)
    ax.set_title(f"Spectrum at {level:4.1f} km")
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/psd.png')
    #plt.show()

    # 3) vertical velocity of thermals with height
    fig, ax = plt.subplots(figsize=(8, 6))

    with open("vv_alt.csv", "w") as f:
            for i in range(numt):
                f.write(f"{vv[i]},{alt[i]/1000.}\n")

    ax.plot(vv, alt / 1000.)
    ax.set_xlabel("vert. vel. (m/s)")
    ax.set_ylabel("Height (km)")
    ax.set_xlim(0., 20.)
    ax.set_ylim(3., 9.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/vv_alt.png')
    #plt.show()

    # 4) trajectories of particles
#    fig, ax = plt.subplots(figsize=(8, 6))
#    for jj in range(MAX1):
#        n = num[jj]
#        ax.plot(xp[jj, :n], zp[jj, :n])
#    ax.set_xlabel("x (km)")
#    ax.set_ylabel("Height (km)")
#    ax.set_xlim(0., TWID + 1.)
#    ax.set_ylim(3., 15.)
#    add_footer(fig)
#    fig.tight_layout()
#    plt.savefig('x_z.png')
#    plt.show()

    # 5) diameter of particles vs height
    fig, ax = plt.subplots(figsize=(8, 6))

    with open("diamp_zp.csv","w") as f:
             for jj in range(MAX1):
                 n = num[jj]
                 row1 = ",".join(map(str, diamp[jj, :n]))
                 row2 = ",".join(map(str, zp[jj, :n]))
                 f.write(f"{row1},{row2}\n")

    for jj in range(MAX1):
        n = num[jj]
        ax.plot(diamp[jj, :n], zp[jj, :n])
    ax.set_xlabel("diam (mm)")
    ax.set_ylabel("Height (km)")
    ax.set_xlim(0., 10.)
    ax.set_ylim(3., 9.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/diamp_zp.png')
    #plt.show()

    # 6) terminal velocity of particles vs height
    fig, ax = plt.subplots(figsize=(8, 6))
    for jj in range(MAX1):
        n = num[jj]
        ax.plot(vtp[jj, :n], zp[jj, :n])
    ax.set_xlabel("vt (m/s)")
    ax.set_ylabel("Height (km)")
    ax.set_xlim(0., 9.)
    ax.set_ylim(3., 15.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/vt_z.png')
    #plt.show()

    # 7) diameter of particles vs time
    fig, ax = plt.subplots(figsize=(8, 6))
    with open("diamp_time.csv","w") as f:
        for jj in range(MAX1):
            n = num[jj]
            row1 = ",".join(map(str, timep[jj, :n]))
            row2 = ",".join(map(str, diamp[jj, :n]))
            f.write(f"{row1},{row2}\n")

    for jj in range(MAX1):
        n = num[jj]
        ax.plot(timep[jj, :n], diamp[jj, :n])
    ax.set_xlabel("time (min)")
    ax.set_ylabel("graupel diam (mm)")
    ax.set_xlim(0., 30.)
    ax.set_ylim(0., 10.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/graupel_diam_time.png')
    #plt.show()

    # 8) reflectivity vs time and height (filled contour)
#    fig, ax = plt.subplots(figsize=(8, 6))
#    refg_plot = _mask_bad(refg[:numt, :]).T  # shape (MAX4, numt)
#    im = ax.contourf(np.arange(numt), np.arange(MAX4), refg_plot,
#                      levels=np.linspace(-20., 30., 11), extend="both")
#    fig.colorbar(im, ax=ax, label="dBZ")
#    ax.set_xlabel("time=x*5/60 (mins)")
#    ax.set_ylabel("alt=0.5*y+3 (km)")
#    add_footer(fig)
#    fig.tight_layout()

    # 9) density of particles vs diameter
    fig, ax = plt.subplots(figsize=(8, 6))

    with open("density_diam.csv","w") as f:
        for jj in range(MAX1):
            n = num[jj]
            row1 = ",".join(map(str, diamp[jj, :n]))
            row2 = ",".join(map(str, rhop[jj, :n]))
            f.write(f"{row1},{row2}\n")
    for jj in range(MAX1):
        n = num[jj]
        ax.plot(diamp[jj, :n], rhop[jj, :n] / 1000.)
    ax.set_xlabel("graupel diam (mm)")
    ax.set_ylabel("density (g/cm^3)")
    ax.set_xlim(0., 10.)
    ax.set_ylim(0., 1.)
    add_footer(fig)
    fig.tight_layout()
    plt.savefig('/Users/ezri/code/alan_model/density_diam.png')
    #plt.show()

    # 10) particle size distributions with height (5 - 8.5 km) at a chosen time,
    #     laid out as a 2-row x 4-column grid of small panels
#    c1_ = 1. / 7.
#    for jj in range(MAX1):
#        wid_ = diam[jj] if jj == 0 else diam[jj] - diam[jj - 1]
#        spec[jj] = conc[jj] / wid_

#    fig, axes_grid = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
#    for kr in range(2):
#        for kc in range(4):
#            ax = axes_grid[kr][kc]
#            nw = kr * 4 + kc
#            k = 4 + nw + 1  # matches C's `k = 4 + nw` with nw starting at 1
#            diamh_slice = diamh[k, :, numt - 5]
#            valid = diamh_slice < 99999.
#            diamx_ = np.where(valid, diamh_slice / 6., np.nan)
#            specy_ = np.where(valid, np.log10(np.where(spec > 0, spec, np.nan)) / 7. + c1_, np.nan)
#            ax.plot(diamx_, specy_, "o-", markersize=3)
#            ax.set_title(f"{zgrid[k]:4.1f} km", fontsize=8)
#    fig.suptitle("x-axis: 0 - 5 mm; y-axis: 0.1 - 10^6 m^-3 mm^-1")
#    add_footer(fig)
#    fig.tight_layout()

#    plt.show()
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 


if __name__ == "__main__":
    main()

print('ran')