"""
Python translation of icemult_ezri.c

icemult -- A wee model to predict the development of ice by the Hallett-
           Mossop (and other) processes.
           Based on the program precip.c (graupel_thermals_ezri.c) with the
           incorporation of ice multiplication processes.

  Design:  a) updraught assigned based on combination of radar
              and aircraft observations; vertical size constant
           b) terminal velocity of particles based on particle size and
              observations of particle types in KA87 project
           c) liquid water content profile based on aircraft observations
           d) growth by vapour diffusion and riming according to graupel model
           e) ice crystal growth from Ryan et al. according to T.
           f) initial spectrum from limited aircraft observations in KA87
              project and various sailplane measurements; start at cloud
              top at various temperatures
           g) position of particles relative to ground and to the top of
              the thermal calculated
           h) output the particle spectrum and particle positions in the
              vertical
           i) lwc depleted above 8 km due to conversion to ice

  Ice multiplication design:
           a) the Hallett-Mossop zone is divided up into 1 deg C sub-zones
           b) splinters are produced in each of these sub-zones at the rate
              given by the formula Nx = rime * Ng * H; rime being the amount
              of rime gathered in the time interval, and H being the rate
              of splinter production per mg of rime collected
              (Other approaches will be tried later - 15/8/94.)

TRANSLATION NOTES
-----------------
1. `trev()` (adiabatic temperature/LWC vs pressure) is *declared* and
   called in the C source but its body lives in another .c file that was
   not supplied (same situation as in the previous file you gave me). A
   stub is provided below -- replace it with the real implementation.

2. `vapour()` and `gr()` are fully defined in this file and are translated
   faithfully.

3. `scale()` is called once (to pick y-axis limits for the "f vs time"
   plot) but is not defined in this file -- it's presumably another
   library routine from the same custom graphics package as `line()`,
   `axes()`, etc. A simple equivalent (`scale_axis()`) is provided below;
   it picks integer-rounded min/max bounds from the data, which is
   consistent with how its result (`nticks = maxy - miny + 1`) is used.

4. `BAD` is a sentinel for "missing/invalid" data, defined in the
   unsupplied "cdfhdr.h" header. As in the previous file, it is assumed
   to be 999999.0 here.

5. As with the previous file, plotting is done with a custom in-house
   graphics library (`gopen`, `gclear`, `window`, `axes`, `line`, `label`,
   `mark`, `gpause`, `gclose`) that isn't available. The numerical content
   is translated exactly; the plotting blocks are re-created as equivalent
   matplotlib figures, in the same order as the original, with a short
   comment noting which original block each one replaces.

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

# ---------------------------------------------------------------------------
# #define constants
# ---------------------------------------------------------------------------
MAX1 = 10          # number of particle size categories
MAX2 = 1200        # number of altitude values
MAX3 = 2000        # number of time steps
MAX5 = 2000        # number of H-M splinter groups
MAX6 = 2010        # number of original size categories + H-M splinter groups
MAX8 = 500         # number of points for plotting

# Standard run - RATTH = 0.4, RATCT = 0.2, RATDEB = 0.1, RATDOWN = 0.1
#                TDEPTH = 2 km, TGAP = 1.2 km, no downdraughts (umax = 0),
#                wmax = 5 m/s, wmin = 2 m/s, max at 7 km, zinit = 6.5

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
ZCT = TDEPTH / 20.  # depth of overall cloud top region
ZCT1 = ZCT / 2.     # depth of cloud top region 1
ZCT2 = ZCT / 2.     # depth of cloud top region 2
GENCTOP = CTOP - 0.5  # height of general cloud top

HMRINIT = 10.0e-6   # initial r of HM splinter
HMFACT = 5.0e7      # number per kg of rime

PFACT = 5           # reduction in pts for plotting

P0 = 1013.
T0 = 288.
GAMMA = 0.0065
RHOI = 920
EPS = 0.622
DELTIM = 5.
RD = 287.05
RV = 461.51
TK = 2.32e-2
D = 2.11e-5
TTR = 273.16
RW = 461.51
LS = 2.837e6
LF = 3.12e5
LV = 2.5e6
NMAX = 500.e6
DIS = 0.25
GRAV = 9.81
ETA = 1.67e-5
CPW = 4.27e3
PI = 3.1415

# see note 4 above -- best-effort guess at the value of BAD
BAD = 999999.0

# Usual spectrum
diam = np.array([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9])

# declared in the C source but not referenced anywhere in main() -- kept
# here for fidelity, unused
diamf = np.array([.05, .15, .25, .35, .5, .75, 1.1, 1.55, 2.1, 2.75,
                   3.45, 4.15, 4.85, 5.55, 6.25])
wid_arr = np.array([.1, .1, .1, .1, .2, .3, .4, .5, .6, .7,
                     .7, .7, .7, .7, .7])

# alternative spectra (commented out in the original):
# diam = np.array([.05, .15, .25, .35, .5, .75, 1.1, 1.55, 2.1, 2.75])
# diam = np.array([.05, .1, .15, .2, .25, .3, .35, .4, .45, .5])
# diam = np.array([.04, .08, .12, .16, .2, .24, .28, .32, .36, .4])
# diam = np.array([.002, .004, .006, .008, .01, .012, .014, .016, .018, .02])

# ---------------------------------------------------------------------------
# Module-level globals (mirrors C file/global-scope [static] variables
# shared between main(), graupel(), drag(), tsurf())
# ---------------------------------------------------------------------------
lw = np.zeros(MAX2)
db = np.zeros(MAX2)
vv = np.zeros(MAX2)
alt = np.zeros(MAX2)
ps = np.zeros(MAX2)
tr = np.zeros(MAX2)
conc = np.zeros(MAX1)
radi = np.zeros(MAX1)
spec = np.zeros(MAX1)
speci = np.zeros(MAX1)
tothm = np.zeros(MAX3)
cumtot = np.zeros(MAX3)
thtop1 = np.zeros(MAX3)
thbase1 = np.zeros(MAX3)
thtop2 = np.zeros(MAX3)
thbase2 = np.zeros(MAX3)
cldtop = np.zeros(MAX3)
downbase = np.zeros(MAX3)
xx = np.zeros(MAX3)
yy = np.zeros(MAX3)
diamp = np.zeros((MAX6, MAX8))
vtp = np.zeros((MAX6, MAX8))
deltsp = np.zeros((MAX6, MAX8))
effp = np.zeros((MAX6, MAX8))
timep = np.zeros((MAX6, MAX8))
rhop = np.zeros((MAX6, MAX8))
zp = np.zeros((MAX6, MAX8))
xp = np.zeros((MAX6, MAX8))
stime = np.zeros(MAX5)
xhm = np.zeros(MAX5)
zhm = np.zeros(MAX5)
nhm = np.zeros(MAX5)

wtht1 = wthb1 = wtht2 = wthb2 = 0.0
rad = 0.0
wi = 0.0
lwc = 0.0
rhoa = 0.0
zinit = 0.0
cf = 0.0
pbase = tbase = 0.0
zbase = cdepth = 0.0
time_ = 0.0
at = alwc = 0.0
c1 = 0.0
xinit = yinit = 0.0
x1 = x2 = wy1 = wy2 = 0.0
maxup = 0.0
wbase = zmax = 0.0
umax = 0.0
slope = intcpt = 0.0
runtime = 0.0
es = esi = 0.0
nre = nsh = nnu = 0.0
xmax = 0.0
totnum = 0.0
level = 0.0
level1 = level2 = level3 = 0.0
hmrime = 0.0
zm8 = zm3 = 0.0

numt = 0
num = np.zeros(MAX6, dtype=int)
hmi = np.zeros(MAX5, dtype=int)
hm = 0
hmj = 0
j = 0
tt = pp = 0
nxtick = 0
topflag = 0
thb1flag = 0
tht2flag = thb2flag = 0
post1 = posb1 = post2 = posb2 = 0
splinter = 0
atlevel1 = atlevel2 = atlevel3 = 0
totpts = 0
hmflag = 0

pname = ""
lab1 = lab2 = lab3 = lab4 = lab5 = lab6 = lab7 = lab8 = lab9 = lab10 = ""
lab11 = lab12 = lab13 = lab14 = lab15 = lab16 = lab17 = lab18 = lab19 = ""
lab20 = lab21 = ""


def vapour(t):
    """Goff-Gratch formula for water saturation vapour pressure."""
    arg1 = 11.344 * (1. - t / 373.16)
    arg2 = 3.49149 * (1. - 373.16 / t)
    e = (-7.90298 * (373.16 / t - 1.) + 5.02808 * math.log10(373.16 / t)
         - 1.3816e-7 * (10. ** arg1 - 1.)
         + 8.1328e-3 * (10. ** arg2 - 1.))
    v = 1013.246 * 10. ** e
    return v


def gr(tempr):
    """dr/dt values from Ryan et al's lab expts for diffusional growth of ice."""
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
    if -21.5 <= tempr < -20.5:
        return 0.4
    # tempr < -21.5
    return 0.3


def trev(pbase_, tbase_, p):
    """
    Adiabatic temperature and liquid water content for pressure p, given
    cloud-base pressure pbase_ and cloud-base temperature tbase_.

    Placeholder -- see note 1 at the top of this file.

    Returns
    -------
    (at, alwc) : tuple
        Adiabatic temperature (deg C) and adiabatic liquid water content.
    """
    raise NotImplementedError(
        "trev() was declared and called in the original C source but its "
        "implementation was not included in the supplied file. Please "
        "provide the real pseudo-adiabatic ascent calculation."
    )


def ztp(zz):
    """Pressure (mb) for a given altitude zz (m), constant lapse rate."""
    p = P0 * ((T0 - GAMMA * zz) / T0) ** (GRAV / (RD * GAMMA))
    return p


def ptz(p):
    """
    Altitude (m) for a given pressure p (mb).
    Uses a constant lapse rate of 6.5 C/km (see Hess pp 82-83).
    """
    ex = (RD * GAMMA) / GRAV
    ptzr = T0 * (1.0 - (p / P0) ** ex) / GAMMA
    return ptzr


def scale_axis(values, step=1.0):
    """
    Best-effort equivalent of the (unsupplied) library routine `scale()`.
    Picks axis min/max, rounded outward to the nearest `step`, from a set
    of (possibly BAD-flagged) data values. Used only for choosing y-axis
    bounds on the "f vs time" plot, where the caller then does
    `nticks = maxy - miny + 1`, i.e. it expects integer-ish bounds.
    """
    valid = values[values < BAD / 10.]
    if valid.size == 0:
        return 0.0, 1.0
    miny = math.floor(valid.min() / step) * step
    maxy = math.ceil(valid.max() / step) * step
    if maxy <= miny:
        maxy = miny + step
    return maxy, miny


def drag(r, rhog, rhoa_):
    """
    Calculate the Reynolds number (stored in the global `nre`) and the drag
    coefficient for a particle of radius r, density rhog, in air of density
    rhoa_.
    """
    global nre

    xd = 32. * rhog * rhoa_ * GRAV * r ** 3 / (3. * ETA * ETA)

    if xd < 1.09e4:
        a = 0.0688
        b = 0.769
    if 1.09e4 <= xd < 6.58e5:
        a = 0.347
        b = 0.595
    if xd >= 6.58e5:
        a = 3.6184
        b = 0.420

    nre = a * xd ** b
    cd = 8. * nre ** (-0.27)
    return cd


def tsurf(kap, tsk):
    """
    Calculate the surface temperature of the growing graupel particle,
    using the relationships derived in Pflaum and Pruppacher.
    """
    global es

    cpv = 1.87e3
    ci = 2.031e3
    mu = 1.667e-5
    to = 273.15
    ik = 0

    es = 100. * vapour(tsk)
    rve = es / (RV * tsk)

    nsc = mu / (rhoa * D)
    npr = mu * cpv / TK

    tm1 = kap * lwc / (rad * PI)

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


def graupel():
    """
    Grow a single particle -- either one of the MAX1 initial cloud
    particles (`splinter == 0`), or one Hallett-Mossop splinter group
    (`splinter == 1`, particle index `j = MAX1 + hmj`).

    Splinters produced by riming in the H-M zones (-3 to -8 C, in five
    1-degree sub-zones) are recorded in the module-level `xhm`/`zhm`/
    `nhm`/`stime`/`hmi` arrays and `hm` is incremented for each new group;
    the caller in main() re-checks `hm` after every call so that newly
    spawned splinter groups are themselves grown.
    """
    global j, hm, hmrime, rad, wi, lwc, rhoa, es, esi, nre, nsh, nnu
    global num

    z = x = pres = temp = tempc = dbar = 0.0
    zold = xold = 0.0
    zkm = 0.0
    zdown = 0.0
    hw = 0.0
    lwc7 = 0.0

    if splinter == 1:
        j = MAX1 + hmj

    tflag = 0
    one = 0
    pp_ = 0
    tt_ = 0
    hw = 0.
    wi = 0.2
    indx = 0
    downflag = 0
    topflag_local = 0
    graup = 0
    zflag = 0
    hmrime = 0.
    hmzone1 = hmzone2 = hmzone3 = hmzone4 = hmzone5 = 0

    vtsave = vt200 = vt300 = None
    ddsave = dd200 = None
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

    if splinter == 1:
        istart = hmi[hmj]
    else:
        istart = 0

    pos = 0

    for i in range(istart, numt):
        # check environmental conditions
        # first round - get initial conditions
        if one == 0:
            if splinter > 0:
                z = 1000 * zhm[hmj]
                x = xhm[hmj]
                time = stime[hmj]
            else:
                z = zinit
                x = 0.
                time = 0.
            zold = z
            xold = x

        for ii in range(MAX2):
            if ii > 0:
                if z < alt[ii] and z >= alt[ii - 1]:
                    pos = ii
        pres = ps[pos] * 100.
        zkm = z / 1000.

        hw = 0
        if downflag == 0:
            # particle in thermal ?
            if ((zkm <= thtop1[i] - ZCT and zkm >= thbase1[i]) or
                    (zkm <= thtop2[i] - ZCT and zkm >= thbase2[i])):
                temp = tr[pos] - 1.
                lwc = RATTH * lw[pos]
                dbar = db[pos]
                wi = vv[pos]

            # in cloud top region (top of thermal 1)
            if zkm <= thtop1[i] and zkm >= thtop1[i] - ZCT:
                temp = tr[pos] - 1.
                lwc = RATCT * lw[pos]
                dbar = db[pos]
                hw = umax
                # work out smooth vertical velocity in cloud top regions
                wi = vv[pos] - vv[pos] * (zkm - (thtop1[i] - ZCT)) / (2. * ZCT)

            # out of thermal, into debris under thermal (over thermal if at start)
            if ((zkm > thtop1[i] or zkm < thbase1[i]) and
                    (zkm > thtop2[i] or zkm < thbase2[i]) and
                    (zkm < downbase[i])) or (zkm > thtop1[i] and zkm > thtop2[i]):
                temp = tr[pos] - 1.
                lwc = RATDEB * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = 0.0

            # in descending thermal remnants
            if zkm > thtop1[i] and zkm <= cldtop[i] and zkm >= downbase[i]:
                temp = tr[pos] - 2.
                lwc = RATDEB * lw[pos]
                dbar = db[pos]
                hw = 0.
                if cldtop[i] > GENCTOP:
                    wi = WCTDEB
                else:
                    wi = 0.

            # horizontal position in km
            if x >= TWID:
                temp = tr[pos] - 2.
                lwc = RATDOWN * lw[pos]
                dbar = db[pos]
                hw = 0.
                wi = -5.
                downflag = 1
                zdown = zkm
        else:
            temp = tr[pos] - 2.
            lwc = RATDOWN * lw[pos]
            dbar = db[pos]
            hw = 0.
            wi = -5.
            # make downdraughts no deeper than DDEPTH km
            if abs(zdown - zkm) >= DDEPTH:
                downflag = 0
                xold = 0
                x = 0.
                zdown = 0.

        # overshoot cloud top?
        if zkm >= CTOP or zkm > cldtop[i]:
            zkm = cldtop[i]

        # deplete the lwc if above 8 km (due to conversion to ice)
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
        # kinematic viscosity
        nu = ETA / rhoa
        tempc = temp - 273.15

        if one == 0:
            # initial radius in m and density kg m^-3
            if splinter < 1:
                rad = radi[j] * 1.0e-3
            else:
                rad = HMRINIT
            rhog = 900.
            # initial mass
            mass = 4. * PI * rad ** 3 * rhog / 3.

            # calculate vt and delmr for a 300 um diameter particle
            # Assume density of 100 g/cm^3 just now as a quick fix - 7/7/94
            # and use simple formula for kapb
            cd = drag(0.15e-3, 100., rhoa)
            if rad * 2.0e3 < 0.2:
                vt300 = 0.6 * 0.3
                # force dr300 to slightly larger than largest value of dr
                # by diffusion
                dr300 = 2.0e-6

        # drag coefficient
        cd = drag(rad, rhog, rhoa)

        # terminal velocity is taken as a function of temperature according
        # to the graph of Fukuta et al. 1982, Chicago conference.
        # If initial diam is greater than 200 um, go straight to vt for graupel
        dmm = rad * 2.0e3
        if dmm <= 0.30:
            if dmm <= 0.2:
                if (-6.5 < tempc <= -4.) or (-16 < tempc <= -14.):
                    vt = 100. * (0.395 * dmm - 0.177 * dmm ** 2
                                 + 0.073 * dmm ** 3 - 0.0153 * dmm ** 4)
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
                    vt200 = vtsave
                vt = vt200 + (vt300 - vt200) * (dmm - 0.2) / 0.1
        else:
            vt = 8. * rad * rhog * GRAV / (3. * rhoa * cd)
            vt = vt ** 0.5

        # On first round, use temp + 2.4 for surface temp
        if one == 0:
            one = 1
            ts = temp + 2.4

        # dm for diffusion
        if dmm <= 0.30:
            if dmm <= 0.20:
                ts = temp
                delrd = 1.0e-6 * gr(tempc)
                ddsave = delrd
            else:
                if graup == 0:
                    dd200 = ddsave
                    graup = 1
                drtrans = dd200 + (dr300 - dd200) * (dmm - 0.2) / 0.1

        esi = 100. * vapour(TTR) * (math.exp((ts - TTR) * LS / (RW * ts * TTR)))
        rvs = esi / (RV * ts)
        es = 100. * vapour(temp)
        re = es / (RV * temp)
        # Sherwood number as per Mason (1971)
        nsh = 0.58 * nre ** 0.5
        # Nusselt number as per Mason (1971)
        nnu = nsh
        # use lab values for vapour growth
        delrd = 1.0e-6 * gr(tempc)

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
                for k in range(24):
                    xi = (26. - (k + 1.)) * 1.0e-6
                    hxa = (xi * 1.0e6 - dbar / 2.) / stdv
                    hxa = hxa ** 2 / 2.
                    yi = NMAX * math.exp(-hxa)
                    sumy = yi + sumy
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
                    sumkap = sumkap + kappa * yi

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
                    vimp = vimp * vt

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
                    sumrh = rhor * yi + sumrh

                rhor = sumrh * 1.0e3 / sumy
                kapb = sumkap / sumy
            else:
                rhor = 900.
                kapb = PI * rad ** 2 * vt

            # Tsurf
            if tflag == 0:
                # only go to sub tsurf if kapb is significant
                if kapb >= 0.2e-10:
                    ts = tsurf(kapb, temp)
                else:
                    ts = temp
                if ts >= 273.:
                    tflag = 1
                    ts = 273.15

            # increase in mass due to riming
            delmr = kapb * lwc * DELTIM

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
                frac = 1.
                delmr = frac * delmr

            # end of riming loop for D > 300 um

        # dR and Rnew
        if rad * 2.e3 <= 0.2:
            delrr = 0.
        if rad * 2.e3 <= 0.3 and rad * 2.e3 > 0.2:
            delrr = drtrans
        if rad * 2.e3 > 0.3:
            delrr = delmr / (4. * PI * rad * rad * rhor)
        delr = delrr + delrd
        rad = rad + delr

        # new bulk density of the graupel particle
        mass = mass + delmr
        rhog = mass * 3. / (4. * PI * rad ** 3)
        if rhog < 100.:
            rhog = 100.
        if rhog > 900.:
            rhog = 900.

        # particle in one of the Hallett-Mossop zones? Store x,z position,
        # start splinter with a diameter of 20 um, give splinter a number,
        # and calculate the concentration of splinters produced. Note, only
        # one calculation per zone; zhm is calculated to be at the centre of
        # the zone. Assume temp is close to the temp at the top of the
        # sub-zone and calculate the mid-point using the current altitude
        # and 6.7 deg -> 1 km. The value of i (which gives the time) is
        # stored for each new splinter group.
        if hmflag == 1:
            if rad * 2e3 >= 0.3 and tempc >= -9.:
                if -8. <= tempc < -7.:
                    hmrime += delmr
                    hmzone1 = 1
                if hmzone1 == 1 and (tempc < -8. or tempc >= -7.):
                    nx = conc[j] * HMFACT * hmrime if splinter == 0 else nhm[hmj] * HMFACT * hmrime
                    nhm[hm] = nx
                    tothm[i] += nx
                    xhm[hm] = x
                    zhm[hm] = zkm - 0.5 / 6.7
                    stime[hm] = time
                    hmi[hm] = i
                    hm += 1
                    hmrime = 0.
                    hmzone1 = 0

                if -7. <= tempc < -6.:
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

        # for plotting (in main program) -- only every PFACT-th step is kept
        pp_ += 1
        if pp_ == PFACT:
            diamp[j][tt_] = rad * 2.e3
            rhop[j][tt_] = rhog
            timep[j][tt_] = time
            zp[j][tt_] = zkm
            xp[j][tt_] = x
            vtp[j][tt_] = vt
            effp[j][tt_] = kapb / (PI * rad ** 2 * vt) if kapb is not None else BAD
            deltsp[j][tt_] = ts - temp
            tt_ += 1
            pp_ = 0

        time = time + DELTIM / 60.

        # particle velocity: vt is positive down
        wpcle = wi - vt

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
    # NOTE: unlike graupel_thermals_ezri.c, this file's growth loop only
    # sets num[j] inside the break above -- there is no unconditional
    # assignment after the loop. If the loop runs to completion without
    # tripping the break condition, num[j] is left at whatever it was
    # before this call (0 by default, since `num` is a zero-initialised
    # global/static array in the C source). This is translated faithfully.

    return 0


def main():
    global zinit, maxup, wbase, zmax, runtime, umax
    global pbase, tbase, zbase, cdepth, slope, intcpt
    global numt, wi, hm, hmj, hmrime, splinter, hmflag, pname, j
    global thtop1, thbase1, thtop2, thbase2, cldtop, downbase
    global wtht1, wthb1, wtht2, wthb2
    global topflag, thb1flag, tht2flag, thb2flag, post1, posb1, post2, posb2
    global totnum, zm8, zm3
    global lab1, lab2, lab3, lab4, lab5, lab6, lab7, lab8, lab9, lab10
    global lab11, lab12, lab13, lab14, lab15, lab16, lab17, lab18, lab19
    global lab20, lab21

    pname = sys.argv[0]

    # -- argument parsing (mirrors `getopt(argc, argv, "h")`) --
    args = sys.argv[1:]
    hmflag = 0
    if args and args[0] == "-h":
        hmflag = 1
        args = args[1:]
    if len(args) != 5:
        print(f"Usage: {pname} [-h] zi (km) wmax wbase (m/s)  z@wmax (km), "
              f"run_time (min) ", file=sys.stderr)
        sys.exit(0)

    zinit = 1000. * float(args[0])
    maxup = float(args[1])
    wbase = float(args[2])
    zmax = float(args[3])
    runtime = float(args[4])

    # umax from continuity
    umax = UFACT * maxup / 2.

    # zero arrays
    tothm[:] = 0.

    # initial spectrum of primary particles
    # the total concentration is 1.6 /L
    totnum = 0
    for jj in range(MAX1):
        radi[jj] = diam[jj] / 2.
        # two exponential fits - spectrum (10/8/94)
        if diam[jj] <= 0.3:
            conc[jj] = 10. ** 4. * math.exp(-39.47 * diam[jj])
        else:
            conc[jj] = 10. ** 0.7 * math.exp(-1.3 * diam[jj])
        totnum += conc[jj]

    # cloud base pressure and temperature
    pbase = PBASE
    tbase = TBASE
    zbase = ptz(pbase)
    print(f"cloud base: p = {pbase:5.1f} mb, T = {tbase:4.1f} C, "
          f"z = {zbase / 1000.:3.1f} km", file=sys.stderr)
    cdepth = CTOP - zbase / 1000.

    # slope and intercept of vertical velocity profile
    slope = (maxup - wbase) / (zmax - zbase / 1000.)
    intcpt = wbase - slope * zbase / 1000.

    # environment profiles
    for ii in range(MAX2):
        ps[ii] = pbase - ii / 2.
        alt[ii] = ptz(ps[ii])
        at_, alwc_ = trev(pbase, tbase, ps[ii])
        lw[ii] = alwc_ * 1.0e-3
        tr[ii] = at_ + 273.15
        db[ii] = DBAR + ii / 125.

        if alt[ii] < zbase:
            vv[ii] = 1.
        elif alt[ii] / 1000. <= zmax:
            vv[ii] = slope * alt[ii] / 1000. + intcpt
        else:
            vv[ii] = (maxup / ((CTOP + 2.) - zmax)) * ((CTOP + 2.) - alt[ii] / 1000.0)

    # number of time points
    numt = 1 + int(runtime * 60. / DELTIM)
    print(f"numt = {numt}", file=sys.stderr)

    # initial positions of thermals
    thtop1[0] = zinit / 1000. + 0.0
    if thtop1[0] > CTOP:
        thtop1[0] = CTOP
    cldtop[0] = thtop1[0]
    thbase1[0] = thtop1[0] - TDEPTH
    if thbase1[0] < 0.:
        thbase1[0] = 0.
    thtop2[0] = thbase1[0] - TGAP
    if thtop2[0] < 0.:
        thtop2[0] = 0.
    thbase2[0] = thtop2[0] - TDEPTH
    if thbase2[0] < 0.:
        thbase2[0] = 0.
    downbase[0] = BAD

    # ascent rate of thermals (assumed to be half the updraught speed)
    post1 = posb1 = post2 = posb2 = 0
    for ii in range(MAX2):
        if ii > 0:
            if thtop1[0] < alt[ii] / 1000. and thtop1[0] >= alt[ii - 1] / 1000.:
                post1 = ii
            if thbase1[0] < alt[ii] / 1000. and thbase1[0] >= alt[ii - 1] / 1000.:
                posb1 = ii
            if thtop2[0] < alt[ii] / 1000. and thtop2[0] >= alt[ii - 1] / 1000.:
                post2 = ii
            if thbase2[0] < alt[ii] / 1000. and thbase2[0] >= alt[ii - 1] / 1000.:
                posb2 = ii
    wtht1 = vv[post1] / 2.
    wthb1 = vv[posb1] / 2.
    wtht2 = vv[post2] / 2.
    wthb2 = vv[posb2] / 2.

    # positions of thermals (in km) with time
    topflag = thb1flag = tht2flag = thb2flag = 0
    for i in range(1, numt):
        thtop1[i] = thtop1[i - 1] + (wtht1 * DELTIM) / 1000.
        if thtop1[i] >= CTOP:
            thtop1[i] = CTOP

        thbase1[i] = thtop1[i] - TDEPTH
        if thbase1[i] > 0.:
            thbase1[i] = thbase1[i - 1] + (wthb1 * DELTIM) / 1000.

        if tht2flag != 1:
            thtop2[i] = thbase1[i] - TGAP
            if thtop2[i] > 0.:
                tht2flag = 1
            if thtop2[i] < 0.:
                thtop2[i] = 0.
        else:
            thtop2[i] = thtop2[i - 1] + (wtht2 * DELTIM) / 1000.

        thbase2[i] = thtop2[i] - TDEPTH
        if thbase2[i] > 0.:
            thbase2[i] = thbase2[i - 1] + (wthb2 * DELTIM) / 1000.

        if thtop1[i] >= CTOP and thbase1[i] >= (CTOP - TDEPTH / 4.) and topflag == 0:
            topflag = 1
            thtop1[i] = thtop2[i]
            thbase1[i] = thbase2[i]
            thtop2[i] = thbase1[i] - TGAP
            thbase2[i] = thtop2[i] - TDEPTH
            thb1flag = tht2flag = thb2flag = 0
            cldtop[i - 1] = CTOP
        if topflag == 1:
            # make cloud top descend to the general cloud top region
            cldtop[i] = cldtop[i - 1] + (WCTDEB * DELTIM) / 1000.
            if cldtop[i] <= GENCTOP:
                cldtop[i] = GENCTOP
            downbase[i] = cldtop[i] - TDEPTH / 4.
            if downbase[i] <= thtop1[i]:
                downbase[i] = thtop1[i]
            if cldtop[i] < thtop1[i]:
                topflag = 0
        else:
            cldtop[i] = thtop1[i]
            downbase[i] = BAD

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

        wtht1 = vv[post1] / 2.
        wthb1 = vv[posb1] / 2.
        wtht2 = vv[post2] / 2.
        wthb2 = vv[posb2] / 2.

    # growth of all particles handled in graupel()
    # Note particles produced in the H-M zones in graupel() are splinters.
    # Their growth is dealt with after the initial set of trajectories, by
    # number.
    wi = 0.
    hm = 0
    hmj = 0
    hmrime = 0.
    splinter = 0
    for jj in range(MAX1):
        j = jj
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


def _mask_bad(arr):
    """Helper: replace BAD-sentinel / non-positive values with NaN."""
    arr = np.asarray(arr, dtype=float)
    return np.where((arr >= 99999.) | np.isnan(arr), np.nan, arr)


def plot_results():
    """
    Re-creates each of the original plotting blocks using matplotlib,
    since the original custom graphics library is not available (see note
    5 at the top of this file). Each figure below corresponds to one
    "window(...)/gpause()/gclear()" block in the C source, in the same
    order, and plots the same data.
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

    plt.show()


if __name__ == "__main__":
    main()
