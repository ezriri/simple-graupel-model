## between lines 649 - 705 in v4 of model
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
    #cd = drag(0.15e-3, rhog, rhoa) ## 
    cd = drag(0.15e-3, 100., rhoa) ## 
    #vt300 = 8. * 0.15 * 1.0e-3 * 220. * GRAV / (3. * rhoa * cd) # terminal velocity for 300um particle
    vt300 = 8. * 0.15 * 1.0e-3 * 100. * GRAV / (3. * rhoa * cd) # terminal velocity for 300um particle
    vt300 = vt300 ** 0.5 # needs to be square rooted
    
    
    #dmr300 = PI * 0.15e-3 ** 2 * vt300 * lwc * namelist['DELTIM'] # riming mass change for 300um particle
    dmr300 = PI * 0.15e-3 ** 2 * vt300 * lwc * namelist['DELTIM'] # riming mass change for 300um particle
    #dr300 = dmr300 / (4. * PI * 0.15e-3 ** 2 * 220.) # radius change for 300um particle
    dr300 = dmr300 / (4. * PI * 0.15e-3 ** 2 * 100.) # radius change for 300um particle
    #dr300 = 2.0e-6
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
        vt = vt_small(dmm, tempc)
    else:
        # for particles between 200-300 um - smooth transition between
        vt200 = vt_small(0.2, tempc) ## terminal velocity for 200um particle
        vt = vt200 + (vt300 - vt200) * (dmm - 0.2) / 0.1 # type: ignore
else:
    # graupel particle - bigger than 300um
    vt = 8. * rad * rhog * GRAV / (3. * rhoa * cd)
    vt = vt ** 0.5