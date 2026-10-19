## the code for dealing where the particles are wrt the thermals

# 2. where is the particle with respect to the thermal? * * * * * * * * * * * * * * * * * * * * * * * * * *
# this modifies the environmental conditions -> alter particle growth and movement
hw = 0 # horizontal wind (m/s) # not persistent between time steps
if downflag == 0: # particle is not in downdraft
    ## *** particle is in thermal *** - condition 1 
    if ((zkm <= thtop1[i] - namelist['ZCT'] and zkm >= thbase1[i]) or (zkm <= thtop2[i] - namelist['ZCT'] and zkm >= thbase2[i])):
        # particle is still in thermal (can be thermal 1 or thermal 2)
        temp = tr[pos] - 1. # thermal envi is 1ºC cooler than the environment
        lwc = namelist['RATTH'] * lw[pos]
        dbar = db[pos] # drop size
        wi = vv[pos] # vertical velocity - based on the vertical velocity profile of the thermal

    ## *** particle is in ZCT zone (only region where it can move horizontally) *** condition 2
    # in very top of the cloud -> horizontal wind kicks in -> move particle into downdraft
    if zkm <= thtop1[i] and zkm >= thtop1[i] - namelist['ZCT']:
        temp = tr[pos] - 1.
        lwc = namelist['RATCT'] * lw[pos]
        dbar = db[pos]
        hw = umax
        # work out smooth vertical velocity in cloud top regions
        wi = vv[pos] - vv[pos] * (zkm - (thtop1[i] - namelist['ZCT'])) / (2. * namelist['ZCT']) # vertical velocity change with height

    # out of thermal -> in debris under thermal // above thermal (if at start) *** condition 3 - debris 
    if ((zkm > thtop1[i] or zkm < thbase1[i]) and (zkm > thtop2[i] or zkm < thbase2[i]) and (zkm < downbase[i])) or (zkm > thtop1[i] and zkm > thtop2[i]):
        temp = tr[pos] - 1.
        lwc = namelist['RATDEB'] * lw[pos]
        dbar = db[pos]
        hw = 0.
        wi = 0.0 # this applies if outside cloud

    # above thermal 1, within cloud top, in descending thermal remnants *** condition 4 -- a and b
    if zkm > thtop1[i] and zkm <= cldtop[i] and zkm >= downbase[i]:
        temp = tr[pos] - 2.
        lwc = namelist['RATDEB'] * lw[pos]
        dbar = db[pos]
        hw = 0.
        if cldtop[i] > namelist['GENCTOP']:
            #*** condition 4a
            # cloud top above general cloud top -> prescribed verical velocity
            wi = namelist['WCTDEB']
        else:
            ## condition 4b
            wi = 0.

    # horizontal position in km **** condition 5
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
    # particle is in downdraft **** condition 6
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
    zkm = cldtop[i] # adjust so don't overshoot top 

# deplete the lwc if above 7 km (due to conversion to ice)
if zkm >= 7:
    if zflag == 0:
        lwc7 = lwc
        zflag = 1
    if zflag > 0:
        lwc = lwc7 - (zkm - 7.) * lwc7 / (10. - 7.)
    if zkm >= 10.:
        lwc = 0.