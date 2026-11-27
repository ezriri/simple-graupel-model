# in graupel()
dmm = (rad * 2)* 1000 # in mm
if dmm <= 0.3:
    # diam is in mm
    conc_n = 10. ** 4. * math.exp(-39.47 * dmm) # calculating concentration of particles
else:
    conc_n = 10. ** 0.7 * math.exp(-1.3 * dmm)





### in main()
for jj in range(MAX1):
    #radi[jj] = diam[jj] / 2. 
    # two exponential fits - spectrum (10/8/94)
    if diam[jj] <= 0.3:
        # diam is in mm
        conc[jj] = 10. ** 4. * math.exp(-39.47 * diam[jj]) # calculating concentration of particles
    else:
        conc[jj] = 10. ** 0.7 * math.exp(-1.3 * diam[jj])
    #totnum += conc[jj] # total intitial particle concentration (m-3)