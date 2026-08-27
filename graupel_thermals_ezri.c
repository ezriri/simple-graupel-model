/* precip   -- A wee model to predict the development of precipitation.   
 *             
 *             Design:  a) updraught assigned based on combination of radar
 *                         and aircraft observations; vertical size constant
 *                      b) terminal velocity of particles based on particle
 *                         size and observations of particle types in KA87
 *                         project
 *                      c) liquid water content profile based on aircraft 
 *                         observations
 *                      d) growth by vapour diffusion and riming according to
 *                         graupel model
 *                      e) ice crystal growth from Ryan et al. according to T.
 *                      f) initial spectrum from limited aircraft observations
 *                         in KA87 project and various sailplane measurements;
 *                         start at cloud top at various temperatures
 *                      g) position of particles relative to ground and to
 *                         the top of the thermal calculated
 *                      h) output the particle size distribution and particle 
 *			   positions in the vertical
 *			i) note concentrations based on exponential fit
 *			   and doesn't change.
 *
 */

#include <strings.h>
#include <math.h>
#include <stdio.h>
#include "cdfhdr.h"

#define MAX1 20                       /* number of particle size categories */
#define MAX2 1500                     /* number of altitude values          */
#define MAX3 9000                     /* number of time steps               */
#define MAX4 24                       /* number of grid pts in vertical     */

#define DBAR   20
#define RATTH  0.1
#define RATCT  0.1
#define RATDEB 0.05 
#define RATDOWN 0.05
#define WCTDEB -3.
#define CTOP 15.
#define TDEPTH 6.
#define TGAP 3.
#define TWID 8.
#define DOWN 0.4
#define UMAX 2.
#define ZCT TDEPTH / 6.
#define ZCT1 ZCT / 3.
#define ZCT2 2. * ZCT / 3.

#define P0 1013.
#define T0 288.
#define GAMMA 0.0065
#define RHOI 920
#define EPS 0.622
#define PI 3.14159
#define DELTIM 5.
#define RD 287.05
#define RV 461.51
#define TK 2.32e-2
#define D 2.11e-5
#define TTR 273.16
#define RW 461.51
#define LS 2.837e6
#define LF 3.12e5
#define LV 2.5e6
#define NMAX 500.e6
#define DIS 0.25
#define GRAV 9.81
#define ETA 1.67e-5
#define CPW 4.27e3

 float diam[20] = { .05, .1, .15, .2, .3, .4, 
                   .5, .6, .7, .8, .9, 1., 1.1, 1.2, 
                   1.4, 1.6, 1.8, 2.0, 2.2, 2.4 } ;
 
/* float diam[20] = { 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3,
 *                  0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1. };
 */

float lw[MAX2],db[MAX2],vv[MAX2];     /* lwc, dbar,vertical velocity        */
float alt[MAX2],ps[MAX2],tr[MAX2];    /* altitude, pressure, tempr          */
float wid;                            /* width of channels                  */
float conc[MAX1];                     /* conc in each size category         */
float diaml[MAX1],concl[MAX1];        /* spectrum at requested level        */
float radi[MAX1];                     /* initial radius                     */
float diamh[MAX4][MAX1][MAX3];        /* diam for particular height         */
float diamx[MAX1];                    /* diam in window coords              */
float diamp[MAX1][MAX3];              /* diameter for straight plotting     */
float vtp[MAX1][MAX3];                /* terminal velocity for plotting     */
float timep[MAX1][MAX3];              /* time stored for plotting           */
float rhop[MAX1][MAX3];               /* density of particle                */
float zp[MAX1][MAX3],                 /* altitude of each size              */
      zptop[MAX1][MAX3];              /* rel to ground and cloud top        */
float xp[MAX1][MAX3];                 /* pos of pcle rel to centre of therm */
float thtop1[MAX3],thbase1[MAX3],
      thtop2[MAX3],thbase2[MAX3];     /* top and bottom alt of thermals     */
float cldtop[MAX3],downbase[MAX3];    /* top and bottom of cloud top debris */
float wtht1,wthb1,wtht2,wthb2;        /* vels of top & bot of thermals      */
float rad;                            /* radius of graupel particles (m)    */
float wi,lwc;                         /* updraught, liquid water content    */
float rhoa;                           /* density of air                     */
float reft[MAX3];                     /* reflectivity with time             */
float refg[MAX3][MAX4];               /* reflectivity with time and height  */
float zinit;                          /* initial zltitude                   */
float cf,sumd[MAX3],sumr[MAX4][MAX3]; /* vars used to calculate reflectivty */
float pbase,tbase;                    /* pres (mb) and temp (C) at c base   */
float zbase,cdepth;                   /* alt of base, and depth of cloud    */
float time;                           /* time in minutes                    */
float at,alwc;                        /* adiabatic temp and lwc             */
float zgrid[MAX4];                    /* vertical grid                      */
float xx[MAX3],yy[MAX3];              /* d.v. for plotting                  */
float c1;                             /* offset for framing spectra         */
float spec[MAX1];                     /* d.v conc                           */
float specy[MAX1];                    /* conc only for valid diameters      */
float xinit,yinit;                    /* initialisation pts for windows     */
float x1,x2,wy1,wy2;                  /* window coordinates                 */
float maxup;                          /* max updraught                      */
float wbase,zmax;                     /* up at base, and alt of max up      */
float slope,intcpt;                   /* vars to calc. v vel profile        */
float runtime;                        /* length of run in mins              */
float es,esi;                         /* water and ice vapour pressure      */
float nre,nsh,nnu;                    /* Reynolds, Sherwood and Nusslet no  */
float xmax;                           /* maximum value on x-axis            */
float level;                          /* level used for spectrum            */

int numt,num[MAX1];                   /* number of time pts for plotting    */ 
int j;                                /* global channel counter             */
int nxtick;                           /* number of ticks on x-axis          */
int topflag;                          /* flag set if base of thermal at top */
int tht2flag,thb2flag;                /* when 2nd therm reaches c.b.        */
int post1,posb1,post2,posb2;          /* index for thermal velocities       */

char pname[LINE];
char realdate[LINE];                  /* array for today's date/time        */
char lab1[LINE],lab2[LINE],
     lab3[LINE],lab4[LINE],
     lab5[LINE],lab6[LINE];           /* for labels                         */

float ptz();
float ztp();                         /* altitude to pressure                */
float trev();                        /* adiabatic LWC and temp              */
float graupel();                     /* graupel growth model                */
float drag();                        /* drag coefficient and Reynolds no    */
float tsurf();                       /* surface temperature                 */
float vapour();                      /* vapour pressure                     */
float gr();                          /* diffusion growth of ice             */

FILE *fdate;

main(argc,argv)
int argc;
char *argv[];
{
  int i,k;                          /* loops: i- time k- alt                */
  int ii,jj;                          /* d.v. for altitude and channel loop */
  int kk,kr,kc,nw;                    /* counters for plotting stuff        */

/* get program name */
  sprintf(pname,"%s",argv[0]);
/* check usage */
  if (argc != 7) {
fprintf(stderr,"Usage: %s zi (km) wmax wbase (m/s) z@wmax (km) run_time (min) zspec (km)\n",
                pname);
    exit(0);
  }

/* read in argument value */
  zinit = 1000. * atof(argv[1]);
  maxup = atof(argv[2]); 
  wbase = atof(argv[3]);
  zmax = atof(argv[4]);
  runtime = atof(argv[5]);
  level = atof(argv[6]);
  
/* zero arrays */
  for (i = 0; i < MAX3; i++) 
    sumd[i] = 0.;
  for (j = 0; j < MAX1; j++)
    concl[j] = diaml[j] = 0.;

/* initial spectrum */
  for (j = 0; j < MAX1; j++) {
    radi[j] = diam[j] / 2.;
/* two exponential fits */
/*    if (diam[j] < 0.5) */
      conc[j] = exp10(5.) * exp(-39.47 * diam[j]);
/*    else 
 *     conc[j] = exp10(3.7) * exp(-6.3 * diam[j]);
 */
/* the following from Jim Dye's paper - see p 35 of book */
/*    conc[j] = 250. * exp(-3.00 * diam[j]);*/
  }

/* cloud base pressure and temperature */
/* pbase = 672.;*/
/* tbase = 8.2;*/
  pbase = 970.;
  tbase = 25.2;
  zbase = ptz(pbase);
  cdepth = CTOP - zbase / 1000.;

/* slope and intercept of vertical velocity profile */
    slope = (maxup - wbase) / (zmax - zbase / 1000.);
    intcpt = wbase - slope * zbase / 1000.;

/* substance for profiles */
  for (ii = 0; ii < MAX2; ii++) {
    alt[ii] = 10. * ii;

/* liquid water content and temperature */
    ps[ii] = ztp(alt[ii]);
    trev(pbase,tbase,ps[ii],&at,&alwc);
    lw[ii] = alwc * 1.0e-3;
    tr[ii] = at + 273.15;
    db[ii] = DBAR + ii/125.;
/* vertical velocity */
    if (alt[ii] < zbase)
      vv[ii] = 1.;
    else if (alt[ii] / 1000. <= zmax)
      vv[ii] = slope * alt[ii] / 1000. + intcpt;
    else
      vv[ii] = maxup - (alt[ii] /1000 - zmax) * 1.75;
  }

/* number of time points */
  numt = (int)(runtime * 60. / DELTIM);

/* initial positions of thermals */
  thtop1[0] = zinit/1000. + 1.;
  cldtop[0] = thtop1[0];
  thbase1[0] = thtop1[0] - TDEPTH;
  if (thbase1[0] - TGAP >= 3.)
    thtop2[0] = thbase1[0] - TGAP;
  else
    thtop2[0] = BAD;
  if (thtop2[0] - TDEPTH >= 3.)
    thbase2[0] = thtop2[0] - TDEPTH;
  else
    thbase2[0] = BAD;
  downbase[0] = BAD;

  post1 = posb1 = post2 = posb2 = 0;
  for (ii = 0; ii < MAX2; ii++) {
    if (ii > 0) {
      if (thtop1[0] < alt[ii]/1000. && thtop1[0] >= alt[ii-1]/1000.)
        post1 = ii;
      if (thbase1[0] < alt[ii]/1000. && thbase1[0] >= alt[ii-1]/1000.)
        posb1 = ii;
      if (thtop2[0] < alt[ii]/1000. && thtop2[0] >= alt[ii-1]/1000.)
        post2 = ii;
      if (thbase2[0] < alt[ii]/1000. && thbase2[0] >= alt[ii-1]/1000.)
        posb2 = ii;
    }
  }
  wtht1 = vv[post1];
  wthb1 = vv[posb1];
  wtht2 = vv[post2];
  wthb2 = vv[posb2];

/* positions of thermals (in km) with time */
  topflag = tht2flag = thb2flag = 0;
  for (i = 1; i < numt; i++) {
    thtop1[i] = thtop1[i-1] + (wtht1 * DELTIM)/1000.;
    if (thtop1[i] >= CTOP)
      thtop1[i] = CTOP;
    thbase1[i] = thbase1[i-1] + (wthb1 * DELTIM)/1000.; 

    if (tht2flag != 1) {
      if (thbase1[i] - TGAP >= 3.) {
        thtop2[i] = thbase1[i] - TGAP;
        tht2flag = 1;
      }
      else
        thtop2[i] = BAD;
    }
    else 
      thtop2[i] = thtop2[i-1] + (wtht2 * DELTIM)/1000.;

    if (thb2flag != 1) {
      if (thtop2[i] - TDEPTH >= 3.) {
        thbase2[i] = thtop2[i] - TDEPTH;
        thb2flag = 1;     
      }
      else 
        thbase2[i] = BAD;
    }
    else
      thbase2[i] = thbase2[i-1] + (wthb2 * DELTIM)/1000.; 

    if (thbase1[i] >= CTOP - TDEPTH / 2. && topflag == 0) {
      topflag = 1;
      thtop1[i] = thtop2[i];
      thbase1[i] = thbase2[i];
      thtop2[i] = thbase1[i] - TGAP;
      thbase2[i] = thtop2[i] - TDEPTH;
      tht2flag = thb2flag = 0;
    }
    if (topflag == 1) {
      cldtop[i] = cldtop[i-1] + (WCTDEB * DELTIM)/1000.;
      downbase[i] = cldtop[i] - TDEPTH / 2.;
      if (downbase[i] <= thtop1[i])
        downbase[i] = thtop1[i];
      if (cldtop[i] < thtop1[i])
        topflag = 0;
    }
    else {
      cldtop[i] = thtop1[i];
      downbase[i] = BAD;
    }
    for (ii = 0; ii < MAX2; ii++) {
      if (ii > 0) {
        if (thtop1[i] < alt[ii]/1000. && thtop1[i] >= alt[ii-1]/1000.)
          post1 = ii;
        if (thbase1[i] < alt[ii]/1000. && thbase1[i] >= alt[ii-1]/1000.)
          posb1 = ii;
        if (thtop2[i] < alt[ii]/1000. && thtop2[i] >= alt[ii-1]/1000.)
          post2 = ii;
        if (thbase2[i] < alt[ii]/1000. && thbase2[i] >= alt[ii-1]/1000.)
          posb2 = ii;
      }
    }
    wtht1 = vv[post1];
    wthb1 = vv[posb1];
    wtht2 = vv[post2];
    wthb2 = vv[posb2];
  }

/* growth of all particles handled in subroutine graupel */
  wi = 0.;
  for (j = 0; j < MAX1; j++) 
    graupel();

/* reflectivity */
  cf = 0.23 / 0.93;
  for (j = 0; j < MAX1; j++) 
    for (i = 0; i < num[j]; i++) 
      sumd[i] += pow(rhop[j][i]*1.e-3,2.) * conc[j] * pow(diamp[j][i],6.);
  for (i = 0; i < numt; i++)
    reft[i] = (sumd[i] > 0.) ? 10. * log10(cf * sumd[i]) : BAD;

/* calculate size distribution and reflectivity on a 500 m vertical
 * "grid" */
  for (k = 0; k < MAX4; k++) { 
    zgrid[k] = 3. + (float) k / 2.;
    for (i = 0; i < MAX3; i++)
      sumr[k][i] = 0.;
  }
  for (k = 0; k < MAX4; k++) {
    for (j = 0; j < MAX1; j++) {
      for (i = 0; i < numt; i++)
        diamh[k][j][i] = BAD;
      for (i = 0; i < num[j]; i++) {
        if (zp[j][i] < zgrid[k] + .25 && zp[j][i] >= zgrid[k] - .25) {
          sumr[k][i] += pow(rhop[j][i]*1.e-3,2.)*conc[j]*pow(diamp[j][i],6.);
          diamh[k][j][i] = diamp[j][i];
        }
      }
    }
    for (i = 0; i < numt; i++) {
      refg[i][k] = (sumr[k][i] > 0.) ? 10. * log10(cf * sumr[k][i]) : BAD;
      refg[i][k] = (refg[i][k] > -10.) ? refg[i][k] : BAD;
    }
  }

/* does particle pass through the requested level? */
  for (j = 0; j < MAX1; j++) {
    for (i = 1; i < num[j]; i++) {
      if ((zp[j][i] < level + 0.1 && zp[j][i] > level - 0.1) &&
          (zp[j][i] < zp[j][i-1]) && (xp[j][i] < TWID - DOWN)) { 
        concl[j] = conc[j];
        diaml[j] = diamp[j][i];
      }
    }
  }

/* plotting */
  fprintf(stderr,"Plotting\n");
  gopen();
  badset(9999.);
  gclear();


/* plot of altitude of each particle size */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,30.,7,"time (min)",3.,15.,7,"Height (km)");
  for (j = 0; j < MAX1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = timep[j][i];
      yy[i] = zp[j][i];
    }
    line(1,1,num[j],xx,yy); 
  }
  for (i = 0; i < numt; i++) 
    xx[i] = timep[0][i];
  line(1,1,numt,xx,cldtop);
  line(1,1,numt,xx,thtop1);
  line(1,2,numt,xx,thbase1); 
  line(1,2,numt,xx,downbase);
  line(1,3,numt,xx,thtop2);
  line(1,4,numt,xx,thbase2);
/* labels */
/* write current date and time */
  system("date '+%T  %a %d %h 19%y' > tempdate");
  if ((fdate = fopen("tempdate","r")) == NULL) {
    fprintf(stderr,"can't open tempdate\n");
    exit(0);
  }
  fgets(realdate,LINE,fdate);
  fclose(fdate);
  realdate[strlen(realdate)-1] = 0;
  system("rm tempdate");
  sprintf(lab1,"%s: %s",pname,realdate);
  sprintf(lab2,"zi = %4.1f km, TDEPTH = %3.1f km",zinit/1000.,TDEPTH);
  sprintf(lab3,"Lmax = %4.2f Lad, wmax = %4.1f m/s",RATTH,maxup);
  sprintf(lab4,"Lct = %4.2f Lad, Ldeb = %4.2f Lad",RATCT,RATDEB);
  sprintf(lab5,"umax = %4.1f m/s",UMAX);
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* particle size distribution at requested level */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.0,5.0,6,"D (mm)",-1.0,6.0,8,"N (m^-3 mm^-1)");
  for (j = 0; j < MAX1; j++) {
    if (j == 0)
      wid = diam[j];
    if (j > 0)
      wid = diam[j] - diam[j - 1];
    spec[j] = (concl[j] > 0.) ? log10(concl[j] / wid) : BAD;
    mark(1,3,diaml[j],spec[j],0.2);
  }
  line(1,1,MAX1,diaml,spec);
  sprintf(lab6,"Spectrum at %4.1f km",level);
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  label(2,0.01,0.64,lab6);
  gpause();
  gclear();

/* vertical velocity of thermals with height */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,20.,6,"vert. vel. (m/s)",3.,15.,7,"Height (km)");
  for (ii = 0; ii < MAX2; ii++)
    yy[ii] = alt[ii] / 1000.;
  line(1,1,MAX2,vv,yy); 
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* trajectories of particles */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,TWID+1.,(int)TWID+2,"x (km)",3.,15.,7,"Height (km)");
  for (j = 0; j < MAX1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = xp[j][i];
      yy[i] = zp[j][i];
    }
    line(1,1,num[j],xx,yy); 
  }
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* diameter of particles vs height */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,10.,6,"diam (mm)",3.,15.,7,"Height (km)");
  for (j = 0; j < MAX1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = diamp[j][i];
      yy[i] = zp[j][i];
    }
    line(1,1,num[j],xx,yy);
  }
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* terminal velocity of particles vs height */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,12.,7,"vt (m/s)",3.,15.,7,"Height (km)");
  for (j = 0; j < MAX1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = vtp[j][i];
      yy[i] = zp[j][i];
    }
    line(1,1,num[j],xx,yy);
  }
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* diameter of particles vs time */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,30.,7,"time (min)",0.,10.,11,"graupel diam (mm)");
  for (j = 0; j < MAX1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = timep[j][i];
      yy[i] = diamp[j][i];
    }
    line(1,1,num[j],xx,yy);
  }
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* reflectivity vs time and height */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  xmax = (float)numt;
  nxtick = 6;
  axes(1,0.,xmax,nxtick,"time=x*5/60 (mins)",0.,24.,7,"alt=0.5*y+3 (km)");
  conto(1,numt,MAX4,5.,1,refg); 
  grid(1,numt,MAX4,20,1,refg);
  fill(1,numt,MAX4,-20.,30.,1,refg);
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* density of particles vs diameter */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,10.,6,"graupel diam (mm)",0.,1.,11,"density (g/cm^3)");
  for (j = 0; j < MAX1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = diamp[j][i];
      yy[i] = rhop[j][i]/1000.;
    }
    line(1,1,num[j],xx,yy);
  }
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* particle size distributions with height in range 5 - 8.5 km for chosen 
*  time */
/* convert spectra into appropriate range for plotting */
/* abscissa: 0.1 -> 10^6 m^-3 mm^-1 */
  c1 = 1./7.;
  for (j = 0; j < MAX1; j++) {
    if (j == 0)
      wid = diam[j];
    if (j > 0)
      wid = diam[j] - diam[j - 1];
    spec[j] = conc[j] / wid;
  }

  kk = nw = 0;
/* row count */
  for (kr=0;kr<2;kr++) {
    yinit = 5 * kr;
/* column count */
    for (kc=0;kc<4;kc++) {
      xinit = 4 * kc;
/* index and window counts */
      kk = kk + 1;
      nw = nw + 1;
/* x & y positions */
      x1 = xinit;
      x2 = x1 + 3.;
      wy1 = yinit;
      wy2 = wy1 + 4.;
      window(nw,x1,x2,wy1,wy2);
/* place ticks */
      ytick(nw);
      xtick(nw);
/* plot PSD for different heights for last time step */
      k = 4 + nw;
      sprintf(lab3,"%4.1f km",zgrid[k]);
      label(nw,0.55,0.9,lab3);
      for (j = 0; j < MAX1; j++) {
        diamx[j] = (diamh[k][j][numt-5]<99999.) ? diamh[k][j][numt-5]/6. : BAD;
        specy[j] = (diamh[k][j][numt-5]<99999.) ? log10(spec[j])/7.+c1 : BAD;
        mark(nw,3,diamx[j],specy[j],0.2);
      }
      line(nw,1,MAX1,diamx,specy);
    }
  }
/* labels */
  window(9,0.,15.,9.,10.);
  label(9,0.01,0.75,lab1);
  label(9,0.01,0.35,lab2);
  sprintf(lab1,"x-axis: 0 - 5 mm; y-axis: 0.1 - 10^6 m^-3 mm^-1");
  label(9,0.5,0.75,lab1);
  gpause();
  gclose();
}

float graupel()
{

  float z,pres,temp,tempc,dbar;        /* current values                     */
  float zold;                          /* old value of z                     */
  float zkm;                           /* height in km                       */
  float hw;                            /* horizontal wind                    */
  float rhog,rhogi,rhor;               /* densities                          */
  float ta,tsk,ts;                     /* air, and sfce temp, and d.v.       */
  float vt;                            /* terminal velocity                  */
  float vtsave;                        /* save terminal velocity             */
  float vt200;                         /* vt for 200 diameter particle       */
  float vt300;                         /* vt for 500 diameter particle       */
  float delm,delrr,delrd,delr,
        delmd,delmr;                   /* increments                         */
  float dmr300,dr300,dd200,ddsave;     /* increments for 200 & 300 um pcles  */
  float drtrans;                       /* increment for 200 < D < 300        */
  float dmm;                           /* diam in mm                         */
  float mass,mom;                      /* mass and momentum                  */
  float kappa;                         /* collection kernel                  */
  float beta;                          /* exponent in collection kernel      */
  float re,rvs;                        /* sat water and ice vapour density   */
  float vimp;                          /* impact velocity                    */
  float arg,arg2,arg3;                 /* dummy arguments                    */
  float kapb;                          /* kappa for collection kernel        */
  float sumkap,sumrh,sumy;             /* sums                               */
  float hxa,xi,yi,stdv;                /* variables for cloud drop spectrum  */
  float nu;                            /* kinematic viscosity                */
  float time;                          /* time variable                      */
  float cd;                            /* drag coefficient                   */
  float frac,frac1,frac2;              /* fractions used in shedding         */
  float tsc;                           /* tempr in C                         */
  float ns;                            /* Stokes parameter                   */
  float w,w2,w3,w4;                    /* arguements used in Reynolds no.    */
  float wpcle;                         /* vertical velocity of particle      */

  int i,k,ii,indx;                     /* counters                           */
  int tflag;                           /* flag for 1st round of loop         */
  int pos;                             /* pointer to current height          */
  int downflag;                        /* flag set to 1 if in downdraught    */
  int graup;                           /* test variable                      */

/* loop for graupel environment values */
  for (ii = 0; ii < MAX2; ii++) 
/* get pointer to initial height */
    if (ii > 0)
      if(zinit < alt[ii] && zinit >= alt[ii-1])
        pos = ii;
  z = alt[pos];
  zold = z;

/* initial values */
  pres = ps[pos] * 100.;
  temp = tr[pos];
  lwc = lw[pos];
  dbar = db[pos];
  timep[j][0] = 0.;
  zp[j][0] = z / 1000.;
  xp[j][0] = 0.;

/* density of air */
  rhoa = pres / (RD * temp); 
/* kinematic viscosity */
  nu = ETA / rhoa;

/* initial radius in m and density kg m^-3 */
  rad = radi[j] * 1.0e-3;
  diamp[j][0] = radi[j] * 2.;
  rhog = 900.;
  rhop[j][0] = rhog; 
/* initial mass */
  mass = 4. * PI * pow(rad,3.) * rhog / 3.;

/* set counters and things */
  time = 0.;
  tflag = 0;
  hw = 0.;
  wi = 0.;
  indx = 0;
  downflag = 0;
  graup = 0;

/* calculate vt and delmr for a 300 um diameter particle */
/* Assume density of 220 g/cm^3 just now as a quick fix - 7/7/94 */
/* and use simple formula for kapb */
  drag(0.15e-3,220.,rhoa,&cd);
  vt300 = 8. * 0.15 * 1.0e-3 * 220. * GRAV / (3. * rhoa * cd);
  dmr300 = PI * pow(0.15e-3,2.) * vt300 * lwc * DELTIM;
  dr300 = dmr300 / (4. * PI * pow(0.15e-3,2.) * 220.);

/* growth loop */
  for (i = 1;i < MAX3;i++) {
    drag(rad,rhog,rhoa,&cd);

/* terminal velocity */
/* initial terminal velocity of particles smaller than D = 500 um 
 * graph taken for filled-in stellars in fig 10-33 of P&K (Curve found
 * using points read-off graph and fitted using Maple) 
 */
    tempc = temp - 273.15;
    dmm = rad*2.0e3;
    if (dmm <= 0.30) {
      if (dmm <= 0.2) {
        if ((tempc > -6.5 && tempc <= -4.) || (tempc > -16 && tempc <= -14.))
          vt = -0.014 + 0.395 * dmm - 0.177 * pow(dmm,2.) +
                0.073 * pow(dmm,3.) - 0.0153 * pow(dmm,4.); 
        if ((tempc > -14. && tempc <= -12) || (tempc > -8 && tempc <= -6.5) ||
             tempc > -4)
          vt = 40. * dmm;
        if (tempc > -12 && tempc <= -8)
          vt = 51. * dmm;
        if (tempc <= -16)
          vt = 60. * dmm;
        vt *= 0.01;
        vtsave = vt;
/* make transition to graupel particle as smoothly as possible */
/* store vt for first time past 200 */
      }
      else {
        if (graup == 0)
          vt200 = vtsave;
        vt = vt200 + (vt300 - vt200) * (dmm - 0.2) / 0.1;
      }
    }
    else {
      vt = 8. * rad * rhog * GRAV / (3. * rhoa * cd);
      vt = pow(vt,0.5);
    }

/* On first round, use temp + 2.4 for surface temp */
    if (i == 1) ts = temp + 2.4;

/* dm for diffusion */
    if (dmm <= 0.30) { 
      if (dmm <= 0.20) { 
        ts = temp;
        delrd = 1.0e-6 * gr(tempc);
        ddsave = delrd;
      }
/* make transition to full riming as smooth as possible */
      else {
        if (graup == 0) {
          dd200 = ddsave; 
          graup = 1;
        }
        drtrans = dd200 + (dr300 - dd200) * (dmm - 0.2) / 0.1;
      }
    } 
    esi = 100. * vapour(TTR) * (exp((ts - TTR) * LS / (RW * ts * TTR)));
    rvs = esi / (RV * ts);
    es = 100. * vapour(temp);
    re = es / (RV * temp);
/* Sherwood number as per Mason (1971) */
    nsh = 0.58 * pow(nre,0.5);
/* Nusselt number as per Mason (1971) */
    nnu = nsh;
    delmd = (dmm > 0.2) ? 2. * PI * rad * D * nsh * (re - rvs) * DELTIM : 0.;

/* dm for riming - only for D > 300 um */
    delmr = 0.;
    if (dmm > 0.30) {
/* droplet spectrum -- Gaussian distribution matched to LWC */
/*                     xi is the radius and yi is the concentration */

/* derived quantities */
      stdv = DIS * dbar / 2.;
      sumkap = 0.;
      sumrh = 0.;
      sumy = 0.;
/* loop to calculate the density and collection kernel 
 * due to different sizes of drops */
/* miss out if ts ~ 0 C */
      if (ts < 273.15) {
        mom = mass * vt * 1.0e5;
        beta = 0.738;
        for (k = 0;k < 24;k++) {
          xi = (26. - (k + 1.)) * 1.0e-6;
          hxa = (xi * 1.0e6 - dbar / 2.) / stdv;
          hxa = pow(hxa,2.) / 2.;
          yi = NMAX * exp(-hxa);
/*        xtmp = xi * 1.0e6;
 *       if (xtmp == dbar/2. - 4.) yconst = yi;
 *       if (xtmp < dbar/2. - 4.) yi = yconst;
 */
          sumy = yi + sumy;
/* collection kernel - formulae derived from B & G relationships */
          if (k == 23) kappa = 1.09 * 1.0e-6 * pow(mom,beta);
          if (k == 22) kappa = 3.56 * 1.0e-6 * pow(mom,beta); 
          if (k == 21) kappa = 5.41 * 1.0e-6 * pow(mom,beta); 
          if (k == 20) kappa = 6.80 * 1.0e-6 * pow(mom,beta); 
          if (k == 19) kappa = 7.75 * 1.0e-6 * pow(mom,beta); 
          if (k == 18) kappa = 8.37 * 1.0e-6 * pow(mom,beta); 
          if (k == 17) kappa = 8.80 * 1.0e-6 * pow(mom,beta); 
          if (k == 16) kappa = 9.13 * 1.0e-6 * pow(mom,beta); 
          if (k == 15) kappa = 9.38 * 1.0e-6 * pow(mom,beta); 
          if (k == 14) kappa = 9.58 * 1.0e-6 * pow(mom,beta); 
          if (k == 13) kappa = 9.75 * 1.0e-6 * pow(mom,beta); 
          if (k == 12) kappa = 9.87 * 1.0e-6 * pow(mom,beta); 
          if (k == 11) kappa = 9.97 * 1.0e-6 * pow(mom,beta); 
          if (k == 10) kappa = 10.07 * 1.0e-6 * pow(mom,beta); 
          if (k == 9) kappa = 10.17 * 1.0e-6 * pow(mom,beta); 
          if (k == 8) kappa = 10.22 * 1.0e-6 * pow(mom,beta);
          if (k == 7) kappa = 10.30 * 1.0e-6 * pow(mom,beta);
          if (k == 6) kappa = 10.37 * 1.0e-6 * pow(mom,beta);
          if (k <= 5) kappa = 10.45 * 1.0e-6 * pow(mom,beta);
          sumkap = sumkap + kappa * yi;

/* impact velocity of impinging cloud drops from R & H for graupel density */
/* Stokes parameter */
          ns = 2. * vt * xi * xi * 1000. / (9. * ETA * rad);
          w = log10(ns);
          w2 = w * w;
          w3 = pow(w,3.);
          w4 = pow(w,4.);
/* no interpolation - just using a range of nre */
          if (nre <= 20.) {
            if (ns >= 0.4 && ns <= 10.) {
              vimp = 0.1701 + 0.7246 * w + 0.2257 * w2 - 1.13 * w3
                   + 0.5756 * w4;
            }
            if (ns < 0.4) vimp = 0;
            if (ns > 10.0) vimp = 0.57;
          }
          if (nre > 20. && nre <= 65.) {
            if (ns >= 0.2 && ns <= 10.) {
              vimp = 0.2927 + 0.5085 * w - 0.03453 * w2 - 0.2184 * w3
                   + 0.03595 * w4;
            }
            if (ns < 0.2) vimp = 0.0;
            if (ns > 10.0) vimp = 0.59;
          }
          if (nre > 65. && nre <= 200.) {
            if (ns >= 0.2 && ns <= 10.0) {
              vimp = 0.3272 + 0.4907 * w - 0.09452 * w2 - 0.1906 * w3
                   + 0.07105 * w4;
            }
            if (ns < 0.2) vimp = 0.0;
            if (ns > 10.0 ) vimp = 0.61;
          }
          if (nre > 200.) {
            if (ns >= 0.2 && ns <= 10.0) {
              vimp = 0.356 + 0.4738 * w - 0.1233 * w2 - 0.1618 * w3
                   + 0.08087 * w4;
            }
            if (ns < 0.2) vimp = 0.0;
            if (ns > 10.0 ) vimp = 0.63;
          }
          vimp = vimp * vt;
/* the density of the newly formed rime  - rbar in um, Ts in C vimp in m/s */
/* -- equation from H & P */
          tsc = ts - 273.15;
          arg = - dbar * vimp / (2. * tsc);
          arg2 = pow(arg,2.);
          arg3 = pow(arg,3.);
          if (tsc <= -5.|| arg >= -1.6) 
            rhor = 0.30 * pow(arg,0.44);
          else 
            rhor = exp(-0.03115 - 1.7030 * arg + 0.9116 * arg2
                       - 0.1224 * arg3);
/*              rhor = 0.261 * pow(arg,0.38);*/
          sumrh = rhor * yi + sumrh;
/* end of spectrum loop */
        }
/* adjust to kg/m3 */
        rhor = sumrh * 1.0e3 / sumy;
/* collection kernel */
        kapb = sumkap / sumy;
/* end of ts condition */
      }
      else {
        rhor = 900.;
/* let collection kernel be traditional expression */
        kapb = PI * pow(rad,2.) * vt;
      }

/* Tsurf */
      if (tflag == 0) {
/* only go to sub tsurf if kapb is significant (what is that?!) */
        if (kapb >= 0.2) 
          ts = tsurf(kapb,temp);
        else
          ts = temp;
        if (ts >= 273.) {
          tflag = 1;
          ts = 273.15;
        }
      }


/* increase in mass due to riming */
      delmr = kapb * lwc * DELTIM;

/* If warm growth, assume unfrozen water is shed - calculate frozen fraction 
 * - see Nelson (1983); also assume eff. = 1 */
      if (tflag == 1) {
/*        delmr = PI * pow(rad,2.) * lwc * 1. * vt * DELTIM;*/
        es = 100. * vapour(ts);
        rvs = es / (RV * ts);
        re = es / (RV * temp);
        frac1 = TK * (temp - ts); 
        frac2 = LV * D * (rvs - re);
        frac = -PI * 2. * rad * nnu * (frac1 - frac2); 
        frac = frac - delmr * CPW * (temp - ts);
        frac = frac / (delmr * LF);
/* increase frac artificially - 29 Sept. 88 */
        frac = 1.;
        delmr = frac * delmr;
      }

/* end of riming loop for D > 300 um */
    }

/* dR and Rnew */
    if (rad*2.e3 <= 0.2)
      delrr = 0.;
    if (rad*2.e3 <= 0.3 && rad*2.e3 > 0.2)
      delrr = drtrans;
    if (rad*2.e3 > 0.3) {
      delrr = delmr / (4. * PI * rad * rad * rhor);
      delrd = fabs(delmd) / (4. * PI * rad * rad * RHOI);
      if (delmd < 0.) delrd = -delrd;
    }
    delr = delrr + delrd;
    rad = rad + delr;
/* equivalent diameter */
/*      arg = rhog / 1.0e3;
 *     diam[i] = pow(arg,0.3333) * diam[i];
 */

/* new bulk density of the graupel particle */
    mass = mass + delmr;
    rhog = mass * 3. / (4. * PI * pow(rad,3.));
/* set limits on graupel density */
    if (rhog < 100.) rhog = 100.;
    if (rhog > 900.) rhog = 900.;

    time = time + DELTIM / 60.;

/* particle velocity: vt is positive down */
    wpcle = wi - vt;

/* vertical and horizontal position */
    z = zold + wpcle * DELTIM;
    xp[j][i] = xp[j][i-1] + hw * DELTIM / 1000.;

/* check and change environmental conditions */
    for (ii = 0; ii < MAX2; ii++) 
      if (ii > 0)
        if (z < alt[ii] && z >= alt[ii-1]) 
          pos = ii;
    pres = ps[pos] * 100.;
    zkm = z/1000.;

/* assume horizontal wind is zero until set later */
    hw = 0;
/* check position of particle relative to thermal if not in downdraught */
    if (downflag == 0) {
/* particle still in thermal ? */
      if ((zkm <= thtop1[i]-ZCT && zkm >= thbase1[i]) || 
          (zkm <= thtop2[i]-ZCT && zkm >= thbase2[i])) {
        temp = tr[pos] - 2.;
        lwc = RATTH * lw[pos];
        dbar = db[pos];
        wi = vv[pos];
      }

/* overshoot cloud top? */
      if (zkm >= CTOP)
        zkm = CTOP;

/* in cloud top region (top of thermal 1) */
      if (zkm <= thtop1[i] && zkm > thtop1[i]-ZCT1) {
        temp = tr[pos] - 4.;
        lwc = RATCT * lw[pos];
        dbar = db[pos];
        hw = UMAX;
        wi = vv[pos] / 2.;
      }
      if (zkm <= thtop1[i]-ZCT1 && zkm > thtop1[i]-ZCT2) {
        temp = tr[pos] - 3.;
        lwc = RATCT * lw[pos];
        dbar = db[pos];
        hw = 2. * UMAX / 3.;
        wi = vv[pos] / 1.7;
      }
      if (zkm <= thtop1[i]-ZCT2 && zkm > thtop1[i]-ZCT) {
        temp = tr[pos] - 3.;
        lwc = RATCT * lw[pos];
        dbar = db[pos];
        hw = UMAX / 3.;
        wi = vv[pos] / 1.3;
      }

/* out of thermal, into debris under thermal (over thermal if at start) */
      if ((zkm > thtop1[i] || zkm < thbase1[i]) &&
          (zkm > thtop2[i] || zkm < thbase2[i]) &&
          (zkm < downbase[i]) || (zkm > thtop1[i] && zkm > thtop2[i])) {
        temp = tr[pos] - 4.;
        lwc = RATDEB * lw[pos];
        dbar = db[pos];
        hw = 0.;
        wi = 0.;
      }

/* in descending thermal remnants */
      if (zkm > thtop1[i] && zkm <= cldtop[i] && zkm >= downbase[i]) {
        temp = tr[pos] - 4.;
        lwc = RATDEB * lw[pos];
        dbar = db[pos];
        hw = 0.;
        wi = WCTDEB; 
      }

/* horizontal position in km */
      if (xp[j][i] >= TWID - DOWN) {
        temp = tr[pos] - 6.;
        lwc = RATDOWN * lw[pos];
        dbar = db[pos];
        hw = 0.;
        wi = -3.;
        downflag = 1;
      }
    }

/* for plotting (in main program) */
    diamp[j][i] = rad * 2.e3;
    rhop[j][i] = rhog;
    timep[j][i] = time;
    zp[j][i] = zkm;
    vtp[j][i-1] = vt;
      
/* if  time > run_time, env temp is warmer than 0, or z < zbase, then break */
    if (time > runtime || temp > 273.15 || z <= zbase) {
      num[j] = indx;
      break; 
    }
/* density of air */
    rhoa = pres / (RD * temp); 
    zold = z;

/* back for more growth */
    indx++;
  }
  num[j] = indx-1;
  return(0);
}

/* drag -- calculate Reynolds number and drag coefficient */

float drag(r,rhog,rhoa,cd)
float r,rhog,rhoa,*cd;
{
  float xd,a,b;                        /* variables used in Best no          */
/* Best (or Davies) number for current pressure level */
  xd = 32. * rhog * rhoa * GRAV * pow(r,3.) / (3. * ETA * ETA);
    
  if (xd < 1.09e4) {
    a = 0.0688;
    b = 0.769;
  }
  if (xd >= 1.09e4 && xd < 6.58e5) {
    a = 0.347;
    b = 0.595;
  }
  if (xd >= 6.58e5) {
    a = 3.6184;
    b = 0.420;
  }
  nre = a * pow(xd,b);
  *cd = 8. * pow(nre,-0.27); 
}

/* tsurf -- subroutine to calculate the surface temperature of the growing */
/*          graupel particle */
/*          uses the relationships derived in Pflaum and Pruppacher */

float tsurf(kap,tsk)
float kap,tsk;

{
  float to,ci,rve,rvs,tsr,tsold,eff,eff1,eff2,nsh2,nnu2,ts1,ts2,
        nsc,mu,npr,cpv,tdiff,tm1;
  int ik;

/* constants */
/*  eff = 0.8;*/
  cpv = 1.87e3;
  ci = 2.031e3;
  mu = 1.667e-5;
  to = 273.15;
  ik = 0;
/* mixing ratio */
  es = 100. * vapour(tsk);
  rve = es / (RV * tsk);
/* Schmidt number */
  nsc = mu / (rhoa * D);
/*  nsh2 = 2. + 0.6 * pow(nsc,0.333) * pow(nre,0.5);*/
/* Prandtl number */
  npr = mu * cpv / TK;
/*  nnu2 = 2. + 0.6 * pow(npr,0.333) * pow(nre,0.5);*/
/*printf("nre %f, nsh,%f, nsh2,%f, nnu,%f, nnu2 %f\n",nre,nsh,nsh2,nnu,nnu2);*/
  tm1 = kap * lwc / (rad * PI);
/* calculate surface temperature */
  tsr = tsk;
  tdiff = 5.;
  while(tdiff > 0.10) {
    ik += 1;
    tsold = tsr;
/*    esi = 100. * vapour(TTR) * (exp((tsr - TTR) * LS / (RW * tsr * TTR)));*/
    esi = 100. * vapour(tsr);
    rvs = esi / (RV * tsr);
    ts1 = tm1 * (LF + CPW * (tsk - to) + ci * to); 
    ts2 = 2. * D * LS * nsh * (rve - rvs) + 2. * TK * tsk * nnu;
    tsr = (ts1 + ts2) / (2. * TK * nnu + tm1 * ci);    
    tdiff = fabs(tsr - tsold); 
    if (ik > 100 && tsr > 273.15) {
      tsr = 273.15;
      break;
    }
  }
  tsk = tsr;
/*  printf("ik tdiff ts nre %d %f %f %f \n",ik,tdiff,ts,nre);*/
  return(tsr);
}
  
float ztp(zz)
float zz;
{
  float p;
  p = P0 * pow((T0 - GAMMA * zz) / T0,GRAV / (RD * GAMMA));
  return(p);
}

/* function to calculate altitude for a given pressure */
/* clacuation uses a constant lapse rate of 6.5 C/km (see Hess pp 82-83) */
/* constants are defined at the top of this program */
float ptz(p)
float p;
{
  float ex,ptzr;

  ex = (RD*GAMMA)/GRAV;
  ptzr = T0*(1.0-pow((p/P0),ex))/GAMMA;
  return(ptzr);
}

float vapour(t)
float t;
{
  float e,v,arg1,arg2;
/* goff-gratch formula for water saturation vapour pressure.*/
  arg1 = 11.344 * (1. - t / 373.16);
  arg2 = 3.49149 * (1. - 373.16 / t);
  e = -7.90298 * (373.16 / t - 1.) + 5.02808 * log10(373.16 / t)
      - 1.3816e-7 * (pow(10.,arg1) - 1.)
      + 8.1328e-3*(pow(10.,arg2) - 1.);
  v = 1013.246 * pow(10.,e);
  return(v);
}

/* places ytick marks */
ytick(w)
int w;
{
  float xt[2],yt[2],dt;
  int i;

  dt = 1.0/7.0;
  xt[0] = 0.0;
  xt[1] = 0.025;
  yt[0] = 0.0;
  for (i=0;i<6;i++) {
    yt[0] = yt[0] + dt;
    yt[1] = yt[0];
    line(w,1,2,xt,yt);
  }
  return(OK);
}

/* places xtick marks */
xtick(w)
int w;
{
  float xt[2],yt[2],dt;
  int i;

  dt = 1.0/5.0;
  yt[0] = 0.0;
  yt[1] = 0.025;
  xt[0] = 0.0;
  for (i=0;i<4;i++) {
    xt[0] = xt[0] + dt;
    xt[1] = xt[0];
    line(w,1,2,xt,yt);
  }
  return(OK);
}

/* gr -- return dr/dt values from Ryan et al's lab expts for diffusional */
/* growth of ice                                                         */

float gr(tempr)
float tempr;
{
  if (tempr >= -3.5) return(0.2);
  if (tempr < -3.5 && tempr >= -4.5) return(0.5);
  if (tempr < -4.5 && tempr >= -5.5) return(1.0);
  if (tempr < -5.5 && tempr >= -6.5) return(1.25);
  if (tempr < -6.5 && tempr >= -7.5) return(0.75);
  if (tempr < -7.5 && tempr >= -8.5) return(0.5);
  if (tempr < -8.5 && tempr >= -9.5) return(0.3);
  if (tempr < -9.5 && tempr >= -10.5) return(0.3);
  if (tempr < -10.5 && tempr >= -11.5) return(0.35);
  if (tempr < -11.5 && tempr >= -12.5) return(0.5);
  if (tempr < -12.5 && tempr >= -13.5) return(0.6);
  if (tempr < -13.5 && tempr >= -14.5) return(1.3);
  if (tempr < -14.5 && tempr >= -15.5) return(1.8);
  if (tempr < -15.5 && tempr >= -16.5) return(1.3);
  if (tempr < -16.5 && tempr >= -17.5) return(0.8);
  if (tempr < -17.5 && tempr >= -18.5) return(0.7);
  if (tempr < -18.5 && tempr >= -19.5) return(0.6);
  if (tempr < -19.5 && tempr >= -20.5) return(0.5);
  if (tempr < -20.5 && tempr >= -21.5) return(0.4);
}
