/* icemult -- A wee model to predict the development of ice by the Hallett-
 *            Mossop (and other) processes.   
 *            Based on the program precip.c with the incorporation of ice 
 *            multiplication processes. 
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
 *                      h) output the particle spectrum and particle positions
 *                         in the vertical
 *                      i) lwc depleted above 8 km due to conversion to ice
 *
 *             Ice multiplication design:
 *                      a) the Hallett-Mossop zone is divided up into 1 deg C
 *                         sub-zones; 
 *                      b) splinters are produced in each of these sub-zones 
 *                         at the rate given by the formula
 *                         Nx = rime * Ng * H; rime being the amount of rime
 *                         gathered in the time interval, and H being the
 *                         rate of splinter production per mg of rime collected
 *                         (Other approaches will be tried later - 15/8/94.)
 * 
 *
 */

#include <strings.h>
#include <math.h>
#include <stdio.h>
#include "cdfhdr.h"

#define MAX1 10                       /* number of particle size categories */
#define MAX2 1200                     /* number of altitude values          */
#define MAX3 2000                     /* number of time steps               */
#define MAX5 2000                     /* number of H-M splinter groups      */
#define MAX6 2010                     /* number of original size categories */
                                      /* plus H-M splinter groups           */
#define MAX8 500                      /* number of points for plotting      */

/* Standard run - RATTH = 0.4, RATCT = 0.2, RATDEB = 0.1, RATDOWN = 0.1 
 *                TDEPTH = 2 km, TGAP = 1.2 km, no downdraughts (umax = 0),
 *                wmax = 5 m/s, wmin = 2 m/s, max at 7 km, zinit = 6.5
 */
 
#define TBASE 10.6                    /* temperature at cloud base          */
#define PBASE 705.                    /* pressure at cloud base             */
#define DBAR   20.                    /* mean cloud droplet diameter        */
#define RATTH  0.7                    /* L/La in thermal                    */
#define RATCT  0.3                    /* L/La in uppermost cloud top region */
#define RATDEB 0.1                    /* L/La in debris                     */
#define RATDOWN 0.1                   /* L/La in downdraught                */
#define WCTDEB -2.                    /* w in descending cloud top region   */
#define UFACT 1.0                     /* mult factor for umax               */
#define DDEPTH 1.0                    /* depth that downdraught descends    */
#define CTOP 11.0                     /* altitude of maximum cloud top      */
#define TDEPTH 3.0                    /* depth of thermal                   */
#define TGAP 2.0                      /* vertical distance between thermals */
#define TWID TDEPTH                   /* width of thermal incl downdraught  */
#define DOWN 0.4                      /* width of downdraught               */
#define ZCT TDEPTH / 20.               /* depth of overall cloud top region  */
#define ZCT1 ZCT / 2.                 /* depth of cloud top region 1        */
#define ZCT2 ZCT / 2.                 /* depth of cloud top region 2        */
#define GENCTOP CTOP - 0.5            /* height of general cloud top        */

#define HMRINIT 10.0e-6               /* initial r of HM splinter           */
#define HMFACT 5.0e7                  /* this is number per kg of rime      */

#define PFACT 5                       /* reduction in pts for plotting      */

#define P0 1013.
#define T0 288.
#define GAMMA 0.0065
#define RHOI 920
#define EPS 0.622
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
#define PI 3.1415

/* float diam[10] = { .05, .15, .25, .35, .5, .75, 1.1, 1.55, 2.1, 2.75} ; */
/* Mod 3/11/94 - no graupel to start with */
/* float diam[10] = { .05, .1, .15, .2, .25, .3, .35, .4, .45, .5}; */
/* float diam[10] = { .04, .08, .12, .16, .2, .24, .28, .32, .36, .4};*/
/* float diam[10] = { .002, .004, .006, .008, .01, .012, .014, .016, .018, .02};*/
/* Usual spectrum */
 float diam[10] = { .05, .1, .2, .3, .4, .5, .6, .7, .8, .9}; 
 float diamf[15] = { .05, .15, .25, .35, .5, .75, 1.1, 1.55, 2.1, 2.75,
                    3.45, 4.15, 4.85, 5.55, 6.25};
 float wid[15]  = { .1, .1, .1, .1, .2, .3, .4, .5, .6, .7,
                    .7, .7, .7, .7, .7};

/* alternative spectra:  */

/* float diam[20] = { .05, .1, .15, .2, .3, .4, 
 *                  .5, .6, .7, .8, .9, 1., 1.1, 1.2, 
 *                  1.4, 1.6, 1.8, 2.0, 2.2, 2.4 } ;
 */
 
/* float diam[20] = { 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3,
 *                  0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1. };
 */

static float lw[MAX2],db[MAX2];
static float vv[MAX2];                 /* lwc, dbar,vertical velocity        */
static float alt[MAX2],ps[MAX2];
static float tr[MAX2];                 /* altitude, pressure, tempr          */
static float conc[MAX1];               /* conc in each size category         */
static float radi[MAX1];               /* initial radius                     */
static float spec[MAX1];               /* d.v conc                           */
static float speci[MAX1];              /* initial spectrum                   */
static float tothm[MAX3];              /* tot no of hm pcles produced        */
static float cumtot[MAX3];             /* cumulative total of hm pcles       */
static float thtop1[MAX3],thbase1[MAX3],
      thtop2[MAX3],thbase2[MAX3];      /* top and bottom alt of thermals     */
static float cldtop[MAX3],
      downbase[MAX3];                  /* top and bottom of cloud top debris */
static float xx[MAX3],yy[MAX3];        /* d.v. for plotting                  */
static float diamp[MAX6][MAX8];        /* diameter for straight plotting     */
static float vtp[MAX6][MAX8];          /* terminal velocity for plotting     */
static float deltsp[MAX6][MAX8];       /* surface tempr excess for plotting  */
static float effp[MAX6][MAX8];         /* efficiency for plotting            */
static float timep[MAX6][MAX8];        /* time of trajs stored for plotting  */
static float rhop[MAX6][MAX8];         /* density of particle                */
static float zp[MAX6][MAX8];           /* altitude of each size              */
static float xp[MAX6][MAX8];           /* pos of pcle rel to centre of therm */
static float stime[MAX5];              /* start time for splinters           */
static float xhm[MAX5],zhm[MAX5];      /* initial position of splinter       */
static float nhm[MAX5];                /* concentration of splinters         */
static float wtht1,wthb1,wtht2,wthb2;  /* vels of top & bot of thermals      */
static float rad;                      /* radius of graupel particles (m)    */
static float wi,lwc;                   /* updraught, liquid water content    */
static float rhoa;                     /* density of air                     */
static float zinit;                    /* initial zltitude                   */
static float cf;                       /* vars used to calculate reflectivty */
static float pbase,tbase;              /* pres (mb) and temp (C) at c base   */
static float zbase,cdepth;             /* alt of base, and depth of cloud    */
static float time;                     /* time in minutes                    */
static float at,alwc;                  /* adiabatic temp and lwc             */
static float c1;                       /* offset for framing spectra         */
static float xinit,yinit;              /* initialisation pts for windows     */
static float x1,x2,wy1,wy2;            /* window coordinates                 */
static float maxup;                    /* max updraught                      */
static float wbase,zmax;               /* up at base, and alt of max up      */
static float umax;                     /* maximum horizontal velocity        */
static float slope,intcpt;             /* vars to calc. v vel profile        */
static float runtime;                  /* length of run in mins              */
static float es,esi;                   /* water and ice vapour pressure      */
static float nre,nsh,nnu;              /* Reynolds, Sherwood and Nusslet no  */
static float xmax;                     /* maximum value on x-axis            */
static float totnum;                   /* total number of primary ice        */
static float level;
static float level1,level2,level3;     /* level used for spectrum            */
static float hmrime;                   /* amount of rime collected in hm 1-5 */
static float zm8,zm3;                  /* altitude of -8 and -3 C levels     */

static int numt,num[MAX6];             /* number of time pts for plotting    */ 
static int hmi[MAX5];                  /* start pointer for splinters        */
static int hm,hmj;                     /* counters for splinter group number */
static int j;                          /* trajectory counter                 */
static int tt,pp;                      /* counters for plotting              */
static int nxtick;                     /* number of ticks on x-axis          */
static int topflag;                    /* flag set if base of thermal at top */
static int thb1flag;                   /* when 1st thermal reaches c.b.      */
static int tht2flag,thb2flag;          /* when 2nd therm reaches c.b.        */
static int post1,posb1,post2,posb2;    /* index for thermal velocities       */
static int splinter;                   /* flag set to 1 for splinters        */
static int atlevel1,atlevel2,atlevel3; /* flag set to 1 if level reached     */
static int totpts;                     /* number of points for plotting      */
static int c,hmflag;                   /* buffer and flag                    */

static char pname[LINE];
static char realdate[LINE];            /* array for today's date/time        */
static char lab1[LINE],lab2[LINE],
            lab3[LINE],lab4[LINE],
            lab5[LINE],lab6[LINE],
            lab6[LINE],lab7[LINE],
            lab8[LINE],lab9[LINE],
            lab10[LINE],lab11[LINE],
            lab12[LINE],lab13[LINE],
            lab14[LINE],lab15[LINE],
            lab16[LINE],lab17[LINE],
            lab18[LINE],lab19[LINE],
            lab20[LINE],lab21[LINE];   /* for labels                         */

extern char *optarg;                   /* stuff for .. */
extern int optind;                     /* ... getopt */

float ptz();
float ztp();                           /* altitude to pressure                */
float trev();                          /* adiabatic LWC and temp              */
float graupel();                       /* graupel growth model                */
float drag();                          /* drag coefficient and Reynolds no    */
float tsurf();                         /* surface temperature                 */
float vapour();                        /* vapour pressure                     */
float gr();                            /* diffusion growth of ice             */

static FILE *fdate;

main(argc,argv)
int argc;
char *argv[];
{
  float maxy,miny;                     /* max/min for axes                   */
  int i;                               /* loops: i time                      */
  int k;                               /* spectrum channel counter           */
  int ii,jj;                           /* d.v. for altitude and channel loop */
  int kk,kr,kc,nw;                     /* counters for plotting stuff        */
  int nticks;                          /* no of ticks for axes               */

/* get program name */
  sprintf(pname,"%s",argv[0]);
/* process arguments */
  hmflag = 0;
  while ((c = getopt(argc,argv,"h")) != -1) {
    if (c == 'h') hmflag = 1;
  }
  if (argc - optind != 5) {
    fprintf(stderr,"Usage: %s [-h] zi (km) wmax wbase (m/s) ",pname);
    fprintf(stderr," z@wmax (km), run_time (min) \n");
    exit(0);
  }

/* read in argument value */
  zinit = 1000. * atof(argv[optind + 0]);
  maxup = atof(argv[optind + 1]); 
  wbase = atof(argv[optind + 2]);
  zmax = atof(argv[optind + 3]);
  runtime = atof(argv[optind + 4]);

/* umax from continuity */
  umax = UFACT * maxup  / 2.;
  
/* zero arrays */
/*  for (j = 0; j < MAX6; j++)
 *   for (i = 0; i < MAX8; i++) 
 *     zp[j][i] = xp[j][i] = timep[j][i] = diamp[j][i] = rhop[j][i] = 0.;
 */
  for (i = 0; i < MAX3; i++)
    tothm[i] = 0.;

/* initial spectrum of primary particles */
/* the total concentration is 1.6 /L */
  totnum = 0;
  for (j = 0; j < MAX1; j++) {
    radi[j] = diam[j] / 2.;
/* two exponential fits - spectrum (10/8/94) */
    if (diam[j] <= 0.3) 
      conc[j] = pow(10,4.) * exp(-39.47 * diam[j]);
    else 
      conc[j] = pow(10,0.7) * exp(-1.3 * diam[j]);
    totnum += conc[j];

/* the following from Jim Dye's paper - see p 35 of book */
/*    conc[j] = 250. * exp(-3.00 * diam[j]);*/
  }

/* cloud base pressure and temperature */
  pbase = PBASE;
  tbase = TBASE;
  zbase = ptz(pbase);
  fprintf(stderr,"cloud base: p = %5.1f mb, T = %4.1f C, z = %3.1f km\n",pbase,tbase,zbase/1000.);
  cdepth = CTOP - zbase / 1000.;

/* slope and intercept of vertical velocity profile */
    slope = (maxup - wbase) / (zmax - zbase / 1000.);
    intcpt = wbase - slope * zbase / 1000.;

/* substance for profiles */
  for (ii = 0; ii < MAX2; ii++) {
    ps[ii] = pbase - ii / 2.;
    alt[ii] = ptz(ps[ii]);
/* liquid water content and temperature */
    trev(pbase,tbase,ps[ii],&at,&alwc);
    lw[ii] = alwc * 1.0e-3;
    tr[ii] = (float)at + 273.15;
    db[ii] = DBAR + (float)ii / 125.;
/* vertical velocity */
    if (alt[ii] < zbase)
      vv[ii] = 1.;
    else if (alt[ii] / 1000. <= zmax)
      vv[ii] = slope * alt[ii] / 1000. + intcpt;
    else
      vv[ii] = (maxup / ((CTOP+2.) - zmax)) * ((CTOP+2.) - alt[ii] / 1000.0);
  }

/* number of time points */
  numt = 1 + (int)(runtime * 60. / DELTIM);
  fprintf(stderr,"numt = %d\n",numt);

/* initial positions of thermals */
  thtop1[0] = zinit/1000. + 0.0;
  if (thtop1[0] > CTOP)
    thtop1[0] = CTOP;
  cldtop[0] = thtop1[0];
  thbase1[0] = thtop1[0] - TDEPTH;
  if (thbase1[0] < 0.)
    thbase1[0] = 0.;
  thtop2[0] = thbase1[0] - TGAP;
  if (thtop2[0] < 0.)
    thtop2[0] = 0.;
  thbase2[0] = thtop2[0] - TDEPTH;
  if (thbase2[0] < 0.)
    thbase2[0] = 0.;
  downbase[0] = BAD;

/* ascent rate of thermals (assumed to be half the updraught speed) */
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
  wtht1 = vv[post1]/2.;
  wthb1 = vv[posb1]/2.;
  wtht2 = vv[post2]/2.;
  wthb2 = vv[posb2]/2.;

/* positions of thermals (in km) with time */
  topflag = thb1flag = tht2flag = thb2flag =  0;
  for (i = 1; i < numt; i++) {
    thtop1[i] = thtop1[i-1] + (wtht1 * DELTIM)/1000.;
    if (thtop1[i] >= CTOP)
      thtop1[i] = CTOP;

    thbase1[i] = thtop1[i] - TDEPTH; 
    if (thbase1[i] > 0.)
      thbase1[i] = thbase1[i-1] + (wthb1 * DELTIM)/1000.;

    if (tht2flag != 1) {
      thtop2[i] = thbase1[i] - TGAP;
      if (thtop2[i] > 0.)
        tht2flag = 1;
      if (thtop2[i] < 0.)
        thtop2[i] = 0.;
    }
    else 
      thtop2[i] = thtop2[i-1] + (wtht2 * DELTIM)/1000.;

    thbase2[i] = thtop2[i] - TDEPTH;
    if (thbase2[i] > 0.)
      thbase2[i] = thbase2[i-1] + (wthb2 * DELTIM)/1000.;

/*    fprintf(stderr,"i=%d, thtop1=%f, thbase1=%f, thtop2=%f, thbase2=%f\n",i,thtop1[i],thbase1[i],thtop2[i],thbase2[i]); */


/*    if (thbase1[i] >= CTOP - TDEPTH / 2. && topflag == 0) { */
    if (thtop1[i] >= CTOP && thbase1[i] >= (CTOP - TDEPTH/4.) && topflag == 0) {
      topflag = 1;
      thtop1[i] = thtop2[i];
      thbase1[i] = thbase2[i];
      thtop2[i] = thbase1[i] - TGAP;
      thbase2[i] = thtop2[i] - TDEPTH;
      thb1flag = tht2flag = thb2flag = 0;
      cldtop[i-1] = CTOP;
    }
    if (topflag == 1) {
/* make cloud top descend to the general cloud top region */
      cldtop[i] = cldtop[i-1] + (WCTDEB * DELTIM)/1000.;
      if (cldtop[i] <= GENCTOP) 
        cldtop[i] = GENCTOP;
      downbase[i] = cldtop[i] - TDEPTH / 4.;
      if (downbase[i] <= thtop1[i])
        downbase[i] = thtop1[i];
      if (cldtop[i] < thtop1[i])
        topflag = 0;
    }
    else {
      cldtop[i] = thtop1[i];
      downbase[i] = BAD;
    }
    post1 = posb1 = post2 = posb2 = 0;
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

/* altitude of H-M zone - note temperature is the same in each zone except 
 * for the downdraughts.
 */
        if (tr[ii] < 266.75 && tr[ii] > 266.5)
          zm8 = alt[ii] / 1000.;
        if (tr[ii] < 272.25 && tr[ii] > 272.0)
          zm3 = alt[ii] / 1000.;
      }
    }
    wtht1 = vv[post1]/2.;
    wthb1 = vv[posb1]/2.;
    wtht2 = vv[post2]/2.;
    wthb2 = vv[posb2]/2.;

  }

/* growth of all particles handled in subroutine graupel 
 * Note particles produced in the H-M zones in the subroutine
 * graupel are splinters.  Their growth is dealt with after the 
 * initial set of trajectories, by number */
  wi = 0.;
  hm = 0;
  hmj = 0;
  hmrime = 0.;
  splinter = 0;
  for (j = 0; j < MAX1; j++) 
    graupel();

  if (hmflag == 0) 
    hm = 1;

  if (hm > 0) {
    splinter = 1;
    for (hmj = 0; hmj < hm-1; hmj++) {
      graupel();
    }
  }

/* total number of h-m particles produced versus time / number of ice pcles
 * present at the beginning - this gives value of f in our paper */

  for (i = 0; i < numt; i++) 
    cumtot[i] = 1;

  for (i = 0; i < numt; i++){
    if (i > 0)
      cumtot[i] = cumtot[i-1] + tothm[i] / totnum;
  }

/* plotting */
  fprintf(stderr,"Plotting\n");
  gopen();
  badset(9999.);
  gclear();

/* height plot */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,90.,7,"t (min)",3.,12.,10,"z (km)"); 
  for (j = 0; j < MAX1+hm; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = (timep[j][i] > 0.) ? timep[j][i] : BAD;
      yy[i] = (zp[j][i] > 0.) ? zp[j][i] : BAD;
    }
    if (j >= MAX1)
      line(1,5,num[j],xx,yy); 
    else
      line(1,13,num[j],xx,yy); 
  }
  xx[0] = 0.;
  for (i = 1; i < numt; i++) {
    xx[i] = xx[i-1] + DELTIM/60.;
    if (thtop1[i-1] < BAD && thtop1[i] < thtop1[i-1])
      thtop1[i] = BAD;
    if (thtop2[i-1] < BAD && thtop2[i] < thtop2[i-1])
      thtop2[i] = BAD;
    if (thbase1[i-1] < BAD && thbase1[i] < thbase1[i-1])
      thbase1[i] = BAD;
    if (thbase2[i-1] < BAD && thbase2[i] < thbase2[i-1])
      thbase2[i] = BAD;
  }
  line(1,1,numt,xx,cldtop);
  line(1,1,numt,xx,thtop1);
  line(1,1,numt,xx,thbase1); 
  line(1,1,numt,xx,downbase);
  line(1,1,numt,xx,thtop2);
  line(1,1,numt,xx,thbase2);
/* H-M zone */
  xx[0] = 0.;
  xx[1] = xx[numt-2];
  yy[0] = zm8;
  yy[1] = yy[0];
  line(1,2,2,xx,yy);
  yy[0] = zm3;
  yy[1] = yy[0];
  line(1,2,2,xx,yy);

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
  label(2,0.01,0.9,lab1);
  sprintf(lab2,"Cloud base: T = %4.1f C; P = %5.1f mb",TBASE,PBASE);
  label(2,0.01,0.86,lab2);
  sprintf(lab3,"TDEPTH = %3.1f km",TDEPTH);
  label(2,0.01,0.8,lab3);
  sprintf(lab4,"TGAP = %3.1f km",TGAP);
  label(2,0.5,0.8,lab4);
  sprintf(lab5,"Lmax = %4.2f Lad",RATTH);
  label(2,0.01,0.76,lab5);
  sprintf(lab6,"wmax = %4.1f m/s",maxup);
  label(2,0.5,0.76,lab6);
  sprintf(lab7,"Lct = %4.2f Lad",RATCT);
  label(2,0.01,0.72,lab7);
  sprintf(lab8,"Ldeb = %4.2f Lad",RATDEB);
  label(2,0.5,0.72,lab8);
  sprintf(lab11,"largest initial diameter = %3.1f mm",diam[9]);
  label(2,0.01,0.68,lab11);
  if (umax > 0) {
    sprintf(lab9,"umax = %4.1f m/s",umax);
    label(2,0.01,0.64,lab9);
    sprintf(lab10,"DDEPTH = %4.1f km",DDEPTH);
    label(2,0.5,0.64,lab10);
    sprintf(lab18,"Ldown = %4.2f Lad",RATDOWN);
    label(2,0.01,0.60,lab18);
  }

  gpause();
  gclear();

/* plot without labels for paper */
/* height plot */
  window(1,1.,9.,0.,8.);
  axes(1,0.,90.,7,"t (min)",3.,12.,10,"z (km)"); 
/*  axes(1,0.,60.,7,"t (min)",3.,9.,7,"z (km)"); */
  for (j = 0; j < MAX1+hm; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = (timep[j][i] > 0.) ? timep[j][i] : BAD;
      yy[i] = (zp[j][i] > 0.) ? zp[j][i] : BAD;
    }
    line(1,1,num[j],xx,yy);
  }
  xx[0] = 0.;
  for (i = 1; i < numt; i++) {
    xx[i] = xx[i-1] + DELTIM/60.;
    if (thtop1[i-1] < BAD && thtop1[i] < thtop1[i-1])
      thtop1[i] = BAD;
    if (thtop2[i-1] < BAD && thtop2[i] < thtop2[i-1])
      thtop2[i] = BAD;
    if (thbase1[i-1] < BAD && thbase1[i] < thbase1[i-1])
      thbase1[i] = BAD;
    if (thbase2[i-1] < BAD && thbase2[i] < thbase2[i-1])
      thbase2[i] = BAD;
  }
  line(1,1,numt,xx,cldtop);
  line(1,1,numt,xx,thtop1);
  line(1,2,numt,xx,thbase1);
  line(1,2,numt,xx,downbase);
  line(1,3,numt,xx,thtop2);
  line(1,4,numt,xx,thbase2);
/* H-M zone */
  xx[0] = 0.;
  xx[1] = xx[numt-2];
  yy[0] = zm8;
  yy[1] = yy[0];
  line(1,2,2,xx,yy);
  yy[0] = zm3;
  yy[1] = yy[0];
  line(1,2,2,xx,yy);

  gpause();
  gclear();

/* efficiency plot */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,8.,9,"diam (m)",0.,1.,6,"Coll. Eff. ");
  for (j = 0; j < MAX1+hm-1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = (diamp[j][i] > 0.) ? diamp[j][i] : BAD;
      yy[i] = (effp[j][i] > 0.) ? effp[j][i] : BAD;
    }
    line(1,1,num[j],xx,yy);
  }

  gpause();
  gclear();

/* surface temperature excess */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,8.,9,"diam (m)",-5.,5.,11,"Ts-Ta (deg)");
  for (j = 0; j < MAX1+hm-1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = (diamp[j][i] > 0.) ? diamp[j][i] : BAD;
      yy[i] = (deltsp[j][i] > -5.) ? deltsp[j][i] : BAD;
    }
    line(1,1,num[j],xx,yy);
  }

  gpause();
  gclear();

/* multiplication factor, f vs time. f is total no of H-M pcles produced */
/* divided by the no of primary particles at t= 0. */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  xx[0] = yy[0] = 0.;
  for (i = 1; i < numt; i++) {
    xx[i] = xx[i-1] + DELTIM/60.;
    if (xx[i] > 9.95 && xx[i] < 10.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab12,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 19.95 && xx[i] < 20.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab13,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 29.95 && xx[i] < 30.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab14,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 39.95 && xx[i] < 40.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab15,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 49.95 && xx[i] < 50.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab16,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 59.95 && xx[i] < 60.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab17,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 69.95 && xx[i] < 70.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab19,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 79.95 && xx[i] < 80.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab20,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    if (xx[i] > 89.95 && xx[i] < 90.05) {
      fprintf(stderr,"Time = %4.1f min, f = %5.2e\n",xx[i],cumtot[i]);
      sprintf(lab21,"Time = %4.1f min ... f = %5.2e",xx[i],cumtot[i]);
    }
    yy[i] = (cumtot[i] > 0) ? log10(cumtot[i]) : BAD;
  }
  scale(numt,1.0,yy,&maxy,&miny);
  nticks = maxy - miny + 1;
  axes(1,0.,90.,7,"t (min)",miny,maxy,nticks,"log f");
  line(1,1,numt,xx,yy);
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.86,lab2);
  label(2,0.01,0.8,lab3);
  label(2,0.5,0.8,lab4);
  label(2,0.01,0.76,lab5);
  label(2,0.5,0.76,lab6);
  label(2,0.01,0.72,lab7);
  label(2,0.5,0.72,lab8);
  label(2,0.01,0.68,lab11);
  if (umax > 0) {
    label(2,0.01,0.64,lab9);
    label(2,0.5,0.64,lab10);
    label(2,0.01,0.60,lab18);
  }
  label(2,0.01,0.54,lab12);
  label(2,0.01,0.50,lab13);
  label(2,0.01,0.46,lab14);
  label(2,0.01,0.42,lab15);
  label(2,0.01,0.38,lab16);
  label(2,0.01,0.34,lab17);
  label(2,0.01,0.30,lab19);
  label(2,0.01,0.26,lab20);
  label(2,0.01,0.22,lab21);

  gpause();
  gclear();

/* plot without labels for paper */
  window(1,1.,9.,0.,8.);
  xx[0] = yy[0] = 0.;
  for (i = 1; i < numt; i++) {
    xx[i] = xx[i-1] + DELTIM/60.;
    yy[i] = (cumtot[i] > 0) ? log10(cumtot[i]) : BAD;
  }
  axes(1,0.,90.,7,"t (min)",miny,maxy,nticks,"log f");
  line(1,1,numt,xx,yy);

  gpause();
  gclear();

/* initial spectrum */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.0,5.0,6,"D (mm)",-1.0,6.0,8,"N (m^-3 mm^-1)");
  for (j = 0; j < MAX1; j++) {
    speci[j] = (conc[j] > 0.) ? log10(conc[j]) : BAD;
    mark(1,3,diam[j],speci[j],0.2);
  }
  line(1,1,MAX1,diam,speci);
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.8,lab2);
  label(2,0.01,0.76,lab3);
  label(2,0.01,0.72,lab4);
  label(2,0.01,0.68,lab5);
  gpause();
  gclear();

/* vertical velocity with height */
/* window(1,1.,9.,0.,8.);
 * window(2,9.,15.,0.,8.);
 * axes(1,0.,20.,6,"vert. vel. (m/s)",3.,15.,7,"Height (km)");
 * for (ii = 0; ii < MAX2; ii++)
 *   yy[ii] = alt[ii] / 1000.;
 * line(1,1,MAX2,vv,yy); 
 * label(2,0.01,0.9,lab1);
 * label(2,0.01,0.8,lab2);
 * label(2,0.01,0.76,lab3);
 * label(2,0.01,0.72,lab4);
 * label(2,0.01,0.68,lab5);
 * gpause();
 * gclear();
 */
/* trajectories */
/* window(1,1.,9.,0.,8.);
 * window(2,9.,15.,0.,8.);
 * axes(1,0.,TWID+1.,(int)TWID+2,"x (km)",3.,15.,7,"Height (km)");
 * for (j = 0; j < MAX1+hm-1; j++) {
 *   for (i = 0; i < num[j]; i++) {
 *     xx[i] = (xp[j][i] > 0.) ? xp[j][i] : BAD;
 *     yy[i] = (zp[j][i] > 0.) ? zp[j][i] : BAD;
 *   }
 *   line(1,1,num[j],xx,yy); 
 * }
 * label(2,0.01,0.9,lab1);
 * label(2,0.01,0.8,lab2);
 * label(2,0.01,0.76,lab3);
 * label(2,0.01,0.72,lab4);
 * label(2,0.01,0.68,lab5);
 * gpause();
 * gclear();
*/
/* diameter vs height */
  window(1,1.,9.,0.,8.);
  window(2,9.,15.,0.,8.);
  axes(1,0.,6.,7,"diam (mm)",3.,13.,5,"Height (km)");
  for (j = 0; j < MAX1+hm-1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = (diamp[j][i] > 0.) ? diamp[j][i] : BAD;
      yy[i] = (zp[j][i] > 0.) ? zp[j][i] : BAD;
    }
    line(1,1,num[j],xx,yy);
  }
  label(2,0.01,0.9,lab1);
  label(2,0.01,0.86,lab2);
  label(2,0.01,0.8,lab3);
  label(2,0.5,0.8,lab4);
  label(2,0.01,0.76,lab5);
  label(2,0.5,0.76,lab6);
  label(2,0.01,0.72,lab7);
  label(2,0.5,0.72,lab8);
  label(2,0.01,0.68,lab9);
  label(2,0.5,0.68,lab10);
  label(2,0.01,0.64,lab11);
  gpause();
  gclear();

/* diameter vs height for paper */
  window(1,1.,9.,0.,8.);
  axes(1,0.,6.,7,"diam (mm)",3.,12.,10,"Height (km)");
  for (j = 0; j < MAX1+hm-1; j++) {
    for (i = 0; i < num[j]; i++) {
      xx[i] = (diamp[j][i] > 0.) ? diamp[j][i] : BAD;
      yy[i] = (zp[j][i] > 0.) ? zp[j][i] : BAD;
    }
    line(1,1,num[j],xx,yy);
  }
  gpause();
  gclear();

/* terminal velocity vs height */
/*  window(1,1.,9.,0.,8.);
 * window(2,9.,15.,0.,8.);
 * axes(1,0.,12.,7,"vt (m/s)",3.,9.,7,"Height (km)");
 * for (j = 0; j < MAX1+hm-1; j++) {
 *   for (i = 0; i < num[j]; i++) {
 *     xx[i] = (vtp[j][i] > 0.) ? vtp[j][i] : BAD;
 *     yy[i] = (zp[j][i] > 0.) ? zp[j][i] : BAD;
 *   }
 *   line(1,1,num[j],xx,yy);
 * }
 * label(2,0.01,0.9,lab1);
 * label(2,0.01,0.8,lab2);
 * label(2,0.01,0.76,lab3);
 * label(2,0.01,0.72,lab4);
 * label(2,0.01,0.68,lab5);
 * gpause();
 * gclear();
 */
/* diameter vs time */
/*  window(1,1.,9.,0.,8.);
 * window(2,9.,15.,0.,8.);
 * axes(1,0.,60.,7,"time (min)",0.,7.,8,"graupel diam (mm)");
 * for (j = 0; j < MAX1+hm; j++) {
 *   for (i = 0; i < num[j]; i++) {
 *     xx[i] = (timep[j][i] > 0.) ? timep[j][i] : BAD;
 *     yy[i] = (diamp[j][i] > 0.) ? diamp[j][i] : BAD;
 *   }
 *   line(1,1,num[j],xx,yy);
 * }
 * label(2,0.01,0.9,lab1);
 * label(2,0.01,0.86,lab2);
 * label(2,0.01,0.8,lab3);
 * label(2,0.5,0.8,lab4);
 * label(2,0.01,0.76,lab5);
 * label(2,0.5,0.76,lab6);
 * label(2,0.01,0.72,lab7);
 * label(2,0.5,0.72,lab8);
 * label(2,0.01,0.68,lab18);
 * label(2,0.01,0.64,lab9);
 * label(2,0.5,0.64,lab10);
 * label(2,0.01,0.60,lab11);

 * gpause();
 * gclear();
*/

/* density vs diameter */
/*  window(1,1.,9.,0.,8.);
 * window(2,9.,15.,0.,8.);
 * axes(1,0.,10.,6,"graupel diam (mm)",0.,1.,11,"density (g/cm^3)");
 * for (j = 0; j < MAX1+hm-1; j++) {
 *   for (i = 0; i < num[j]; i++) {
 *     xx[i] = (diamp[j][i] > 0.) ? diamp[j][i] : BAD;
 *     yy[i] = (rhop[j][i] > 0.) ? rhop[j][i]/1000. : BAD;
 *   }
 *   line(1,1,num[j],xx,yy);
 * }
 * label(2,0.01,0.9,lab1);
 * label(2,0.01,0.8,lab2);
 * label(2,0.01,0.76,lab3);
 * label(2,0.01,0.72,lab4);
 * label(2,0.01,0.68,lab5);
 * gpause();
 * gclear();
 */

  gclose();
}

float graupel()
{

  float z,x,pres,temp,tempc,dbar;      /* current values                     */
  float zold,xold;                     /* old values of z & x                */
  float zkm;                           /* height in km                       */
  float zdown;                         /* starting height in downdraught     */
  float hw;                            /* horizontal wind                    */
  float lwc7;                          /* lwc at 7 km                        */
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
  float w,w2,w3,w4;                    /* arguments used in Reynolds no.     */
  float wpcle;                         /* vertical velocity of particle      */
  float nx;                            /* conc of splinters in HM process    */

  int i,k,ii,indx;                     /* counters                           */
  int tflag;                           /* flag for temperature               */
  int zflag;                           /* flag for z = 8 km                  */
  int one;                             /* flag for for first round of loop   */
  int pos;                             /* pointer to current height          */
  int downflag;                        /* flag set to 1 if in downdraught    */
  int graup;                           /* test variable                      */
  int hmzone1,hmzone2,hmzone3,
      hmzone4,hmzone5;                 /* flag for entering HM zones         */
  int istart;                          /* starting value for i               */

/* set counters and things */
  if (splinter == 1)
    j = MAX1 + hmj;
  tflag = 0;
  one = 0;
  pp = 0;
  tt = 0;
  hw = 0.;
  wi = 0.2;
  indx = 0;
  downflag = 0;
  topflag = 0;
  graup = 0;
  zflag = 0;
  hmrime = 0;
  hmzone1 = hmzone2 = hmzone3 = hmzone4 = hmzone5 = 0;

/* growth loop */
  if (splinter == 1)
    istart = hmi[hmj];
  else
    istart = 0;
  for (i = istart;i < numt;i++) {
/* check environmental conditions */
/* first round - get initial conditions */
    if (one == 0) {
/* different initial conditions if this is for a splinter */
      if (splinter > 0) {
        z = 1000 * zhm[hmj];
        x = xhm[hmj];
        time = stime[hmj];
      }
      else {
        z = zinit;
        x = 0.;
        time = 0.;
      }
      zold = z;
      xold = x;
    }
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
/* particle in thermal ? */
      if ((zkm <= thtop1[i]-ZCT && zkm >= thbase1[i]) || 
          (zkm <= thtop2[i]-ZCT && zkm >= thbase2[i])) {
        temp = tr[pos] - 1.;
        lwc = RATTH * lw[pos];
        dbar = db[pos];
        wi = vv[pos];
      }

/* in cloud top region (top of thermal 1) */
      if (zkm <= thtop1[i] && zkm >= thtop1[i]-ZCT) {
        temp = tr[pos] - 1.;
        lwc = RATCT * lw[pos];
        dbar = db[pos];
        hw = umax;
/* work out smooth vertical velocity in cloud top regions */
        wi = vv[pos]-vv[pos]*(zkm-(thtop1[i]-ZCT))/(2.*ZCT); 
      }

/* out of thermal, into debris under thermal (over thermal if at start) */
      if ((zkm > thtop1[i] || zkm < thbase1[i]) &&
          (zkm > thtop2[i] || zkm < thbase2[i]) &&
          (zkm < downbase[i]) || (zkm > thtop1[i] && zkm > thtop2[i])) {
        temp = tr[pos] - 1.;
        lwc = RATDEB * lw[pos];
        dbar = db[pos];
        hw = 0.;
        wi = 0.0;
      }

/* in descending thermal remnants */
      if (zkm > thtop1[i] && zkm <= cldtop[i] && zkm >= downbase[i]) {
        temp = tr[pos] - 2.;
        lwc = RATDEB * lw[pos];
        dbar = db[pos];
        hw = 0.;
        if (cldtop[i] > GENCTOP) 
          wi = WCTDEB;
        else
          wi = 0.;
      }

/* horizontal position in km */
      if (x >= TWID) {
        temp = tr[pos] - 2.;
        lwc = RATDOWN * lw[pos];
        dbar = db[pos];
        hw = 0.;
        wi = -5.;
        downflag = 1;
        zdown = zkm;
/* reduce number of pcles mixing from downdraught back into cloud */
/*        if (splinter == 0)
 *         conc[j] /= 3.;
 *       else
 *         tothm[i] /= 3.;
 */
      }
    }
    else {
      temp = tr[pos] - 2.;
      lwc = RATDOWN * lw[pos];
      dbar = db[pos];
      hw = 0.;
      wi = -5.;
/* reduce number of pcles mixing from downdraught back into cloud */
/*        if (splinter == 0)
 *         conc[j] /= 3.;
 *       else
 *         tothm[i] /= 3.;
 */
/* make downdraughts no deeper than DDEPTH km */
      if (fabs(zdown - zkm) >= DDEPTH) {
        downflag = 0;
        xold = 0;
        x = 0.;
        zdown = 0.;
      }
    }

/* overshoot cloud top? */
      if (zkm >= CTOP || zkm > cldtop[i])
        zkm = cldtop[i];

/* deplete the lwc if above 8 km (due to conversion to ice) */
    if (zkm >= 7) {
      if (zflag == 0) {
        lwc7 = lwc;
        zflag = 1;
      }
      if (zflag  > 0)
        lwc = lwc7 - (zkm - 7.) * lwc7 / (10. - 7.);
      if (zkm >= 10.)
        lwc = 0.;
    }

/* density of air */
    rhoa = pres / (RD * temp); 
/* kinematic viscosity */
    nu = ETA / rhoa;
    tempc = temp - 273.15;

    if (one == 0) {
/* initial radius in m and density kg m^-3 */
      if (splinter < 1)
        rad = radi[j] * 1.0e-3;
      else
        rad = HMRINIT;
/*      diamp[j][i] = rad * 2.0e3; */
      rhog = 900.;
/*      rhop[j][i] = rhog; */
/* initial mass */
      mass = 4. * PI * pow(rad,3.) * rhog / 3.;

/* calculate vt and delmr for a 300 um diameter particle */
/* Assume density of 100 g/cm^3 just now as a quick 
 * fix - 7/7/94 and use simple formula for kapb */
      drag(0.15e-3,100.,rhoa,&cd);
      if (rad*2.0e3 < 0.2) {
/*        vt300 = 8. * 0.15 * 1.0e-3 * 100. * GRAV / (3. * rhoa * cd); */
/*        vt300 = pow(vt300,0.5); */
        vt300 = 0.6 * 0.3;
/*        dmr300 = PI * pow(0.15e-3,2.) * vt300 * 0.5 * 1.0e-3 * DELTIM; */
/*        dr300 = dmr300 / (4. * PI * pow(0.15e-3,2.) * 100.); */
/* force dr300 to slightly larger than largest value of dr by diffusion */
        dr300 = 2.0e-6;
      }
    }

/* drag coefficient */
    drag(rad,rhog,rhoa,&cd);

/* terminal velocity is taken as a function of temperature according to
 * the graph of Fukuta et al. 1982, Chicago conference.
 * Smallest terminal velocity of particles between -6.5 and -4 and -16 and -14:
 * graph taken for filled-in stellars in fig 10-33 of P&K (Curve found
 * using points read-off graph and fitted using Maple) 
 * Middle vt for -14 < T <= -12, -8 < T <= =6.5 and T > -4:
 * vt = 40*D
 * High vt for -12 < T <= -8: 
 * vt = 51*D
 * Largest vt for T <= -16: 
 * vt = 60*D
 */

/* If initial diam is greater than 200 um, go straight to vt for graupel */
    dmm = rad*2.0e3;
/* PROBLEMS HERE? */
/*    if (dmm <= 0.30 || diam[j] <= 0.2) { */
    if (dmm <= 0.30) {
      if (dmm <= 0.2) {
        if ((tempc > -6.5 && tempc <= -4.) || (tempc > -16 && tempc <= -14.))
          vt = 100. * (0.395 * dmm - 0.177 * pow(dmm,2.) +
                0.073 * pow(dmm,3.) - 0.0153 * pow(dmm,4.)); 
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

/*    if(dmm > 0.1 && dmm < 0.4)*/
/* fprintf(stderr,"dmm = %5.3f, T = %4.2f, vt = %6.4f\n",dmm,temp,vt); */

/* On first round, use temp + 2.4 for surface temp */
    if (one == 0) {
      one = 1;
      ts = temp + 2.4;
    }

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
/*   delmd = (dmm > 0.2) ? 2. * PI * rad * D * nsh * (re - rvs) * DELTIM : 0.;*/
/* use lab values for vapour growth */
    delrd = 1.0e-6 * gr(tempc);

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
        if (kapb >= 0.2e-10) 
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
/*      delrd = fabs(delmd) / (4. * PI * rad * rad * RHOI); */
/*      if (delmd < 0.) delrd = -delrd; */
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

/* particle in one of Hallett-Mossop zones? Store x,z position, start splinter
 * with a diameter of 20 um, give splinter a number, and calculate the
 * concentration of splinters produced.  Note, only one calculation
 * per zone; the zhm is calculated to be at the centre of the zone. 
 * Assume temp is close to the temp at the top of the sub-zone and calculate 
 * the mid-point using the current altitude and 6.7 deg -> 1 km 
 * The value of i (which gives the time) is stored for each new splinter 
 * group */
    
    if (hmflag == 1) {
    if (rad*2e3 >= 0.3 && tempc >= -9.) {
/*      if (splinter < 1) 
 *        nx = PI * pow(rad,2.) * vt * lwc * conc[j] * HMFACT * (1000./(6.7 * 
 *            fabs(wi)));
 *      else
 *        nx = PI * pow(rad,2.) * vt * lwc * nhm[hmj] * HMFACT * (1000./(6.7 * 
 *             fabs(wi)));
 */
      if (tempc >= -8. && tempc < -7.) {
        hmrime += delmr;
        hmzone1 = 1;
      }
/* Particle has been in zone 1 - now calculate the number of particles */
      if (hmzone1 == 1 && (tempc < -8. || tempc >= -7.)) {
        if (splinter == 0)
          nx = conc[j] * HMFACT * hmrime;
        else
          nx = nhm[hmj] * HMFACT * hmrime;
        nhm[hm] = nx;
        tothm[i] += nx;
        xhm[hm] = x;
        zhm[hm] = zkm - 0.5 / 6.7;
        stime[hm] = time;
        hmi[hm] = i;
        hm++;
        hmrime = 0.;
        hmzone1 = 0;
      }

      if (tempc >= -7. && tempc < -6.) {
        hmrime += delmr;
        hmzone2 = 1;
      }
/* Particle has been in zone 2 - now calculate the concentatrion */
      if (hmzone2 == 1 && (tempc < -7. || tempc >= -6.)) {
        if (splinter == 0)
          nx = conc[j] * HMFACT * hmrime;
        else
          nx = nhm[hmj] * HMFACT * hmrime;
        nhm[hm] = nx;
        tothm[i] += nx;
        xhm[hm] = x;
        zhm[hm] = zkm - 0.5 / 6.7;
        stime[hm] = time;
        hmi[hm] = i;
        hm++;
        hmrime = 0.;
        hmzone2 = 0;
      }

      if (tempc >= -6. && tempc < -5.) {
        hmrime += delmr;
        hmzone3 = 1;
      }
/* Particle has been in zone 3 - now calculate the concentatrion */
      if (hmzone3 == 1 && (tempc < -6. || tempc >= -5.)) {
        if (splinter == 0)
          nx = conc[j] * HMFACT * hmrime;
        else
          nx = nhm[hmj] * HMFACT * hmrime;
        nhm[hm] = nx;
        tothm[i] += nx;
        xhm[hm] = x;
        zhm[hm] = zkm - 0.5 / 6.7;
        stime[hm] = time;
        hmi[hm] = i;
        hm++;
        hmrime = 0.;
        hmzone3 = 0;
      }

      if (tempc >= -5. && tempc < -4.) {
        hmrime += delmr;
        hmzone4 = 1;
      }
/* Particle has been in zone 4 - now calculate the concentatrion */
      if (hmzone4 == 1 && (tempc < -5. || tempc >= -4.)) {
        if (splinter == 0)
          nx = conc[j] * HMFACT * hmrime;
        else
          nx = nhm[hmj] * HMFACT * hmrime;
        nhm[hm] = nx;
        tothm[i] += nx;
        xhm[hm] = x;
        zhm[hm] = zkm - 0.5 / 6.7;
        stime[hm] = time;
        hmi[hm] = i;
        hm++;
        hmrime = 0.;
        hmzone4 = 0;
      }

      if (tempc >= -4. && tempc < -3.) {
        hmrime += delmr;
        hmzone5 = 1;
      }
/* Particle has been in zone 5 - now calculate the concentatrion */
      if (hmzone5 == 1 && (tempc < -4. || tempc >= -3.)) {
        if (splinter == 0)
          nx = conc[j] * HMFACT * hmrime;
        else
          nx = nhm[hmj] * HMFACT * hmrime;
        nhm[hm] = nx;
        tothm[i] += nx;
        xhm[hm] = x;
        zhm[hm] = zkm - 0.5 / 6.7;
        stime[hm] = time;
        hmi[hm] = i;
        hm++;
        hmrime = 0.;
        hmzone5 = 0;
      }
    }
    }

/* for plotting (in main program) */
    pp++;
    if (pp == PFACT) {
      diamp[j][tt] = rad * 2.e3;
      rhop[j][tt] = rhog;
      timep[j][tt] = time;
      zp[j][tt] = zkm;
      xp[j][tt] = x;
      vtp[j][tt] = vt;
      effp[j][tt] = kapb / (PI * pow(rad,2.) * vt);
      deltsp[j][tt] = ts - temp;
      tt++;
      pp = 0;
    }

    time = time + DELTIM / 60.;

/* particle velocity: vt is positive down */
    wpcle = wi - vt;

/* vertical and horizontal position */
    z = zold + wpcle * DELTIM;
    x = xold + hw * DELTIM / 1000.;
      
/* if  time > run_time, env temp is warmer than 0, or z < zbase, then break */
    if (time >= runtime || temp > 273.15 || z <= zbase) {
      num[j] = tt - 1;
      break; 
    }
/* density of air */
    rhoa = pres / (RD * temp); 
    zold = z;
    xold = x;

/* back for more growth  --- loop on i */
    nx = 0;
    indx++;
  }
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
  if (tempr < -21.5) return(0.3);
}
