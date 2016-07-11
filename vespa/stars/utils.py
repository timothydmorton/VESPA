from __future__ import print_function,division

import os,os.path
import pkg_resources
import logging

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import numpy as np
    import numpy.random as rand
else:
    on_rtd = True
    np, rand = (None, None)

try:
    from simpledist.distributions import KDE_Distribution
except ImportError:
    logging.warning('simpledist not available.')
    KDE_Distribution = None

if not on_rtd:
    from astropy.units import Quantity
else:
    Quantity = None

if not on_rtd:
    DATAFOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    RAGHAVAN_PERS = np.recfromtxt('{}/raghavan_periods.dat'.format(DATAFOLDER))
    RAGHAVAN_LOGPERS = np.log10(RAGHAVAN_PERS.f1[RAGHAVAN_PERS.f0 == b'Y'])
    RAGHAVAN_BINPERS = RAGHAVAN_PERS.f1[RAGHAVAN_PERS.f0 == b'Y']
    RAGHAVAN_BINPERKDE = KDE_Distribution(RAGHAVAN_BINPERS,adaptive=False)
    RAGHAVAN_LOGPERKDE = KDE_Distribution(RAGHAVAN_LOGPERS,adaptive=False)

    #from Multiple Star Catalog
    MSC_TRIPDATA = np.recfromtxt('{}/multiple_pecc.txt'.format(DATAFOLDER),names=True)
    MSC_TRIPLEPERS = MSC_TRIPDATA.P
    MSC_TRIPPERKDE = KDE_Distribution(MSC_TRIPLEPERS,adaptive=False)
    MSC_TRIPLOGPERKDE = KDE_Distribution(np.log10(MSC_TRIPLEPERS),adaptive=False)
else:
    DATAFOLDER = None
    RAGHAVAN_PERS, RAGHAVAN_LOGPERS, RAGHAVAN_BINPERS = (None, None, None)
    RAGHAVAN_BINPERKDE, RAGHAVAN_LOGPERKDE = (None, None)
    MSC_TRIPDATA, MSC_TRIPLEPERS, MSC_TRIPPERKDE, MSC_TRIPLOGPERKDE = (None, None,
                                                                       None, None)



def randpos_in_circle(n,rad,return_rad=False):
    x = rand.random(n)*2*rad - rad
    y = rand.random(n)*2*rad - rad
    mask = (x**2 + y**2 > rad**2)
    nw = mask.sum()
    while nw > 0:
        x[mask] = rand.random(nw)*2*rad-rad
        y[mask] = rand.random(nw)*2*rad-rad
        mask = (x**2 + y**2 > rad**2)
        nw = mask.sum()
    if return_rad:
        return np.sqrt(x**2 + y**2)
    else:
        return x,y

def draw_pers_eccs(n,**kwargs):
    """
    Draw random periods and eccentricities according to empirical survey data.
    """
    pers = draw_raghavan_periods(n)
    eccs = draw_eccs(n,pers,**kwargs)
    return pers,eccs

def flat_massratio(n, qmin=0.1, qmax=1.):
    return rand.uniform(size=n)*(qmax - qmin) + qmin

def flat_massratio_fn(qmin=0.1,qmax=1.):
    def fn(n):
        return rand.uniform(size=n)*(qmax - qmin) + qmin
    return fn

def draw_raghavan_periods(n):
    """
    Draw orbital periods according to Raghavan (2010)
    """
    logps = RAGHAVAN_LOGPERKDE.resample(n)
    return 10**logps

def draw_msc_periods(n):
    """
    Draw orbital periods according to Multiple Star Catalog
    """
    logps = MSC_TRIPLOGPERKDE.resample(n)
    return 10**logps

def draw_eccs(n,per=10,binsize=0.1,fuzz=0.05,maxecc=0.97):
    """draws eccentricities appropriate to given periods, generated according to empirical data from Multiple Star Catalog
    """
    if np.size(per) == 1 or np.std(np.atleast_1d(per))==0:
        if np.size(per)>1:
            per = per[0]
        if per==0:
            es = np.zeros(n)
        else:
            ne=0
            while ne<10:
                mask = np.absolute(np.log10(MSC_TRIPLEPERS)-np.log10(per))<binsize/2.
                es = MSC_TRIPDATA.e[mask]
                ne = len(es)
                if ne<10:
                    binsize*=1.1
            inds = rand.randint(ne,size=n)
            es = es[inds] * (1 + rand.normal(size=n)*fuzz)
    else:
        longmask = (per > 25)
        shortmask = (per <= 25)
        es = np.zeros(np.size(per))

        elongs = MSC_TRIPDATA.e[MSC_TRIPLEPERS > 25]
        eshorts = MSC_TRIPDATA.e[MSC_TRIPLEPERS <= 25]

        n = np.size(per)
        nlong = longmask.sum()
        nshort = shortmask.sum()
        nelongs = np.size(elongs)
        neshorts = np.size(eshorts)
        ilongs = rand.randint(nelongs,size=nlong)
        ishorts = rand.randint(neshorts,size=nshort)

        es[longmask] = elongs[ilongs]
        es[shortmask] = eshorts[ishorts]

    es = es * (1 + rand.normal(size=n)*fuzz)
    es[es>maxecc] = maxecc
    return np.absolute(es)

########## other utility functions; copied from old code

if not on_rtd:
    import astropy.constants as const
    AU = const.au.cgs.value
    RSUN = const.R_sun.cgs.value
    MSUN = const.M_sun.cgs.value
    DAY = 86400 #seconds
    G = const.G.cgs.value
else:
    const, AU, RSUN, MSUN, DAY, G = (None, None, None, None, None, None)

def rochelobe(q):
    """returns r1/a; q = M1/M2"""
    return 0.49*q**(2./3)/(0.6*q**(2./3) + np.log(1+q**(1./3)))

def withinroche(semimajors,M1,R1,M2,R2):
    """
    Returns boolean array that is True where two stars are within Roche lobe
    """
    q = M1/M2
    return ((R1+R2)*RSUN) > (rochelobe(q)*semimajors*AU)

def semimajor(P,mstar=1):
    """Returns semimajor axis in AU given P in days, mstar in solar masses.
    """
    return ((P*DAY/2/np.pi)**2*G*mstar*MSUN)**(1./3)/AU

def period_from_a(a,mstar):
    return np.sqrt(4*np.pi**2*(a*AU)**3/(G*mstar*MSUN))/DAY

def addmags(*mags):
    """
    "Adds" magnitudes.  Yay astronomical units!
    """
    tot=0
    for mag in mags:
        tot += 10**(-0.4*mag)
    return -2.5*np.log10(tot)

def fluxfrac(*mags):
    """Returns fraction of total flux in first argument, assuming all are magnitudes.
    """
    Ftot = 0
    for mag in mags:
        Ftot += 10**(-0.4*mag)
    F1 = 10**(-0.4*mags[0])
    return F1/Ftot

def dfromdm(dm):
    """Returns distance given distance modulus.
    """
    if np.size(dm)>1:
        dm = np.atleast_1d(dm)
    return 10**(1+dm/5)

def distancemodulus(d):
    """Returns distance modulus given d in parsec.
    """
    if type(d)==Quantity:
        x = d.to('pc').value
    else:
        x = d #assumed to be pc

    if np.size(x)>1:
        d = np.atleast_1d(x)
    return 5*np.log10(x/10)

def fbofm(M):
    return 0.45 - (0.7-M)/4


def mult_masses(mA, f_binary=0.4, f_triple=0.12,
                minmass=0.11, qmin=0.1, n=1e5):
    """Returns m1, m2, and m3 appropriate for TripleStarPopulation, given "primary" mass (most massive of system) and binary/triple fractions.


    star with m1 orbits (m2 + m3).  This means that the primary mass mA will correspond
    either to m1 or m2.  Any mass set to 0 means that component does not exist.
    """

    if np.size(mA) > 1:
        n = len(mA)
    else:
        mA = np.ones(n) * mA


    r = rand.random(n)
    is_single = r > (f_binary + f_triple)
    is_double = (r > f_triple) & (r < (f_binary + f_triple))
    is_triple = r <= f_triple

    CwA = rand.random(n) < 0.5
    CwB = ~CwA


    #these for Triples:

    minq2_A = minmass/mA
    q2_A = rand.random(n)*(1-minq2_A) + minq2_A
    minq1_A = (minmass/mA)/(1+q2_A)
    maxq1_A = 1/(1+q2_A)
    q1_A = rand.random(n)*(maxq1_A-minq1_A) + minq1_A

    minq1_B = 2*minmass/mA
    q1_B = rand.random(n)*(1-minq1_B) + minq1_B
    minq2_B = np.maximum(((q1_B*mA)-minmass)/minmass,
                         (q1_B*mA - minmass)/(q1_B*mA + minmass))
    maxq2_B = 1.
    q2_B = rand.random(n)*(maxq2_B-minq2_B) + minq2_B


    mB_A = q1_A*(1 + q2_A) * mA
    mC_A = q2_A * mA

    mB_B = (q1_B/(1 + q2_B)) * mA
    mC_B = (q1_B*q2_B)/(1 + q2_B) * mA

    mB = CwA*mB_A + CwB*mB_B
    mC = CwA*mC_A + CwB*mC_B

    #for binaries-only
    qmin = minmass/mA
    q = rand.random(n)*(1-qmin) + qmin
    mB[is_double] = q[is_double]*mA[is_double]


    #now need to define the proper mapping from A,B,C to 1,2,3:
    # If no B or C present, then A=1
    # If B present but not C, then A=1, B=2
    # If both B and C present then:
    #     If C is with A, then A=2, C=3, B=1
    #     If C is with B, then A=1, B=2, C=3
    m1 = (mA)*(is_single + is_double) + (CwA*mB + CwB*mA)*is_triple
    m2 = (mB)*is_double + (CwA*mA + CwB*mB)*is_triple
    m3 = mC*is_triple

    return m1, m2, m3

