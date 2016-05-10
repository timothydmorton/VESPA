from __future__ import print_function, division

import os
import logging
import pkg_resources

import warnings
import math

warnings.filterwarnings('ignore', '.*duration.*')

#test if building documentation on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import numpy as np
    from scipy.optimize import minimize
    from numba import jit
    import numpy.random as rand
    from scipy.optimize import leastsq
    from scipy.ndimage import convolve1d
    from scipy.interpolate import LinearNDInterpolator as interpnd
else:
    np, rand, leastsq, convolve1d, interpnd = (None, None, None, None, None)
    # make fake decorators to allow RTD docs to build without numba
    def jit(*args, **kwargs):
        def foo(*args, **kwargs):
            pass
        return foo
    def vectorize(*args, **kwargs):
        def foo(*args, **kwargs):
            pass
        return foo

if not on_rtd:    
    from batman import _quadratic_ld
else:
    _quadratic_ld = None

    
#from .orbits.kepler import Efn
from .orbits.kepler import calculate_eccentric_anomaly, calculate_eccentric_anomalies
from .stars.utils import rochelobe, withinroche, semimajor

if not on_rtd:
    from ._transitutils import angle_from_transit, angle_from_occultation
    from ._transitutils import zs_of_Ms, transit_duration
    import emcee
else:
    find_eclipse, traptransit, traptransit_resid = (None, None, None)
    emcee = None

#from transit import Central, System, Body
    
if not on_rtd:
    import astropy.constants as const
    AU = const.au.cgs.value
    RSUN = const.R_sun.cgs.value
    MSUN = const.M_sun.cgs.value
    REARTH = const.R_earth.cgs.value
    MEARTH = const.M_earth.cgs.value
    G = const.G.cgs.value
    DAY = 86400.

    DATAFOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    LDDATA = np.recfromtxt('{}/keplerld.dat'.format(DATAFOLDER),names=True)
    LDOK = ((LDDATA.teff < 10000) & (LDDATA.logg > 2.0) & (LDDATA.feh > -2))
    LDPOINTS = np.array([LDDATA.teff[LDOK],LDDATA.logg[LDOK]]).T
    U1FN = interpnd(LDPOINTS,LDDATA.u1[LDOK])
    U2FN = interpnd(LDPOINTS,LDDATA.u2[LDOK])
else:
    const, AU, RSUN, MSUN = (None, None, None, None)
    REARTH, MEARTH, DAY = (None, None, None)
    DATAFOLDER = None
    LDDATA, LDOK, LDPOINTS, U1FN, U2FN = (None, None, None, None, None)
    

MAXSLOPE = 30

def ldcoeffs(teff,logg=4.5,feh=0):
    """
    Returns limb-darkening coefficients in Kepler band.
    """
    teffs = np.atleast_1d(teff)
    loggs = np.atleast_1d(logg)

    Tmin,Tmax = (LDPOINTS[:,0].min(),LDPOINTS[:,0].max())
    gmin,gmax = (LDPOINTS[:,1].min(),LDPOINTS[:,1].max())

    teffs[(teffs < Tmin)] = Tmin + 1
    teffs[(teffs > Tmax)] = Tmax - 1
    loggs[(loggs < gmin)] = gmin + 0.01
    loggs[(loggs > gmax)] = gmax - 0.01

    u1,u2 = (U1FN(teffs,loggs),U2FN(teffs,loggs))
    return u1,u2

class MAInterpolationFunction(object):
    """
    Object enabling fast, vectorized evaluations of Mandel-Agol transit model.

    Interpolates on pre-defined grid calculating Mandel & Agol (2002)
    and Agol & Eastman (2008) calculations.

    This object is generally used as follows::

        >>> import numpy as np
        >>> from vespa import MAInterpolationFunction
        >>> mafn = MAInterpolationFunction() #takes a few seconds
        >>> ps = 0.1 # radius ratio; can be float or array-like
        >>> zs = np.abs(np.linspace(-1,1,1000)) #impact parameters
        >>> fs = mafn(ps, zs) # relative flux

    Even cooler, it can be called with different-sized arrays for
    radius ratio and impact parameter, in which case it returns a
    flux array of shape ``(nps, nzs)``.  This is clearly awesome
    for generating populations of eclipses::

        >>> ps = np.linspace(0.01,0.1,100) # radius ratios
        >>> zs = np.abs(np.linspace(-1,1,1000)) #impact parameters
        >>> fs = mafn(ps, zs)
        >>> fs.shape
            (100, 1000)

    It can also be called with different limb darkening parameters,
    in which case arrays of ``u1`` and ``u2`` should be the third
    and fourth argument, after ``ps`` and ``zs``, with the same shape
    as ``ps`` (radius ratios).  
            
    :param u1,u2: (optional)
        Default quadratic limb darkening parameters. Setting
        these only enables faster evaluation; you can always call with
        different values.

    :param pmin,pmax,nps,nzs,zmax: (optional)
        Parameters describing grid in p and z.
        
    """
    def __init__(self,u1=0.394,u2=0.261,pmin=0.007,pmax=2,nps=200,nzs=200,zmax=None):
        self.u1 = u1
        self.u2 = u2
        self.pmin = pmin
        self.pmax = pmax
        if zmax is None:
            zmax = 1+pmax
        self.zmax = zmax
        self.nps = nps

        ps = np.logspace(np.log10(pmin),np.log10(pmax),nps)
        if pmax < 0.5:
            zs = np.concatenate([np.array([0]),ps-1e-10,ps,np.arange(pmax,1-pmax,0.01),
                              np.arange(1-pmax,zmax,0.005)])
        elif pmax < 1:
            zs = np.concatenate([np.array([0]),ps-1e-10,ps,np.arange(1-pmax,zmax,0.005)])
        else:
            zs = np.concatenate([np.array([0]),ps-1e-10,ps,np.arange(pmax,zmax,0.005)])

        self.nzs = np.size(zs)
        #zs = linspace(0,zmax,nzs)
        #zs = concatenate([zs,ps,ps+1e-10])

        mu0s = np.zeros((np.size(ps),np.size(zs)))
        lambdads = np.zeros((np.size(ps),np.size(zs)))
        etads = np.zeros((np.size(ps),np.size(zs)))
        fs = np.zeros((np.size(ps),np.size(zs)))
        for i,p0 in enumerate(ps):
            f,res = occultquad(zs,u1,u2,p0,return_components=True)
            mu0s[i,:] = res[0]
            lambdads[i,:] = res[1]
            etads[i,:] = res[2]
            fs[i,:] = f
        P,Z = np.meshgrid(ps,zs)
        points = np.array([P.ravel(),Z.ravel()]).T
        self.mu0 = interpnd(points,mu0s.T.ravel())
        
        ##need to make two interpolation functions for lambdad 
        ## b/c it's strongly discontinuous at z=p
        mask = (Z<P)
        pointmask = points[:,1] < points[:,0]

        w1 = np.where(mask)
        w2 = np.where(~mask)
        wp1 = np.where(pointmask)
        wp2 = np.where(~pointmask)

        self.lambdad1 = interpnd(points[wp1],lambdads.T[w1].ravel())
        self.lambdad2 = interpnd(points[wp2],lambdads.T[w2].ravel())
        def lambdad(p,z):
            #where p and z are exactly equal, this will return nan....
            p = np.atleast_1d(p)
            z = np.atleast_1d(z)
            l1 = self.lambdad1(p,z)
            l2 = self.lambdad2(p,z)
            bad1 = np.isnan(l1)
            l1[np.where(bad1)]=0
            l2[np.where(~bad1)]=0
            return l1*~bad1 + l2*bad1
        self.lambdad = lambdad
        
        #self.lambdad = interpnd(points,lambdads.T.ravel())
        self.etad = interpnd(points,etads.T.ravel())        
        self.fn = interpnd(points,fs.T.ravel())

    def __call__(self,ps,zs,u1=.394,u2=0.261,force_broadcast=False):
        """  returns array of fluxes; if ps and zs aren't the same shape, then returns array of 
        shape (nps, nzs)
        """
        #return self.fn(ps,zs)

        if np.size(ps)>1 and (np.size(ps)!=np.size(zs) or force_broadcast):
            P = ps[:,None]
            if np.size(u1)>1 or np.size(u2)>1:
                if u1.shape != ps.shape or u2.shape != ps.shape:
                    raise ValueError('limb darkening coefficients must be same size as ps')
                U1 = u1[:,None]
                U2 = u2[:,None]
            else:
                U1 = u1
                U2 = u2
        else:
            P = ps
            U1 = u1
            U2 = u2
            
        if np.size(u1)>1 or np.any(u1 != self.u1) or np.any(u2 != self.u2):
            mu0 = self.mu0(P,zs)
            lambdad = self.lambdad(P,zs)
            etad = self.etad(P,zs)
            fs = 1. - ((1-U1-2*U2)*(1-mu0) + (U1+2*U2)*(lambdad+2./3*(P > zs)) + U2*etad)/(1.-U1/3.-U2/6.)
            
            #if fix:
            #    fs = correct_fs(fs)
        else:
            fs = self.fn(P,zs)
            
        return fs

def impact_parameter(a, R, inc, ecc=0, w=0, return_occ=False):
    """a in AU, R in Rsun, inc & w in radians
    """
    b_tra = a*AU*np.cos(inc)/(R*RSUN) * (1-ecc**2)/(1 + ecc*np.sin(w))

    if return_occ:
        b_tra = a*AU*np.cos(inc)/(R*RSUN) * (1-ecc**2)/(1 - ecc*np.sin(w))
        return b_tra, b_occ
    else:
        return b_tra

def transit_T14(P,Rp,Rs=1,b=0,Ms=1,ecc=0,w=0):
    """P in days, Rp in Earth radii, Rs in Solar radii, b=impact parameter, Ms Solar masses. Returns T14 in hours. w in deg.
    """
    a = semimajor(P,Ms)*AU
    k = Rp*REARTH/(Rs*RSUN)
    inc = np.pi/2 - b*RSUN/a
    return  P*DAY/np.pi*np.arcsin(Rs*RSUN/a * np.sqrt((1+k)**2 - b**2)/np.sin(inc)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180)) / 3600.

def transit_T23(P,Rp,Rs=1,b=0,Ms=1,ecc=0,w=0):
    a = semimajor(P,Ms)*AU
    k = Rp*REARTH/(Rs*RSUN)
    inc = np.pi/2 - b*RSUN/a

    return P*DAY/np.pi*np.arcsin(Rs*RSUN/a * np.sqrt((1-k)**2 - b**2)/np.sin(inc)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*pi/180)) / 3600.#*24*60    

def eclipse_depth(mafn,Rp,Rs,b,u1=0.394,u2=0.261,max_only=False,npts=100,force_1d=False):
    """ Calculates average (or max) eclipse depth

    ***why does b>1 take so freaking long?...
    """
    k = Rp*REARTH/(Rs*RSUN)
    
    if max_only:
        return 1 - mafn(k,b,u1,u2)

    if np.size(b) == 1:
        x = np.linspace(0,np.sqrt(1-b**2),npts)
        y = b
        zs = np.sqrt(x**2 + y**2)
        fs = mafn(k,zs,u1,u2) # returns array of shape (nks,nzs)
        depth = 1-fs
    else:
        xmax = np.sqrt(1-b**2)
        x = np.linspace(0,1,npts)*xmax[:,Nones]
        y = b[:,None]
        zs = np.sqrt(x**2 + y**2)
        fs = mafn(k,zs.ravel(),u1,u2)
        if not force_1d:
            fs = fs.reshape(size(k),*zs.shape)
        depth = 1-fs
        
    meandepth = np.squeeze(depth.mean(axis=depth.ndim-1))
    
    return meandepth  #array of average depths, shape (nks,nbs)

def minimum_inclination(P,M1,M2,R1,R2):
    """
    Returns the minimum inclination at which two bodies from two given sets eclipse

    Only counts systems not within each other's Roche radius
    
    :param P:
        Orbital periods.

    :param M1,M2,R1,R2:
        Masses and radii of primary and secondary stars.  
    """
    P,M1,M2,R1,R2 = (np.atleast_1d(P),
                     np.atleast_1d(M1),
                     np.atleast_1d(M2),
                     np.atleast_1d(R1),
                     np.atleast_1d(R2))
    semimajors = semimajor(P,M1+M2)
    rads = ((R1+R2)*RSUN/(semimajors*AU))
    ok = (~np.isnan(rads) & ~withinroche(semimajors,M1,R1,M2,R2))
    if ok.sum() == 0:
        logging.error('P: {}'.format(P))
        logging.error('M1: {}'.format(M1))
        logging.error('M2: {}'.format(M2))
        logging.error('R1: {}'.format(R1))
        logging.error('R2: {}'.format(R2))
        if np.all(withinroche(semimajors,M1,R1,M2,R2)):
            raise AllWithinRocheError('All simulated systems within Roche lobe')
        else:
            raise EmptyPopulationError('no valid systems! (see above)')
    mininc = np.arccos(rads[ok].max())*180/np.pi
    return mininc

def a_over_Rs(P,R2,M2,M1=1,R1=1,planet=True):
    """
    Returns a/Rs for given parameters.
    """
    if planet:
        M2 *= REARTH/RSUN
        R2 *= MEARTH/MSUN
    return semimajor(P,M1+M2)*AU/(R1*RSUN)

def eclipse(p0,b,aR,P=1,ecc=0,w=0,npts=100,u1=0.394,u2=0.261,width=3,
            conv=True,cadence=1626./86400,frac=1,sec=False,tol=1e-4):
    
    dur = transit_duration(p0, P, b, aR, ecc, w*np.pi/180, sec)
    if np.isnan(dur):
        raise NoEclipseError
    if dur < 2*cadence:
        dur = 2*cadence

    if sec:
        M0 = minimize(angle_from_occultation, 
                      -np.pi/2 - w*np.pi/180, 
                      args=(ecc, w*np.pi/180), 
                      method='Nelder-Mead', tol=tol).x[0]
    else:
        M0 = minimize(angle_from_transit, 
                      np.pi/2 - w*np.pi/180, 
                      args=(ecc, w*np.pi/180), 
                      method='Nelder-Mead', tol=tol).x[0]

    Mlo = M0 - (dur/P)*2*np.pi * width/2.
    Mhi = M0 + (dur/P)*2*np.pi * width/2.

    logging.debug('M0={}; Mlo={}; Mhi={} (dur={})'.format(M0,Mlo,Mhi,dur))

    Ms = np.linspace(Mlo, Mhi, npts)
    ts = (Ms - M0) / (2*np.pi) * P
    
    zs = zs_of_Ms(Ms, b, aR, ecc, w*np.pi/180, sec)


    #logging.debug('zs={}'.format(zs))

    if sec:
        zs *= 1./p0
        fs = _quadratic_ld._quadratic_ld(zs, 1./p0, u1, u2, 1)
    else:
        fs = _quadratic_ld._quadratic_ld(zs, p0, u1, u2, 1)

    if conv:
        dt = ts[1]-ts[0]
        npts = int(np.round(cadence/dt))
        if npts % 2 == 0:
            npts += 1
        boxcar = np.ones(npts)/npts
        fs = convolve1d(fs,boxcar)
    fs = 1 - frac*(1-fs)
    return ts,fs    


def eclipse_pars(P,M1,M2,R1,R2,ecc=0,inc=90,w=0,sec=False):
    """retuns p,b,aR from P,M1,M2,R1,R2,ecc,inc,w"""
    a = semimajor(P,M1+M2)
    if sec:
        b = a*AU*np.cos(inc*np.pi/180)/(R1*RSUN) * (1-ecc**2)/(1 - ecc*np.sin(w*np.pi/180))
        #aR = a*AU/(R2*RSUN) #I feel like this was to correct a bug, but this should not be.
        #p0 = R1/R2 #why this also?
    else:
        b = a*AU*np.cos(inc*np.pi/180)/(R1*RSUN) * (1-ecc**2)/(1 + ecc*np.sin(w*np.pi/180))
        #aR = a*AU/(R1*RSUN)
        #p0 = R2/R1
    p0 = R2/R1
    aR = a*AU/(R1*RSUN)
    return p0,b,aR


def eclipse_new(p0,b,aR,P=1,ecc=0,w=0,npts=200,MAfn=None,u1=0.394,u2=0.261,width=3,conv=False,cadence=1626./86400,frac=1,sec=False,dt=2,approx=False,new=True):
    """

    """

    # Given both a/R* and P.
    #  Assume R* = Rsun
    a = aR * RSUN
    M = a**3 / G * (4*np.pi**2)/(P*DAY)**2 / MSUN

    if sec:
        central = Central(mu1=u1, mu2=u2, mass=M, radius=p0)
        s = System(central)
        body = Body(r=1, a=aR, b=b, e=ecc, omega=np.mod(w+math.pi, 2*math.pi))
        s.add_body(body)
        incl = body.incl
        si = math.sin(math.radians(incl))
        arg = 1./aR * math.sqrt((1+1/p0) ** 2 - b**2) / si
        dur = math.asin(arg) * P * math.pi #*
    else:
        central = Central(mu1=u1, mu2=u2, mass=M)
        s = System(central)
        body = Body(r=p0, a=aR, b=b, e=ecc, omega=w)
        s.add_body(body)

    dur = body.duration
    ts = np.linspace(-1.5*dur, 1.5*dur, npts)
    fs = s.light_curve(ts, texp=cadence)
    return ts, fs


def eclipse_tt(p0,b,aR,P=1,ecc=0,w=0,npts=100,u1=0.394,u2=0.261,conv=True,
               cadence=1626./86400,frac=1,sec=False,pars0=None,tol=1e-4,width=3):
    """
    Trapezoidal parameters for simulated orbit.
    
    All arguments passed to :func:`eclipse` except the following:

    :param pars0: (optional)
        Initial guess for least-sq optimization for trapezoid parameters.

    :return dur,dep,slope:
        Best-fit duration, depth, and T/tau for eclipse shape.
    
    """
    ts,fs = eclipse(p0=p0,b=b,aR=aR,P=P,ecc=ecc,w=w,npts=npts,u1=u1,u2=u2,
                    conv=conv,cadence=cadence,frac=frac,sec=sec,tol=tol,width=width)
    
    #logging.debug('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(p0,b,aR,P,ecc,w,xmax,npts,u1,u2,leastsq,conv,cadence,frac,sec,new))
    #logging.debug('ts: {} fs: {}'.format(ts,fs))

    if pars0 is None:
        depth = 1 - fs.min()
        duration = (fs < (1-0.01*depth)).sum()/float(len(fs)) * (ts[-1] - ts[0])
        tc0 = ts[fs.argmin()]
        pars0 = np.array([duration,depth,5.,tc0])
    
    dur,dep,slope,epoch = fit_traptransit(ts,fs,pars0)
    return dur,dep,slope


def occultquad(z,u1,u2,p0,return_components=False):
    """
    #### Mandel-Agol code:
    #   Python translation of IDL code.
    #   This routine computes the lightcurve for occultation of a
    #   quadratically limb-darkened source without microlensing.  Please
    #   cite Mandel & Agol (2002) and Eastman & Agol (2008) if you make use
    #   of this routine in your research.  Please report errors or bugs to
    #   jdeast@astronomy.ohio-state.edu

    .. note::

        Should probably wrap the Fortran code at some point.
        (This particular part of the code was put together awhile ago.)

    """
    z = np.atleast_1d(z)
    nz = np.size(z)
    lambdad = np.zeros(nz)
    etad = np.zeros(nz)
    lambdae = np.zeros(nz)
    omega=1.-u1/3.-u2/6.

    ## tolerance for double precision equalities
    ## special case integrations
    tol = 1e-14

    p = np.absolute(p0)
    
    z = np.where(np.absolute(p-z) < tol,p,z)
    z = np.where(np.absolute((p-1)-z) < tol,p-1.,z)
    z = np.where(np.absolute((1-p)-z) < tol,1.-p,z)
    z = np.where(z < tol,0.,z)
               
    x1=(p-z)**2.
    x2=(p+z)**2.
    x3=p**2.-z**2.
    

    def finish(p,z,u1,u2,lambdae,lambdad,etad):
        omega = 1. - u1/3. - u2/6.
        #avoid Lutz-Kelker bias
        if p0 > 0:
            #limb darkened flux
            muo1 = 1 - ((1-u1-2*u2)*lambdae+(u1+2*u2)*(lambdad+2./3*(p > z)) + u2*etad)/omega
            #uniform disk
            mu0 = 1 - lambdae
        else:
            #limb darkened flux
            muo1 = 1 + ((1-u1-2*u2)*lambdae+(u1+2*u2)*(lambdad+2./3*(p > z)) + u2*etad)/omega
            #uniform disk
            mu0 = 1 + lambdae
        if return_components:
            return muo1,(mu0,lambdad,etad)
        else:
            return muo1



    ## trivial case of no planet
    if p <= 0.:
        return finish(p,z,u1,u2,lambdae,lambdad,etad)

    ## Case 1 - the star is unocculted:
    ## only consider points with z lt 1+p
    notusedyet = np.where( z < (1. + p) )[0]
    if np.size(notusedyet) == 0:
        return finish(p,z,u1,u2,lambdae,lambdad,etad)

    # Case 11 - the  source is completely occulted:
    if p >= 1.:
        cond = z[notusedyet] <= p-1.
        occulted = np.where(cond)#,complement=notused2)
        notused2 = np.where(~cond)
        #occulted = where(z[notusedyet] <= p-1.)#,complement=notused2)
        if np.size(occulted) != 0:
            ndxuse = notusedyet[occulted]
            etad[ndxuse] = 0.5 # corrected typo in paper
            lambdae[ndxuse] = 1.
            # lambdad = 0 already
            #notused2 = where(z[notusedyet] > p-1)
            if np.size(notused2) == 0:
                return finish(p,z,u1,u2,lambdae,lambdad,etad)
            notusedyet = notusedyet[notused2]
                
    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    inegressuni = np.where((z[notusedyet] >= np.absolute(1.-p)) & (z[notusedyet] < 1.+p))
    if np.size(inegressuni) != 0:
        ndxuse = notusedyet[inegressuni]
        tmp = (1.-p**2.+z[ndxuse]**2.)/2./z[ndxuse]
        tmp = np.where(tmp > 1.,1.,tmp)
        tmp = np.where(tmp < -1.,-1.,tmp)
        kap1 = np.arccos(tmp)
        tmp = (p**2.+z[ndxuse]**2-1.)/2./p/z[ndxuse]
        tmp = np.where(tmp > 1.,1.,tmp)
        tmp = np.where(tmp < -1.,-1.,tmp)
        kap0 = np.arccos(tmp)
        tmp = 4.*z[ndxuse]**2-(1.+z[ndxuse]**2-p**2)**2
        tmp = np.where(tmp < 0,0,tmp)
        lambdae[ndxuse] = (p**2*kap0+kap1 - 0.5*np.sqrt(tmp))/np.pi
        # eta_1
        etad[ndxuse] = 1./2./np.pi*(kap1+p**2*(p**2+2.*z[ndxuse]**2)*kap0- \
           (1.+5.*p**2+z[ndxuse]**2)/4.*np.sqrt((1.-x1[ndxuse])*(x2[ndxuse]-1.)))
    
    # Case 5, 6, 7 - the edge of planet lies at origin of star
    cond = z[notusedyet] == p
    ocltor = np.where(cond)#, complement=notused3)
    notused3 = np.where(~cond)
    #ocltor = where(z[notusedyet] == p)#, complement=notused3)
    t = np.where(z[notusedyet] == p)
    if np.size(ocltor) != 0:
        ndxuse = notusedyet[ocltor] 
        if p < 0.5:
            # Case 5
            q=2.*p  # corrected typo in paper (2k -> 2p)
            Ek,Kk = ellke(q)
            # lambda_4
            lambdad[ndxuse] = 1./3.+2./9./np.pi*(4.*(2.*p**2-1.)*Ek+\
                                              (1.-4.*p**2)*Kk)
            # eta_2
            etad[ndxuse] = p**2/2.*(p**2+2.*z[ndxuse]**2)        
            lambdae[ndxuse] = p**2 # uniform disk
        elif p > 0.5:
            # Case 7
            q=0.5/p # corrected typo in paper (1/2k -> 1/2p)
            Ek,Kk = ellke(q)
            # lambda_3
            lambdad[ndxuse] = 1./3.+16.*p/9./np.pi*(2.*p**2-1.)*Ek-\
                              (32.*p**4-20.*p**2+3.)/9./np.pi/p*Kk
            # etad = eta_1 already
        else:
            # Case 6
            lambdad[ndxuse] = 1./3.-4./np.pi/9.
            etad[ndxuse] = 3./32.
        #notused3 = where(z[notusedyet] != p)
        if np.size(notused3) == 0:
            return finish(p,z,u1,u2,lambdae,lambdad,etad)
        notusedyet = notusedyet[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)
    cond = ((z[notusedyet] > 0.5+np.absolute(p-0.5)) & \
                       (z[notusedyet] < 1.+p))  | \
                      ( (p > 0.5) & (z[notusedyet] > np.absolute(1.-p)) & \
                        (z[notusedyet] < p))
    inegress = np.where(cond)
    notused4 = np.where(~cond)
    #inegress = where( ((z[notusedyet] > 0.5+abs(p-0.5)) & \
        #(z[notusedyet] < 1.+p))  | \
        #( (p > 0.5) & (z[notusedyet] > abs(1.-p)) & \
        #(z[notusedyet] < p)) )#, complement=notused4)
    if np.size(inegress) != 0:

        ndxuse = notusedyet[inegress]
        q=np.sqrt((1.-x1[ndxuse])/(x2[ndxuse]-x1[ndxuse]))
        Ek,Kk = ellke(q)
        n=1./x1[ndxuse]-1.

        # lambda_1:
        lambdad[ndxuse]=2./9./np.pi/np.sqrt(x2[ndxuse]-x1[ndxuse])*\
                         (((1.-x2[ndxuse])*(2.*x2[ndxuse]+x1[ndxuse]-3.)-\
                           3.*x3[ndxuse]*(x2[ndxuse]-2.))*Kk+(x2[ndxuse]-\
                           x1[ndxuse])*(z[ndxuse]**2+7.*p**2-4.)*Ek-\
                          3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        #notused4 = where( ( (z[notusedyet] <= 0.5+abs(p-0.5)) | \
        #                    (z[notusedyet] >= 1.+p) ) & ( (p <= 0.5) | \
        #                    (z[notusedyet] <= abs(1.-p)) | \
        #                    (z[notusedyet] >= p) ))
        if np.size(notused4) == 0:
            return finish(p,z,u1,u2,lambdae,lambdad,etad)
        notusedyet = notusedyet[notused4]

    # Case 3, 4, 9, 10 - planet completely inside star
    if p < 1.:
        cond = z[notusedyet] <= (1.-p)
        inside = np.where(cond)
        notused5 = np.where(~cond)
        #inside = where(z[notusedyet] <= (1.-p))#, complement=notused5)
        if np.size(inside) != 0:
            ndxuse = notusedyet[inside]

            ## eta_2
            etad[ndxuse] = p**2/2.*(p**2+2.*z[ndxuse]**2)

            ## uniform disk
            lambdae[ndxuse] = p**2

            ## Case 4 - edge of planet hits edge of star
            edge = np.where(z[ndxuse] == 1.-p)#, complement=notused6)
            if np.size(edge[0]) != 0:
                ## lambda_5
                lambdad[ndxuse[edge]] = 2./3./np.pi*np.arccos(1.-2.*p)-\
                                      4./9./np.pi*np.sqrt(p*(1.-p))*(3.+2.*p-8.*p**2)
                if p > 0.5:
                    lambdad[ndxuse[edge]] -= 2./3.
                notused6 = np.where(z[ndxuse] != 1.-p)
                if np.size(notused6) == 0:
                    return finish(p,z,u1,u2,lambdae,lambdad,etad)
                ndxuse = ndxuse[notused6[0]]

            ## Case 10 - origin of planet hits origin of star
            origin = np.where(z[ndxuse] == 0)#, complement=notused7)
            if np.size(origin) != 0:
                ## lambda_6
                lambdad[ndxuse[origin]] = -2./3.*(1.-p**2)**1.5
                notused7 = np.where(z[ndxuse] != 0)
                if np.size(notused7) == 0:
                    return finish(p,z,u1,u2,lambdae,lambdad,etad)
                ndxuse = ndxuse[notused7[0]]
   
            q=np.sqrt((x2[ndxuse]-x1[ndxuse])/(1.-x1[ndxuse]))
            n=x2[ndxuse]/x1[ndxuse]-1.
            Ek,Kk = ellke(q)    

            ## Case 3, Case 9 - anywhere in between
            ## lambda_2
            lambdad[ndxuse] = 2./9./np.pi/np.sqrt(1.-x1[ndxuse])*\
                              ((1.-5.*z[ndxuse]**2+p**2+x3[ndxuse]**2)*Kk+\
                               (1.-x1[ndxuse])*(z[ndxuse]**2+7.*p**2-4.)*Ek-\
                               3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        ## if there are still unused elements, there's a bug in the code
        ## (please report it)
        #notused5 = where(z[notusedyet] > (1.-p))
        if notused5[0] != 0:
            logging.error("The following values of z didn't fit into a case:")

        return finish(p,z,u1,u2,lambdae,lambdad,etad)

# Computes Hasting's polynomial approximation for the complete
# elliptic integral of the first (ek) and second (kk) kind
def ellke(k):
    m1=1.-k**2
    logm1 = np.log(m1)

    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639
    ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-logm1)
    ek = ee1+ee2
        
    a0=1.38629436112
    a1=0.09666344259
    a2=0.03590092383
    a3=0.03742563713
    a4=0.01451196212
    b0=0.5
    b1=0.12498593597
    b2=0.06880248576
    b3=0.03328355346
    b4=0.00441787012
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*logm1
    kk = ek1-ek2
    
    return [ek,kk]

# Computes the complete elliptical integral of the third kind using
# the algorithm of Bulirsch (1965):
def ellpic_bulirsch(n,k):
    kc=np.sqrt(1.-k**2); p=n+1.
    if(p.min() < 0.):
        logging.warning('Negative p')
    m0=1.; c=1.; p=np.sqrt(p); d=1./p; e=kc
    while 1:
        f = c; c = d/p+c; g = e/p; d = 2.*(f*g+d)
        p = g + p; g = m0; m0 = kc + m0
        if (np.absolute(1.-kc/g)).max() > 1.e-8:
            kc = 2*np.sqrt(e); e=kc*m0
        else:
            return 0.5*np.pi*(c*m0+d)/(m0*(m0+p))


def fit_traptransit(ts,fs,p0):
    """
    Fits trapezoid model to provided ts,fs
    """
    pfit,success = leastsq(traptransit_resid,p0,args=(ts,fs))
    if success not in [1,2,3,4]:
        raise NoFitError
    #logging.debug('success = {}'.format(success))
    return pfit

@jit(nopython=True)
def traptransit(ts, pars):
    npts = len(ts)
    fs = np.empty(npts, dtype=np.float64)

    if pars[2] < 2 or pars[0] <= 0:
        for i in range(npts):
            fs[i] = np.inf
    else:
        p0_2 = pars[0]/2.
        t1 = pars[3] - p0_2
        t2 = pars[3] - p0_2 + pars[0]/pars[2]
        t3 = pars[3] + p0_2 - pars[0]/pars[2]
        t4 = pars[3] + p0_2
        for i in range(npts):
            if (ts[i]  > t1) and (ts[i] < t2):
                fs[i] = 1-pars[1]*pars[2]/pars[0]*(ts[i] - t1)
            elif (ts[i] > t2) and (ts[i] < t3):
                fs[i] = 1-pars[1]
            elif (ts[i] > t3) and (ts[i] < t4):
                fs[i] = 1-pars[1] + pars[1]*pars[2]/pars[0]*(ts[i]-t3)
            else:
                fs[i] = 1
    return fs

@jit(nopython=True)
def traptransit_resid(pars, ts, fs):
    resid = np.empty(len(fs), dtype=np.float64)

    fmod = traptransit(ts, pars)
    
    for i in range(len(resid)):
        resid[i] = fmod[i] - fs[i]

    return resid
    

class TraptransitModel(object):
    """
    Model to enable MCMC fitting of trapezoidal shape.
    """
    def __init__(self,ts,fs,sigs=1e-4,maxslope=MAXSLOPE):
        self.n = np.size(ts)
        if np.size(sigs)==1:
            sigs = np.ones(self.n)*sigs
        self.ts = ts
        self.fs = fs
        self.sigs = sigs
        self.maxslope = maxslope
        
    def __call__(self,pars):
        pars = np.array(pars)
        return traptransit_lhood(pars,self.ts,self.fs,self.sigs,maxslope=self.maxslope)


@jit(nopython=True)
def traptransit_lhood(pars, ts, fs, sigs, maxslope=MAXSLOPE):
    """
    Params: depth, duration, slope, t0
    """
    if pars[0] < 0 or pars[1] < 0 or pars[2] < 2 or pars[2] > maxslope:
        return -np.inf

    fmod = traptransit(ts, pars)
    tot = 0
    for i in range(len(ts)):
        tot += -0.5*(fmod[i] - fs[i])*(fmod[i] - fs[i]) / (sigs[i]*sigs[i])
        #tot += np.log(1./pars[2]) #logflat prior on slope

    return tot
    
def traptransit_lhood_old(pars,ts,fs,sigs,maxslope=MAXSLOPE):
    if pars[0] < 0 or pars[1] < 0 or pars[2] < 2 or pars[2] > maxslope:
        return -np.inf
    resid = traptransit_resid(pars,ts,fs)
    return (-0.5*resid**2/sigs**2).sum()

def traptransit_MCMC(ts,fs,dfs=1e-5,nwalkers=200,nburn=300,niter=1000,
                     threads=1,p0=[0.1,0.1,3,0],return_sampler=False,
                     maxslope=MAXSLOPE):
    """
    Fit trapezoidal model to provided ts, fs, [dfs] using MCMC.

    Standard emcee usage.
    """
    model = TraptransitModel(ts,fs,dfs,maxslope=maxslope)
    sampler = emcee.EnsembleSampler(nwalkers,4,model,threads=threads)
    T0 = p0[0]*(1+rand.normal(size=nwalkers)*0.1)
    d0 = p0[1]*(1+rand.normal(size=nwalkers)*0.1)
    slope0 = p0[2]*(1+rand.normal(size=nwalkers)*0.1)
    ep0 = p0[3]+rand.normal(size=nwalkers)*0.0001

    p0 = np.array([T0,d0,slope0,ep0]).T

    pos, prob, state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(pos, niter, rstate0=state)
    if return_sampler:
        return sampler
    else:
        return sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3]

##### Custom Exceptions

class NoEclipseError(Exception):
    pass

class NoFitError(Exception):
    pass

class EmptyPopulationError(Exception):
    pass

class NotImplementedError(Exception):
    pass

class AllWithinRocheError(Exception):
    pass
