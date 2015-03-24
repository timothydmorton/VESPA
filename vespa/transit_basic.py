from __future__ import print_function, division

import os
import logging
import pkg_resources

#test if building documentation on readthedocs.org
on_rtd = False

try:
    import numpy as np
    import numpy.random as rand
    from scipy.optimize import leastsq
    from scipy.ndimage import convolve1d
    from scipy.interpolate import LinearNDInterpolator as interpnd
except ImportError:
    on_rtd = True
    np, rand, leastsq, convolve1d, interpnd = (None, None, None, None, None)
    

from .orbits.kepler import Efn
from .stars.utils import rochelobe, withinroche, semimajor

if not on_rtd:
    from vespa_transitutils import find_eclipse
    from vespa_transitutils import traptransit, traptransit_resid
    
    import emcee
else:
    find_eclipse, traptransit, traptransit_resid = (None, None, None)
    emcee = None

if not on_rtd:
    import astropy.constants as const
    AU = const.au.cgs.value
    RSUN = const.R_sun.cgs.value
    MSUN = const.M_sun.cgs.value
    REARTH = const.R_earth.cgs.value
    MEARTH = const.M_earth.cgs.value
    DAY = 86400

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

"""
def correct_fs(fs):
    fflat = fs.ravel().copy()
    wbad = np.where(fflat > 1)[0]     

    #identify lowest and highest index of valid flux
    ilowest=0
    while fflat[ilowest] > 1:
        ilowest += 1
    ihighest = len(fflat)-1
    while fflat[ihighest] > 1:
        ihighest -= 1

    wlo = wbad - 1
    whi = wbad + 1


    #find places where wlo index is still in wbad
    ilo = np.searchsorted(wbad,wlo)
    mask = wbad[ilo]==wlo
    while np.any(mask):
        wlo[mask] -= 1
        ilo = np.searchsorted(wbad,wlo)
        mask = wbad[ilo]==wlo
    ihi = np.searchsorted(wbad,whi)
    ihi = np.clip(ihi,0,len(wbad)-1) #make sure no IndexError
    mask = wbad[ihi]==whi

    while np.any(mask):
        whi[mask] += 1
        ihi = np.searchsorted(wbad,whi)
        ihi = np.clip(ihi,0,len(wbad)-1) #make sure no IndexError
        mask = wbad[ihi]==whi
    
    wlo = np.clip(wlo,ilowest,ihighest)
    whi = np.clip(whi,ilowest,ihighest)

    fflat[wbad] = (fflat[whi] + fflat[wlo])/2. #slightly kludge-y, esp. if there are consecutive bad vals
    return fflat.reshape(fs.shape)
"""

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

def eclipse_tz(P,b,aR,ecc=0,w=0,npts=200,width=1.5,sec=False,dt=1,approx=False,new=False):
    """Returns ts and zs for an eclipse (npts points right around the eclipse)

    
    :param P,b,aR:
        Period, impact parameter, a/Rstar.

    :param ecc,w:
        Eccentricity, argument of periapse.

    :param npts:
        Number of points in transit to return.

    :param width:
        How much to go out of transit. 1.5 is a good choice.

    :param sec:
        Whether to return the values relevant to the secondary occultation
        rather than primary eclipse.

    :param dt:
        Spacing of simulated data points, in minutes.

    :param approx:
        Whether to use the approximate expressions to find the mean
        anomalies.  Default is ``False`` (exact calculation).

    :param new:
        Meaningless.

    :return ts,zs:
        Times from mid-transit (in days) and impact parameters.
        
    """
    if sec:
        eccfactor = np.sqrt(1-ecc**2)/(1-ecc*np.sin(w*np.pi/180))
    else:
        eccfactor = np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180))
    if eccfactor < 1:
        width /= eccfactor
        #if width > 5:
        #    width = 5
        
    if new:
        Ms = np.linspace(-np.pi,np.pi,2e3)
        if ecc != 0:
            Es = Efn(Ms,ecc) #eccentric anomalies
        else:
            Es = Ms
        zs,in_eclipse = find_eclipse(Es,b,aR,ecc,w,width,sec)

        if in_eclipse.sum() < 2:
            raise NoEclipseError

        wecl = np.where(in_eclipse)
        subMs = Ms[wecl]

        dMs = subMs[1:] - subMs[:-1]

        if np.any(subMs < 0) and dMs.max()>1: #if there's a discontinuous wrap-around...
            subMs[np.where(subMs < 0)] += 2*np.pi


        #logging.debug('subMs: {}'.format(subMs))


        minM,maxM = (subMs.min(),subMs.max())
        #logging.debug('minM: {}, maxM: {}'.format(minM,maxM))

        dM = 2*np.pi*dt/(P*24*60)   #the spacing in mean anomaly that corresponds to dt (minutes)
        Ms = np.arange(minM,maxM+dM,dM)
        if ecc != 0:
            Es = Efn(Ms,ecc) #eccentric anomalies
        else:
            Es = Ms

        zs,in_eclipse = find_eclipse(Es,b,aR,ecc,w,width,sec)

        Mcenter = Ms[zs.argmin()]
        phs = (Ms - Mcenter) / (2*np.pi)
        ts = phs*P
        return ts,zs
    
    if not approx:
        if sec:
            inc = np.arccos(b/aR*(1-ecc*np.sin(w*np.pi/180))/(1-ecc**2))
        else:
            inc = np.arccos(b/aR*(1+ecc*np.sin(w*np.pi/180))/(1-ecc**2))

        Ms = np.linspace(-np.pi,np.pi,2e3) #mean anomalies around whole orbit
        if ecc != 0:
            Es = Efn(Ms,ecc) #eccentric anomalies
            nus = 2 * np.arctan2(np.sqrt(1+ecc)*np.sin(Es/2),
                                 np.sqrt(1-ecc)*np.cos(Es/2)) #true anomalies
        else:
            nus = Ms

        r = aR*(1-ecc**2)/(1+ecc*np.cos(nus))  #secondary distance from primary in units of R1

        X = -r*np.cos(w*np.pi/180 + nus)
        Y = -r*np.sin(w*np.pi/180 + nus)*np.cos(inc)
        rsky = np.sqrt(X**2 + Y**2)

        if not sec:
            inds = np.where((np.sin(nus + w*np.pi/180) > 0) & (rsky < width))  #where "front half" of orbit and w/in width
        if sec:
            inds = np.where((np.sin(nus + w*np.pi/180) < 0) & (rsky < width))  #where "front half" of orbit and w/in width
        subMs = Ms[inds].copy()

        if np.any((subMs[1:]-subMs[:-1]) > np.pi):
            subMs[np.where(subMs < 0)] += 2*np.pi

        if np.size(subMs)<2:
            logging.error(subMs)
            raise NoEclipseError

        minM,maxM = (subMs.min(),subMs.max())
        dM = 2*np.pi*dt/(P*24*60)   #the spacing in mean anomaly that corresponds to dt (minutes)
        Ms = np.arange(minM,maxM+dM,dM)
        if ecc != 0:
            Es = Efn(Ms,ecc) #eccentric anomalies
            nus = 2 * np.arctan2(np.sqrt(1+ecc)*np.sin(Es/2),
                                 np.sqrt(1-ecc)*np.cos(Es/2)) #true anomalies
        else:
            nus = Ms
        r = aR*(1-ecc**2)/(1+ecc*np.cos(nus))
        X = -r*np.cos(w*np.pi/180 + nus)
        Y = -r*np.sin(w*np.pi/180 + nus)*np.cos(inc)
        zs = np.sqrt(X**2 + Y**2)  #rsky
    
        #center = absolute(X).argmin()
        #c = polyfit(Ms[center-1:center+2],X[center-1:center+2],1)
        #Mcenter = -c[1]/c[0]
        if not sec:
            Mcenter = Ms[np.absolute(X[np.where(np.sin(nus + w*np.pi/180) > 0)]).argmin()]
        else:
            Mcenter = Ms[np.absolute(X[np.where(np.sin(nus + w*np.pi/180) < 0)]).argmin()]
        phs = (Ms - Mcenter) / (2*np.pi)
        wmin = np.absolute(phs).argmin()
        ts = phs*P

        return ts,zs
    else:
        if sec:
            f0 = -pi/2 - (w*pi/180)
            inc = arccos(b/aR*(1-ecc*sin(w*pi/180))/(1-ecc**2))
        else:
            f0 = pi/2 - (w*pi/180)
            inc = arccos(b/aR*(1+ecc*sin(w*pi/180))/(1-ecc**2))
        fmin = -arcsin(1./aR*sqrt(width**2 - b**2)/sin(inc))
        fmax = arcsin(1./aR*sqrt(width**2 - b**2)/sin(inc))
        if isnan(fmin) or isnan(fmax):
            raise NoEclipseError('no eclipse:  P=%.2f, b=%.3f, aR=%.2f, ecc=%0.2f, w=%.1f' % (P,b,aR,ecc,w))
        fs = linspace(fmin,fmax,npts)
        if sec:
            ts = fs*P/2./pi * sqrt(1-ecc**2)/(1 - ecc*sin(w)) #approximation of constant angular velocity
        else:
            ts = fs*P/2./pi * sqrt(1-ecc**2)/(1 + ecc*sin(w)) #approximation of constant ang. vel.
        fs += f0
        rs = aR*(1-ecc**2)/(1+ecc*cos(fs))
        xs = -rs*cos(w*pi/180 + fs)
        ys = -rs*sin(w*pi/180 + fs)*cos(inc)
        zs = aR*(1-ecc**2)/(1+ecc*cos(fs))*sqrt(1-(sin(w*pi/180 + fs))**2 * (sin(inc))**2)
        return ts,zs


def eclipse_pars(P,M1,M2,R1,R2,ecc=0,inc=90,w=0,sec=False):
    """retuns p,b,aR from P,M1,M2,R1,R2,ecc,inc,w"""
    a = semimajor(P,M1+M2)
    if sec:
        b = a*AU*np.cos(inc*np.pi/180)/(R1*RSUN) * (1-ecc**2)/(1 - ecc*np.sin(w*np.pi/180))
        aR = a*AU/(R2*RSUN)
        p0 = R1/R2
    else:
        b = a*AU*np.cos(inc*np.pi/180)/(R1*RSUN) * (1-ecc**2)/(1 + ecc*np.sin(w*np.pi/180))
        aR = a*AU/(R1*RSUN)
        p0 = R2/R1
    return p0,b,aR

def eclipse(p0,b,aR,P=1,ecc=0,w=0,npts=200,MAfn=None,u1=0.394,u2=0.261,width=3,conv=False,cadence=0.020434028,frac=1,sec=False,dt=2,approx=False,new=False):
    """Returns ts, fs of simulated eclipse.


    :param p0,b,aR:
        Radius ratio, impact parameter, a/Rstar.

    :param P:
        Orbital period

    :param ecc,w: (optional)
        Eccentricity, argument of periapse.

    :param npts: (optional)
        Number of points to simulate.

    :param MAfn: (optional)
        :class:`MAInterpolationFunction` object.

    :param u1,u2: (optional)
        Quadratic limb darkening parameters.

    :param width: (optional)
        Argument defining how much out-of-transit to simulate.  3 is good.

    :param conv: (optional)
        Whether to convolve with box-car to simulate integration time.

    :param cadence: (optional)
        Cadence to simulate; default is Kepler cadence.

    :param frac: (optional)
        Fraction of total light in eclipsed object (for dilution purposes).

    :param sec: (optional)
        If ``True``, then simulate secondary occultation rather than eclipse.

    :param dt: (optional)
        Simulated spacing of theoretical data points, in minutes.

    :param approx: (optional)
        Whether to approximate solution to Kepler's equation or not.

    :param new: (optional)
        Meaningless relic.


    :return ts,fs:
        Times from mid-transit [days] and relative fluxes of simulated eclipse.

    """

    if sec:
        ts,zs = eclipse_tz(P,b/p0,aR/p0,ecc,w,npts=npts,width=(1+1/p0)*width,
                           sec=sec,dt=dt,approx=approx,new=new)
        if zs.min() > (1 + 1/p0):
            #logging.debug('ts: {}'.format(ts))
            #logging.debug('zs: {}'.format(zs))
            raise NoEclipseError
    else:
        ts,zs = eclipse_tz(P,b,aR,ecc,w,npts=npts,width=(1+p0)*width,
                           sec=sec,dt=dt,approx=approx,new=new)
        if zs.min() > (1+p0):
            #logging.debug('ts: {}'.format(ts))
            #logging.debug('zs: {}'.format(zs))
            raise NoEclipseError
        
    if MAfn is None:
        if sec:
            fs = occultquad(zs,u1,u2,1/p0)
        else:
            fs = occultquad(zs,u1,u2,p0)            
    else:
        if sec:
            fs = MAfn(1/p0,zs,u1,u2)
        else:
            fs = MAfn(p0,zs,u1,u2)
        fs[np.isnan(fs)] = 1.

    if conv:
        dt = ts[1]-ts[0]
        npts = np.round(cadence/dt)
        if npts % 2 == 0:
            npts += 1
        boxcar = np.ones(npts)/npts
        fs = convolve1d(fs,boxcar)
    fs = 1 - frac*(1-fs)
    return ts,fs #ts are in the same units P is given in.

def eclipse_tt(p0,b,aR,P=1,ecc=0,w=0,npts=200,MAfn=None,u1=0.394,u2=0.261,conv=False,cadence=0.020434028,frac=1,sec=False,new=True,pars0=None):
    """
    Trapezoidal parameters for simulated orbit.
    
    All arguments passed to :func:`eclipse` except the following:

    :param pars0: (optional)
        Initial guess for least-sq optimization for trapezoid parameters.

    :return dur,dep,slope:
        Best-fit duration, depth, and T/tau for eclipse shape.
    
    """
    ts,fs = eclipse(p0,b,aR,P,ecc,w,npts,MAfn,u1,u2,
                    conv=conv,cadence=cadence,frac=frac,sec=sec,new=new)
    
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


#def traptransit(ts,p):
#    return traptransit(ts,p)

def fit_traptransit(ts,fs,p0):
    """
    Fits trapezoid model to provided ts,fs
    """
    pfit,success = leastsq(traptransit_resid,p0,args=(ts,fs))
    if success not in [1,2,3,4]:
        raise NoFitError
    #logging.debug('success = {}'.format(success))
    return pfit

class TraptransitModel(object):
    """
    Model to enable MCMC fitting of trapezoidal shape.
    """
    def __init__(self,ts,fs,sigs=1e-4,maxslope=30):
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

def traptransit_lhood(pars,ts,fs,sigs,maxslope=30):
    if pars[0] < 0 or pars[1] < 0 or pars[2] < 2 or pars[2] > maxslope:
        return -np.inf
    resid = traptransit_resid(pars,ts,fs)
    return (-0.5*resid**2/sigs**2).sum()

def traptransit_MCMC(ts,fs,dfs=1e-5,nwalkers=200,nburn=300,niter=1000,
                     threads=1,p0=[0.1,0.1,3,0],return_sampler=False,
                     maxslope=30):
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
