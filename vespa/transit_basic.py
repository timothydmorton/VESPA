from __future__ import print_function, division

import os
import numpy as np

import logging

import pkg_resources

import orbitutils as ou

from scipy.optimize import leastsq
from scipy.ndimage import convolve1d

from scipy.interpolate import LinearNDInterpolator as interpnd

from starutils.utils import rochelobe, withinroche, semimajor
import transit_utils as tru

try:
    import transit_utils as tru
except ImportError:
    print('transit_basic: did not import transit_utils.')

import emcee

DATAFOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

LDDATA = recfromtxt('{}/keplerld.dat'.format(DATAFOLDER),names=True)
LDOK = ((LDDATA.teff < 10000) & (LDDATA.logg > 2.0) & (LDDATA.feh > -2))
LDPOINTS = array([LDDATA.teff[LDOK],LDDATA.logg[LDOK]]).T
U1FN = interpnd(LDPOINTS,LDDATA.u1[LDOK])
U2FN = interpnd(LDPOINTS,LDDATA.u2[LDOK])

def ldcoeffs(teff,logg=4.5,feh=0):
    teffs = atleast_1d(teff)
    loggs = atleast_1d(logg)

    Tmin,Tmax = (LDPOINTS[:,0].min(),LDPOINTS[:,0].max())
    gmin,gmax = (LDPOINTS[:,1].min(),LDPOINTS[:,1].max())

    teffs[where(teffs < Tmin)] = Tmin + 1
    teffs[where(teffs > Tmax)] = Tmax - 1
    loggs[where(loggs < gmin)] = gmin + 0.01
    loggs[where(loggs > gmax)] = gmax - 0.01

    u1,u2 = (U1FN(teffs,loggs),U2FN(teffs,loggs))
    return u1,u2

def correct_fs(fs):
    """ patch-y fix to anything with messed-up fs
    """
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

class MAInterpolationFunction(object):
    def __init__(self,u1=0.394,u2=0.261,pmin=0.007,pmax=2,nps=200,nzs=200,zmax=None):
    #def __init__(self,pmin=0.007,pmax=2,nps=500,nzs=500):
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

    def __call__(self,ps,zs,u1=.394,u2=0.261,force_broadcast=False,fix=False):
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
            
        if size(u1)>1 or any(u1 != self.u1) or any(u2 != self.u2):
            mu0 = self.mu0(P,zs)
            lambdad = self.lambdad(P,zs)
            etad = self.etad(P,zs)
            fs = 1. - ((1-U1-2*U2)*(1-mu0) + (U1+2*U2)*(lambdad+2./3*(P > zs)) + U2*etad)/(1.-U1/3.-U2/6.)
            
            if fix:
                fs = correct_fs(fs)
        else:
            fs = self.fn(P,zs)
            
        return fs

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
    P,M1,M2,R1,R2 = (np.atleast_1d(P),
                     np.atleast_1d(M1),
                     np.atleast_1d(M2),
                     np.atleast_1d(R1),
                     np.atleast_1d(R2))
    semimajors = semimajor(P,M1+M2)
    rads = ((R1+R2)*RSUN/(semimajors*AU))
    ok = (~np.isnan(rads) & ~withinroche(semimajors,M1,R1,M2,R2))
    if ok.sum() == 0:
        logging.error('P: '+P)
        logging.error('M1: '+M1)
        logging.error('M2: '+M2)
        logging.error('R1: '+R1)
        logging.error('R2: '+R2)
        if np.all(withinroche(semimajors,M1,R1,M2,R2)):
            raise AllWithinRocheError('All simulated systems within Roche lobe')
        else:
            raise EmptyPopulationError('no valid systems! (see above)')
    mininc = np.arccos(rads[ok].max())*180/pi
    return mininc

def a_over_Rs(P,R2,M2,M1=1,R1=1,planet=True):
    if planet:
        M2 *= REARTH/RSUN
        R2 *= MEARTH/MSUN
    return semimajor(P,M1+M2)*AU/(R1*RSUN)

def eclipse_tz(P,b,aR,ecc=0,w=0,npts=200,width=1.5,sec=False,dt=1,approx=False,new=False):
    """Returns ts and zs for an eclipse (npts points right around the eclipse)
    """
    if sec:
        eccfactor = np.sqrt(1-ecc**2)/(1-ecc*np.sin(w*pi/180))
    else:
        eccfactor = np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*pi/180))
    if eccfactor < 1:
        width /= eccfactor
        #if width > 5:
        #    width = 5
        
    if new:
        Ms = np.linspace(-np.pi,np.pi,2e3)
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
        else:
            Es = Ms
        zs,in_eclipse = tru.find_eclipse(Es,b,aR,ecc,w,width,sec)

        if in_eclipse.sum() < 2:
            raise NoEclipseError

        subMs = Ms[in_eclipse]

        dMs = subMs[1:] - subMs[:-1]

        if np.any(subMs < 0) and dMs.max()>1: #if there's a discontinuous wrap-around...
            subMs[(subMs < 0)] += 2*pi

                      
        logging.debug(subMs)

        minM,maxM = (subMs.min(),subMs.max())
                      
        logging.debug(minM,maxM)

        dM = 2*np.pi*dt/(P*24*60)   #the spacing in mean anomaly that corresponds to dt (minutes)
        Ms = np.arange(minM,maxM+dM,dM)
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
        else:
            Es = Ms

        zs,in_eclipse = tru.find_eclipse(Es,b,aR,ecc,w,width,sec)

        Mcenter = Ms[zs.argmin()]
        phs = (Ms - Mcenter) / (2*pi)
        ts = phs*P
        return ts,zs
    
    if not approx:
        if sec:
            inc = np.arccos(b/aR*(1-ecc*np.sin(w*pi/180))/(1-ecc**2))
        else:
            inc = np.arccos(b/aR*(1+ecc*np.sin(w*pi/180))/(1-ecc**2))

        Ms = np.linspace(-np.pi,np.pi,2e3) #mean anomalies around whole orbit
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
            nus = 2 * np.arctan2(np.sqrt(1+ecc)*np.sin(Es/2),
                                 np.sqrt(1-ecc)*np.cos(Es/2)) #true anomalies
        else:
            nus = Ms

        r = aR*(1-ecc**2)/(1+ecc*np.cos(nus))  #secondary distance from primary in units of R1

        X = -r*np.cos(w*np.pi/180 + nus)
        Y = -r*np.sin(w*np.pi/180 + nus)*np.cos(inc)
        rsky = np.sqrt(X**2 + Y**2)

        if not sec:
            inds = np.where((np.sin(nus + w*np.pi/180) > 0) & 
                            (rsky < width))  #where "front half" of orbit and w/in width
        if sec:
            inds = np.where((np.sin(nus + w*np.pi/180) < 0) & 
                            (rsky < width))  #where "front half" of orbit and w/in width
        subMs = Ms[inds].copy()

        if np.any((subMs[1:]-subMs[:-1]) > np.pi):
            subMs[(subMs < 0)] += 2*np.pi

        if np.size(subMs)<2:
            logging.error(subMs)
            raise NoEclipseError

        minM,maxM = (subMs.min(),subMs.max())
        dM = 2*np.pi*dt/(P*24*60)   #the spacing in mean anomaly that corresponds to dt (minutes)
        Ms = np.arange(minM,maxM+dM,dM)
        if ecc != 0:
            Es = ou.Efn(Ms,ecc) #eccentric anomalies
            nus = 2 * np.arctan2(np.sqrt(1+ecc)*np.sin(Es/2),
                                 np.sqrt(1-ecc)*np.cos(Es/2)) #true anomalies
        else:
            nus = Ms
        r = aR*(1-ecc**2)/(1+ecc*np.cos(nus))
        X = -r*np.cos(w*np.pi/180 + nus)
        Y = -r*np.sin(w*np.pi/180 + nus)*np.cos(inc)
        zs = np.sqrt(X**2 + Y**2)  #rsky
    
        if not sec:
            Mcenter = Ms[np.absolute(X[(np.sin(nus + w*np.pi/180) > 0)]).argmin()]
        else:
            Mcenter = Ms[np.absolute(X[(np.sin(nus + w*np.pi/180) < 0)]).argmin()]
        phs = (Ms - Mcenter) / (2*np.pi)
        wmin = np.absolute(phs).argmin()
        ts = phs*P

        return ts,zs
    else:
        if sec:
            f0 = -np.pi/2 - (w*np.pi/180)
            inc = np.arccos(b/aR*(1-ecc*np.sin(w*np.pi/180))/(1-ecc**2))
        else:
            f0 = np.pi/2 - (w*np.pi/180)
            inc = np.arccos(b/aR*(1+ecc*np.sin(w*np.pi/180))/(1-ecc**2))
        fmin = -np.arcsin(1./aR*np.sqrt(width**2 - b**2)/np.sin(inc))
        fmax = np.arcsin(1./aR*np.sqrt(width**2 - b**2)/np.sin(inc))
        if np.isnan(fmin) or np.isnan(fmax):
            raise NoEclipseError('no eclipse:  P=%.2f, b=%.3f, aR=%.2f, ecc=%0.2f, w=%.1f' % (P,b,aR,ecc,w))
        fs = np.linspace(fmin,fmax,npts)
        if sec:
            ts = fs*P/2./np.pi * np.sqrt(1-ecc**2)/(1 - ecc*np.sin(w)) #approximation of constant angular velocity
        else:
            ts = fs*P/2./np.pi * np.sqrt(1-ecc**2)/(1 + ecc*np.sin(w)) #approximation of constant ang. vel.
        fs += f0
        rs = aR*(1-ecc**2)/(1+ecc*np.cos(fs))
        xs = -rs*np.cos(w*np.pi/180 + fs)
        ys = -rs*np.sin(w*np.pi/180 + fs)*np.cos(inc)
        zs = aR*(1-ecc**2)/(1+ecc*np.cos(fs))*np.sqrt(1-(np.sin(w*np.pi/180 + fs))**2 * (np.sin(inc))**2)
        return ts,zs
