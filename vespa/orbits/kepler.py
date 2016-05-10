from __future__ import division,print_function

import os,os.path
import pkg_resources
import math

#test whether we're building documentation on readthedocs.org...
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    from numba import jit, vectorize
    import numpy as np
    from scipy.optimize import newton
    from scipy.interpolate import LinearNDInterpolator as interpnd
else:
    np, newton, interpnd = (None, None, None)
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
    DATAFOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    Es = np.load(os.path.join(DATAFOLDER,'Etable.npy'))
    eccs = np.load(os.path.join(DATAFOLDER,'Etable_eccs.npy'))
    Ms = np.load(os.path.join(DATAFOLDER,'Etable_Ms.npy'))
    ECCS,MS = np.meshgrid(eccs,Ms)
    points = np.array([MS.ravel(),ECCS.ravel()]).T
    EFN = interpnd(points,Es.ravel())
else:
    DATAFOLDER, Es, eccs, Ms, ECCS, MS = (None, None, None, None, None, None)
    points, EFN = (None, None) 
    
def Efn(Ms,eccs):
    """
    Returns Eccentric anomaly, interpolated from pre-computed grid of M, ecc

    Instantaneous solution of Kepler's equation!

    Works for ``-2*np.pi < Ms < 2*np.pi`` and ``eccs <= 0.97``
    
    :param Ms: (``float`` or array-like)
        Mean anomaly

    :param eccs: (``float`` or array-like)

    """
    Ms = np.atleast_1d(Ms)
    eccs = np.atleast_1d(eccs)
    unit = np.floor(Ms / (2*np.pi))
    Es = EFN((Ms % (2*np.pi)),eccs)
    Es += unit*(2*np.pi)
    return Es

@jit(nopython=True)
def calculate_eccentric_anomaly(mean_anomaly, eccentricity):
    done = False
    guess = mean_anomaly
    i = 0
    tol = 1e-5
    maxiter = 100
    while not done:
        f = guess - eccentricity * math.sin(guess) - mean_anomaly
        f_prime = 1 - eccentricity * math.cos(guess)
        newguess = guess - f/f_prime
        if abs(newguess - guess) < tol:
            done = True
        i += 1
        if i == maxiter:
            done = True
        guess = newguess

    return guess

@vectorize
def calculate_eccentric_anomalies(M, e):
    return calculate_eccentric_anomaly(M, e)

def calculate_eccentric_anomaly_old(mean_anomaly, eccentricity):

    def f(eccentric_anomaly_guess):
        return eccentric_anomaly_guess - eccentricity * math.sin(eccentric_anomaly_guess) - mean_anomaly

    def f_prime(eccentric_anomaly_guess):
        return 1 - eccentricity * math.cos(eccentric_anomaly_guess)

    return newton(f, mean_anomaly, f_prime, maxiter=100)

def calculate_eccentric_anomalies_old(eccentricity, mean_anomalies):
    def _calculate_one_ecc_anom(mean_anomaly):
        return calculate_eccentric_anomaly(mean_anomaly, eccentricity)

    vectorized_calculate = np.vectorize(_calculate_one_ecc_anom)
    return vectorized_calculate(mean_anomalies)

def Egrid(decc=0.01,dM=0.01):
    eccs = np.arange(0,0.98,decc)
    Ms = np.arange(0,2*pi,dM)
    Es = np.zeros((len(Ms),len(eccs)))
    i=0
    for e in eccs:
        Es[:,i] = calculate_eccentric_anomalies(e,Ms)
        i+=1
    Ms,eccs = np.meshgrid(Ms,eccs)
    return Ms.ravel(),eccs.ravel(),Es.ravel()


def writeEtable(emax=0.97,npts_e=200,npts_M=500):
    eccs = np.linspace(0,emax,npts_e)
    Ms = np.linspace(0,2*np.pi,npts_M)
    Es = np.zeros((len(Ms),len(eccs)))
    i=0
    for e in eccs:
        Es[:,i] = calculate_eccentric_anomalies(e,Ms)
        i+=1
    np.save(os.path.join(DATAFOLDER,'Etable.npy'),Es)
    np.save(os.path.join(DATAFOLDER,'Etable_eccs.npy'),eccs)
    np.save(os.path.join(DATAFOLDER,'Etable_Ms.npy'),Ms)

