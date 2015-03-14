from __future__ import division,print_function

import os,os.path
import pkg_resources
import numpy as np
import math
from scipy.optimize import newton
from scipy.interpolate import LinearNDInterpolator as interpnd


DATAFOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

Es = np.load(os.path.join(DATAFOLDER,'Etable.npy'))
eccs = np.load(os.path.join(DATAFOLDER,'Etable_eccs.npy'))
Ms = np.load(os.path.join(DATAFOLDER,'Etable_Ms.npy'))
ECCS,MS = np.meshgrid(eccs,Ms)
points = np.array([MS.ravel(),ECCS.ravel()]).T
EFN = interpnd(points,Es.ravel())

def Efn(Ms,eccs):
    """works for -2pi < Ms < 2pi, e <= 0.97"""
    Ms = np.atleast_1d(Ms)
    eccs = np.atleast_1d(eccs)
    unit = np.floor(Ms / (2*np.pi))
    Es = EFN((Ms % (2*np.pi)),eccs)
    Es += unit*(2*np.pi)
    return Es


def calculate_eccentric_anomaly(mean_anomaly, eccentricity):

    def f(eccentric_anomaly_guess):
        return eccentric_anomaly_guess - eccentricity * math.sin(eccentric_anomaly_guess) - mean_anomaly

    def f_prime(eccentric_anomaly_guess):
        return 1 - eccentricity * math.cos(eccentric_anomaly_guess)

    return newton(f, mean_anomaly, f_prime,maxiter=100)

def calculate_eccentric_anomalies(eccentricity, mean_anomalies):
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

