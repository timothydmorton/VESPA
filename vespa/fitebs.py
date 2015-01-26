#! /usr/bin/env python
from __future__ import print_function, division

import numpy as np
import logging
import pandas as pd

import transit_basic as tr
import sys, re, os
try:
    from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
except ImportError:
    logging.warning('progressbar not imported')

def fitebs(data, path='', MAfn=None, conv=True, use_pbar=True, msg='', 
           cadence=0.020434028):
    """Fits trapezoidal shape to eclispes described by parameters in data

    data is a pandas DataFrame.  Must have 'P', 'M1', 'M2', 'R1', 'R2',
    'inc', 'ecc', 'w', 'dpri', 'dsec', 'b_sec', 'b_pri', 'fluxfrac1', 
    'fluxfrac2', 'u11', 'u12', 'u21', 'u22'
    """
    n = len(data)


    p0s, bs, aRs = tr.eclipse_pars(data['P'], data['M1'], data['M2'],
                                   data['R1'], data['R2'], inc=data['inc'],
                                   ecc=data['ecc'], w=data['w'])

    deps, durs, slopes = (np.zeros(n), np.zeros(n), np.zeros(n))
    secs = np.zeros(n).astype(bool)
    dsec = np.zeros(n)

    if use_pbar:
        widgets = [msg+'fitting shape parameters for %i systems: ' % n,Percentage(),
                   ' ',Bar(marker=RotatingMarker()),' ',ETA()]
        pbar = ProgressBar(widgets=widgets,maxval=n)
        pbar.start()

    for i in arange(n):
        pri = (data['dpri'][i] > data['dsec'][i]) or np.isnan(data['dsec'][i])
        sec = not pri
        secs[i] = sec
        p0, aR = (p0s[i], aRs[i])
        if sec:
            b = data['b_sec'][i]
            frac = data['fluxfrac2'][i]
            dsec[i] = data['dpri'][i]
            u1 = data['u12'][i]
            u2 = data['u22'][i]
        else:
            b = data['b_pri'][i]
            frac = data['fluxfrac1'][i]
            dsec[i] = data['dsec'][i]
            u1 = data['u11'][i]
            u2 = data['u21'][i]
        try:
            if MAfn is not None:
                if p0 > MAfn.pmax or p0 < MAfn.pmin:
                    trap_pars = tr.eclipse_tt(p0,b,aR,data.P[i],conv=conv,MAfn=None,
                                              cadence=cadence, frac=frac,
                                              ecc=data.ecc[i],w=data.w[i],
                                              sec=sec,u1=u1,u2=u2)
                else:
                    trap_pars = tr.eclipse_tt(p0,b,aR,data.P[i],conv=conv,MAfn=MAfn,
                                              cadence=cadence, frac=frac,
                                              ecc=data.ecc[i],w=data.w[i],
                                              sec=sec,u1=u1,u2=u2)
            else:
                trap_pars = tr.eclipse_tt(p0,b,aR,data.P[i],conv=conv,MAfn=MAfn,
                                          cadence=cadence, frac=frac,
                                          ecc=data.ecc[i],w=data.w[i],
                                          sec=sec,u1=u1,u2=u2)
            logging.debug('{}'.format(trap_pars))
            
            durs[i], deps[i], slopes[i] = trap_pars
            if use_pbar:
                pbar.update(i)

        except tr.NoEclipseError:
            logging.error('No eclipse registered for index {}'.format(i))
            continue
        except tr.NoFitError:
            logging.error('Fit did not converge for index {}'.format(i))
            continue
        except:
            logging.error('unknown error for index {}'.format(i))
            continue

    return pd.DataFrame({'depth':deps, 'duration':durs,
                         'slope':slopes, 'secdepth':dsec,
                         'secondary':secs})
