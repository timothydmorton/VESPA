#! /usr/bin/env python
from __future__ import print_function, division

import numpy as np
import logging
import pandas as pd

import transit_basic as tr
from .transit_basic import eclipse_pars, eclipse_tt
from .transit_basic import NoEclipseError, NoFitError
import sys, re, os
try:
    from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
except ImportError:
    logging.warning('progressbar not imported')

def fitebs(data, MAfn=None, conv=True, use_pbar=True, msg='', 
           cadence=0.020434028):
    """Fits trapezoidal shape to eclispes described by parameters in data

    data is a pandas DataFrame.  Must have 'P', 'mass_1', 'mass_2', 
    'radius_1', 'radius_2',
    'inc', 'ecc', 'w', 'dpri', 'dsec', 'b_sec', 'b_pri', 'fluxfrac_1', 
    'fluxfrac_2', 'u1_1', 'u1_2', 'u2_1', 'u2_2'

    """
    n = len(data)


    p0s, bs, aRs = eclipse_pars(data['P'], data['mass_1'], data['mass_2'],
                                data['radius_1'], data['radius_2'], 
                                inc=data['inc'],
                                ecc=data['ecc'], w=data['w'])

    deps, durs, slopes = (np.zeros(n), np.zeros(n), np.zeros(n))
    secs = np.zeros(n).astype(bool)
    dsec = np.zeros(n)

    if use_pbar:
        widgets = [msg+'fitting shape parameters for %i systems: ' % n,Percentage(),
                   ' ',Bar(marker=RotatingMarker()),' ',ETA()]
        pbar = ProgressBar(widgets=widgets,maxval=n)
        pbar.start()

    for i in xrange(n):
        pri = (data['dpri'][i] > data['dsec'][i]) or np.isnan(data['dsec'][i])
        sec = not pri
        secs[i] = sec
        p0, aR = (p0s[i], aRs[i])
        if sec:
            b = data['b_sec'][i]
            frac = data['fluxfrac_2'][i]
            dsec[i] = data['dpri'][i]
            u1 = data['u1_2'][i]
            u2 = data['u2_2'][i]
        else:
            b = data['b_pri'][i]
            frac = data['fluxfrac_1'][i]
            dsec[i] = data['dsec'][i]
            u1 = data['u1_1'][i]
            u2 = data['u2_1'][i]
        try:
            if MAfn is not None:
                if p0 > MAfn.pmax or p0 < MAfn.pmin:
                    trap_pars = eclipse_tt(p0,b,aR,data['P'][i],conv=conv,MAfn=None,
                                              cadence=cadence, frac=frac,
                                              ecc=data['ecc'][i],w=data['w'][i],
                                              sec=sec,u1=u1,u2=u2)
                else:
                    trap_pars = eclipse_tt(p0,b,aR,data['P'][i],conv=conv,MAfn=MAfn,
                                              cadence=cadence, frac=frac,
                                              ecc=data['ecc'][i],w=data['w'][i],
                                              sec=sec,u1=u1,u2=u2)
            else:
                trap_pars = eclipse_tt(p0,b,aR,data['P'][i],conv=conv,MAfn=MAfn,
                                          cadence=cadence, frac=frac,
                                          ecc=data['ecc'][i],w=data['w'][i],
                                          sec=sec,u1=u1,u2=u2)
            #logging.debug('{}'.format(trap_pars))
            
            durs[i], deps[i], slopes[i] = trap_pars
            if use_pbar:
                pbar.update(i)

        except NoEclipseError:
            logging.error('No eclipse registered for index {}'.format(i))
            continue
        except NoFitError:
            logging.error('Fit did not converge for index {}'.format(i))
            continue
        except KeyboardInterrupt:
            raise
        except:
            logging.error('unknown error for index {}'.format(i))
            raise
            continue

    return pd.DataFrame({'depth':deps, 'duration':durs,
                         'slope':slopes, 'secdepth':dsec,
                         'secondary':secs})
