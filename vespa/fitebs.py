#! /usr/bin/env python
from __future__ import print_function, division

import logging
import sys, re, os

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np, pd = (None, None)

from .transit_basic import eclipse_pars, eclipse_tt
from .transit_basic import NoEclipseError, NoFitError
try:
    from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
    pbar_ok = True
except ImportError:
    logging.warning('progressbar not imported')
    pbar_ok = False

def fitebs(data, MAfn=None, conv=True, cadence=0.020434028,
           use_pbar=True, msg=''):
    """Fits trapezoidal shape to eclipses described by parameters in data

    Takes a few minutes for 20,000 simulations.

    :param data:
        Input ``DataFrame`` holding data.  Must have following columns:
        ``'P', 'mass_1', 'mass_2', 'radius_1', 'radius_2'``,
        ``'inc', 'ecc', 'w', 'dpri', 'dsec', 'b_sec', 'b_pri'``,
        ``'fluxfrac_1', 'fluxfrac_2', 'u1_1', 'u1_2', 'u2_1', 'u2_2'``.

    :param MAfn:
        :class:`MAInterpolationFunction` object; if not passed, it will be
        created.

    :param conv:
        Whether to convolve theoretical transit shape
        with box-car filter to simulate observation integration time.

    :param cadence:
        Integration time used if ``conv`` is ``True``.  Defaults to
        Kepler mission cadence.

    :param use_pbar:
        Whether to use nifty visual progressbar when doing calculation.

    :param msg:
        Message to print for pbar.

    :return df:
        Returns dataframe with the following columns:
        ``'depth', 'duration', 'slope'``,
        ``'secdepth', 'secondary'``.

    """
    n = len(data)


    p0s, bs, aRs = eclipse_pars(data['P'], data['mass_1'], data['mass_2'],
                                data['radius_1'], data['radius_2'],
                                inc=data['inc'],
                                ecc=data['ecc'], w=data['w'])

    deps, durs, slopes = (np.zeros(n), np.zeros(n), np.zeros(n))
    secs = np.zeros(n).astype(bool)
    dsec = np.zeros(n)

    if use_pbar and pbar_ok:
        widgets = [msg+'fitting shape parameters for %i systems: ' % n,Percentage(),
                   ' ',Bar(marker=RotatingMarker()),' ',ETA()]
        pbar = ProgressBar(widgets=widgets,maxval=n)
        pbar.start()

    for i in range(n):
        logging.debug('Fitting star {}'.format(i))
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
            logging.debug('p0={}, b={}, aR={}, P={}, frac={}, ecc={}, w={}, sec={}, u1={}, u2={}'.format(p0,b,aR,data['P'][i],frac,data['ecc'][i],data['w'][i],sec,u1,u2))
            logging.debug('dpri={}, dsec={}'.format(data['dpri'][i], data['dsec'][i]))
            trap_pars = eclipse_tt(p0,b,aR,data['P'][i],conv=conv,
                                   cadence=cadence, frac=frac,
                                   ecc=data['ecc'][i],w=data['w'][i],
                                   sec=sec,u1=u1,u2=u2)
            #logging.debug('{}'.format(trap_pars))

            durs[i], deps[i], slopes[i] = trap_pars
            if use_pbar and pbar_ok:
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
