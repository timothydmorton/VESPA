from __future__  import print_function, division
import numpy as np
import pandas as pd
import os, os.path

from .transitsignal import TransitSignal
from keputils.koiutils import koiname

import kplr

KPLR_ROOT = os.getenv('KPLR_ROOT',os.path.expanduser('~/.kplr'))

class KeplerTransitSignal(TransitSignal):
    def __init__(self, koi, data_root=KPLR_ROOT):
        self.koi = koiname(koi)
        
        client = kplr.API(data_root=data_root)
        koinum = koiname(koi, koinum=True)
        k = client.koi(koinum)
        
        #get all data
        df = k.all_LCdata

        time = np.array(df['TIME'])
        flux = np.array(df['SAP_FLUX'])
        err = np.array(df['SAP_FLUX_ERR'])
        qual = np.array(df['SAP_QUALITY'])
        m = np.isfinite(time)*np.isfinite(flux)*np.isfinite(err)
        m *= qual==0

        time = time[m]
        flux = flux[m]
        err = err[m]

        period = k.koi_period
        epoch = k.koi_time0bk
        duration = k.koi_duration

        #create phase-folded, detrended time, flux, err, within 
        # 2x duration of transit center
        

