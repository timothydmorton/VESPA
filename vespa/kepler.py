from __future__  import print_function, division
import numpy as np
import pandas as pd
import os, os.path
import logging
import pickle

from .transitsignal import TransitSignal
from .populations import PopulationSet
from .fpp import FPPCalculation
from keputils.koiutils import koiname
from keputils import koiutils as ku
from keputils import kicutils as kicu

from simpledist import distributions as dists

import kplr

KPLR_ROOT = os.getenv('KPLR_ROOT',os.path.expanduser('~/.kplr'))
JROWE_DIR = os.getenv('JROWE_DIR','~/.jrowe')

KOI_FPPDIR = os.getenv('KOI_FPPDIR',os.path.expanduser('~/.koifpp'))

#temporary, only local solution
CHAINSDIR = '{}/data/chains'.format(os.getenv('KEPLERDIR','~/.kepler'))

import astropy.constants as const
G = const.G.cgs.value
DAY = 86400


def koi_propdist(koi, prop):
    """
    """
    if type(koi) != kplr.api.KOI:
        client = kplr.API(data_root=KPLR_ROOT)
        koi = client.koi(koiname(koi, koinum=True))

    try:
        val = getattr(koi, prop)
        u1 = getattr(koi, prop+'_err1') #upper error bar (positive)
        u2 = getattr(koi, prop+'_err2') #lower error bar (negative)
        return dists.fit_doublegauss(val, -u2, u1)
    except:
        #try Huber table
        val = kicu.DATA.ix[koi.kepid, prop]
        u1 = kicu.DATA.ix[koi.kepid, prop+'_err1']
        u2 = kicu.DATA.ix[koi.kepid, prop+'_err2']
        return dists.fit_doublegauss(val, -u2, u1)

class KOI_FPPCalculation(FPPCalculation):
    def __init__(self, koi, recalc=False,
                 use_JRowe=True, trsig_kws=None,
                 tag=None,
                 **kwargs):

        koi = koiname(koi)

        if trsig_kws is None:
            trsig_kws = {}

        if use_JRowe:
            sig = JRowe_KeplerTransitSignal(koi, **trsig_kws)
        else:
            sig = KeplerTransitSignal(koi, **trsig_kws)

        #if saved popset exists, load
        folder = os.path.join(KOI_FPPDIR,koi)
        if tag is not None:
            folder += '_{}'.format(tag)

        popsetfile = os.path.join(folder,'popset.h5')
        if os.path.exists(popsetfile):
            popset = PopulationSet(popsetfile, **kwargs)

        else:
            client = kplr.API(data_root=data_root)
            koinum = koiname(koi, koinum=True)
            k = client.koi(koinum)

            if 'mass' not in kwargs:
                kwargs['mass'] = koi_propdist(k, 'mass')
            if 'radius' not in kwargs:
                kwargs['radius'] = koi_propdist(k, 'radius')
            if 'feh' not in kwargs:
                kwargs['feh'] = koi_propdist(k, 'feh')
            if 'age' not in kwargs:
                try:
                    kwargs['age'] = koi_propdist(k, 'age')
                except:
                    kwargs['age'] = (9.7,0.1) #default age
            if 'Teff' not in kwargs:
                kwargs['Teff'] = kicu.DATA.ix[k.kepid,'teff']
            if 'logg' not in kwargs:
                kwargs['logg'] = kicu.DATA.ix[k.kepid,'logg']
            if 'rprs' not in kwargs:
                if use_Jrowe:
                    kwargs['rprs'] = sig.rowefit.ix['RD1','val']
                else:
                    kwargs['rprs'] = k.koi_ror 
                    
            if 'mags' not in kwargs:
                kwargs['mags'] = ku.KICmags(koi)
            if 'ra' not in kwargs:
                kwargs['ra'], kwargs['dec'] = ku.radec(koi)
            if 'period' not in kwargs:
                kwargs['period'] = k.koi_period

            popset = PopulationSet(**kwargs)

        self.folder = folder
        lhoodcachefile = os.path.join(self.folder,'lhoodcache.dat')

        FPPCalculation.__init__(self, sig, popset,
                                lhoodcachefile=lhoodcachefile)
            

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
        # 2x duration of transit center, masking out any other
        # kois, etc., etc.
        
class JRowe_KeplerTransitSignal(KeplerTransitSignal):
    def __init__(self,koi,mcmc=True,maxslope=None,refit_mcmc=False,
                 **kwargs):

        self.folder = '%s/koi%i.n' % (JROWE_DIR,
                                      koiname(koi,star=True,
                                                 koinum=True))
        num = np.round(koiname(koi,koinum=True) % 1 * 100)

        self.lcfile = '%s/tremove.%i.dat' % (self.folder,num)
        if not os.path.exists(self.lcfile):
            raise MissingKOIError('{} does not exist.'.format(self.lcfile))
        logging.debug('Reading photometry from {}'.format(self.lcfile))

        #break if photometry file is empty
        if os.stat(self.lcfile)[6]==0:
            raise EmptyPhotometryError('{} photometry file ({}) is empty'.format(koiname(koi),
                                                                                  self.lcfile))

        lc = pd.read_table(self.lcfile,names=['t','f','df'],
                                                  delimiter='\s+')
        self.ttfile = '%s/koi%07.2f.tt' % (self.folder,koiname(koi,koinum=True))
        self.has_ttvs = os.path.exists(self.ttfile)
        if self.has_ttvs:            
            if os.stat(self.ttfile)[6]==0:
                self.has_ttvs = False
                logging.warning('TTV file exists for {}, but is empty.  No TTVs applied.'.format(koiname(koi)))
            else:
                logging.debug('Reading transit times from {}'.format(self.ttfile))
                tts = pd.read_table(self.ttfile,names=['tc','foo1','foo2'],delimiter='\s+')

        self.rowefitfile = '%s/n%i.dat' % (self.folder,num)

        self.rowefit = pd.read_table(self.rowefitfile,index_col=0,usecols=(0,1,3),
                                    names=['par','val','a','err','c'],
                                    delimiter='\s+')

        logging.debug('JRowe fitfile: {}'.format(self.rowefitfile))

        P = self.rowefit.ix['PE1','val']
        RR = self.rowefit.ix['RD1','val']
        aR = (self.rowefit.ix['RHO','val']*G*(P*DAY)**2/(3*np.pi))**(1./3)
        cosi = self.rowefit.ix['BB1','val']/aR
        Tdur = P*DAY/np.pi*np.arcsin(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5))/DAY

        if 1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5) > 1:
            logging.warning('arcsin argument in Tdur calculation > 1; setting to 1 for purposes of rough Tdur calculation...')
            Tdur = P*DAY/np.pi*np.arcsin(1)/DAY

        if (1+RR) < (self.rowefit.ix['BB1','val']):
            #Tdur = P*DAY/np.pi*np.arcsin(1/aR * (((1+RR)**2 - (aR*0)**2)/(1 - 0**2))**(0.5))/DAY/2.
            raise BadRoweFitError('best-fit impact parameter ({:.2f}) inconsistent with best-fit radius ratio ({}).'.format(self.rowefit.ix['BB1','val'],RR))

        if RR < 0:
            raise BadRoweFitError('{0} has negative RoR ({1}) from JRowe MCMC fit'.format(koiname(koi),RR))
        if RR > 1:
            raise BadRoweFitError('{0} has RoR > 1 ({1}) from JRowe MCMC fit'.format(koiname(koi),RR))            
        if aR < 1:
            raise BadRoweFitError('{} has a/Rstar < 1 ({}) from JRowe MCMC fit'.format(koiname(koi),aR))


        self.P = P
        self.aR = aR
        self.Tdur = Tdur
        self.epoch = self.rowefit.ix['EP1','val'] + 2504900

        logging.debug('Tdur = {:.2f}'.format(self.Tdur))
        logging.debug('aR={0}, cosi={1}, RR={2}'.format(aR,cosi,RR))
        logging.debug('arcsin arg={}'.format(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5)))
        logging.debug('inside sqrt in arcsin arg={}'.format((((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))))
        logging.debug('best-fit impact parameter={:.2f}'.format(self.rowefit.ix['BB1','val']))

        lc['t'] += (2450000+0.5)
        lc['f'] += 1 - self.rowefit.ix['ZPT','val']

        if self.has_ttvs:
            tts['tc'] += 2504900

        ts = pd.Series()
        fs = pd.Series()
        dfs = pd.Series()

        if self.has_ttvs:
            for t0 in tts['tc']:
                t = lc['t'] - t0
                ok = np.absolute(t) < 2*self.Tdur
                ts = ts.append(t[ok])
                fs = fs.append(lc['f'][ok])
                dfs = dfs.append(lc['df'][ok])
        else:
            center = self.epoch % self.P
            t = np.mod(lc['t'] - center + self.P/2,self.P) - self.P/2
            ok = np.absolute(t) < 2*self.Tdur
            ts = t[ok]
            fs = lc['f'][ok]
            dfs = lc['df'][ok]

        logging.debug('{0}: has_ttvs is {1}'.format(koi,self.has_ttvs))
        logging.debug('{} light curve points used'.format(ok.sum()))


        if maxslope is None:
            #set maxslope using duration
            maxslope = max(Tdur*24/0.5 * 2, 30) #hardcoded in transitFPP as default=30

        p0 = [Tdur,RR**2,3,0]
        self.p0 = p0
        logging.debug('initial trapezoid parameters guess: {}'.format(p0))
        TransitSignal.__init__(self,np.array(ts),np.array(fs),
                               np.array(dfs),p0=p0,
                               name=koiname(koi),
                               P=P,maxslope=maxslope)
        
        if mcmc:
            self.MCMC(refit=refit_mcmc)

        if self.hasMCMC and not self.fit_converged:
            logging.warning('Trapezoidal MCMC fit did not converge for {}.'.format(self.name))


    def MCMC(self,**kwargs):
        folder = '%s/%s' % (CHAINSDIR,self.name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        super(JRowe_KeplerTransitSignal,self).MCMC(savedir=folder,**kwargs)


###############Exceptions################

class BadPhotometryError(Exception):
    pass

class MissingKOIError(Exception):
    pass

class BadRoweFitError(Exception):
    pass

class EmptyPhotometryError(Exception):
    pass

class NoWeakSecondaryError(Exception):
    pass
