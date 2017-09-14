from __future__  import print_function, division
import numpy as np
import pandas as pd
import os, os.path, shutil
import re
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle

from pkg_resources import resource_filename

from configobj import ConfigObj

from scipy.integrate import quad

from tqdm import tqdm
from schwimmbad import choose_pool

from astropy.coordinates import SkyCoord

from isochrones.starmodel import StarModel
from isochrones.dartmouth import Dartmouth_Isochrone

from .transitsignal import TransitSignal
from .populations import PopulationSet
from .populations import fp_fressin
from .fpp import FPPCalculation

from .stars import get_AV_infinity

try:
    from keputils.koiutils import koiname
    from keputils import koiutils as ku
    from keputils import kicutils as kicu
except ImportError:
    logging.warning('keputils not available')

#from simpledist import distributions as dists

import kplr

KPLR_ROOT = os.getenv('KPLR_ROOT', os.path.expanduser('~/.kplr'))
JROWE_DIR = os.getenv('JROWE_DIR', os.path.expanduser('~/.jrowe'))
JROWE_FILE = resource_filename('vespa', 'data/jrowe_mcmc_fits.csv')
JROWE_DATA = pd.read_csv(JROWE_FILE, index_col=0)

KOI_FPPDIR = os.getenv('KOI_FPPDIR', os.path.expanduser('~/.koifpp'))
TTV_DIR = os.getenv('TTV_DIR', os.path.expanduser('~/.koi_ttv'))
STARFIELD_DIR = os.path.join(KOI_FPPDIR, 'starfields')
STARMODEL_DIR = os.path.join(KOI_FPPDIR, 'starmodels')


CHIPLOC_FILE = resource_filename('vespa','data/kepler_chiplocs.txt')

#temporary, only local solution
#CHAINSDIR = '{}/data/chains'.format(os.getenv('KEPLERDIR','~/.kepler'))
CHAINSDIR = os.path.join(KOI_FPPDIR, 'trap_chains')

DATAFOLDER = resource_filename('vespa','data')
# WEAKSECFILE = os.path.join(DATAFOLDER, 'weakSecondary_socv9p2vv.csv')
# WEAKSECDATA = pd.read_csv(WEAKSECFILE,skiprows=8)
# WEAKSECDATA.index = WEAKSECDATA['KOI'].apply(ku.koiname)

# ROBOVETFILE = os.path.join(DATAFOLDER, 'RoboVetter-Input.txt')
try:
    ROBOVETFILE = os.path.join(DATAFOLDER, 'dr25', 'kplr_dr25_obs_robovetter_input.h5')
    ROBOVETDATA = pd.read_hdf(ROBOVETFILE, 'df')
    ROBOVETDATA.index = ROBOVETDATA['TCE_ID']
except IOError:
    logging.warning("DR25 robovet data not available.")


KOI_MAXAV_FILE = os.path.join(DATAFOLDER, 'koi_maxAV.txt')
try:
    MAXAV = pd.read_table(KOI_MAXAV_FILE,
                          delim_whitespace=True,
                          names=['koi','maxAV'])
    MAXAV.index = MAXAV['koi']
except IOError:
    logging.warning('{} does not exist.'.format(KOI_MAXAV_FILE))
    MAXAV = None

import astropy.constants as const
G = const.G.cgs.value
DAY = 86400
RSUN = const.R_sun.cgs.value
REARTH = const.R_earth.cgs.value

KOIDATA = ku.DR25 #change to ku.DATA for cumulative table

def _get_starfields(**kwargs):
    from starutils.trilegal import get_trilegal
    if not os.path.exists(STARFIELD_DIR):
        os.makedirs(STARFIELD_DIR)
    chips, ras, decs = np.loadtxt(CHIPLOC_FILE, unpack=True)
    for i,ra,dec in zip(chips, ras, decs):
        print('field {} of {}'.format(i,chips[-1]))
        filename = '{}/kepler_starfield_{}'.format(STARFIELD_DIR,i)
        if not os.path.exists('{}.h5'.format(filename)):
            get_trilegal(filename, ra, dec, **kwargs)

def kepler_starfield_file(koi):
    """
    """
    ra,dec = ku.radec(koi)
    c = SkyCoord(ra,dec, unit='deg')
    chips,ras,decs = np.loadtxt(CHIPLOC_FILE,unpack=True)
    ds = ((c.ra.deg-ras)**2 + (c.dec.deg-decs)**2)
    chip = chips[np.argmin(ds)]
    return os.path.abspath('{}/kepler_starfield_{}.h5'.format(STARFIELD_DIR,chip))

def modelshift_weaksec(koi):
    """
    Max secondary depth based on model-shift secondary test from Jeff Coughlin

    secondary metric: mod_depth_sec_dv * (1 + 3*mod_fred_dv / mod_sig_sec_dv)
    """
    num = KOIDATA.ix[ku.koiname(koi), 'koi_tce_plnt_num']
    if np.isnan(num):
        num = 1
    kid = KOIDATA.ix[ku.koiname(koi), 'kepid']
    tce = '{:09.0f}-{:02.0f}'.format(kid,num)

    #return largest depth between DV detrending and alternate detrending
    try:
        r = ROBOVETDATA.ix[tce]
    except KeyError:
        raise NoWeakSecondaryError(koi)

    depth_dv = r['mod_depth_sec_dv'] * (1 + 3*r['mod_fred_dv'] / r['mod_sig_sec_dv'])
    depth_alt = r['mod_depth_sec_alt'] * (1 + 3*r['mod_fred_alt'] / r['mod_sig_sec_alt'])

    logging.debug(r[['mod_depth_sec_dv','mod_fred_dv','mod_sig_sec_dv']])
    logging.debug(r[['mod_depth_sec_alt','mod_fred_alt','mod_sig_sec_alt']])

    if np.isnan(depth_dv) and np.isnan(depth_alt):
        #return weaksec_vv2(koi)
        raise NoWeakSecondaryError(koi)
    elif np.isnan(depth_dv):
        return depth_alt
    elif np.isnan(depth_alt):
        return depth_dv
    else:
        return max(depth_dv, depth_alt)

def pipeline_weaksec(koi):
    try:
        val = modelshift_weaksec(koi)
    except NoWeakSecondaryError:
        val = weaksec_vv2(koi)

    #if val < 30e-6:
    #    val = 30e-6

    return val

def weaksec_vv2(koi):
    try:
        raise KeyError # just skip this, as WEAKSECDATA is obselete
        
        weaksec = WEAKSECDATA.ix[ku.koiname(koi)]
        secthresh = (weaksec['depth'] + 3*weaksec['e_depth'])*1e-6
        if weaksec['depth'] <= 0:
            raise KeyError

    except KeyError:
        koi = ku.koiname(koi)
        secthresh = 10*KOIDATA.ix[koi,'koi_depth_err1'] * 1e-6
        if np.isnan(secthresh):
            secthresh = KOIDATA.ix[koi,'koi_depth'] / 2 * 1e-6
            logging.warning('No (or bad) weak secondary info for {}, and no reported depth error. Defaulting to 1/2 reported depth = {}'.format(koi, secthresh))
        else:
            logging.warning('No (or bad) weak secondary info for {}. Defaulting to 10x reported depth error = {}'.format(koi, secthresh))

    if np.isnan(secthresh):
        raise NoWeakSecondaryError(koi)

    return secthresh


def default_r_exclusion(koi,rmin=0.5):
    try:
        r_excl = KOIDATA.ix[koi,'koi_dicco_msky_err'] * 3
        r_excl = max(r_excl, rmin)
        if np.isnan(r_excl):
            raise ValueError
    except:
        r_excl = 4
        logging.warning('No koi_dicco_msky_err info for {}. Defaulting to 4 arcsec.'.format(koi))

    return r_excl

def koi_maxAV(koi):
    try:
        if MAXAV is None:
            raise KeyError
        maxAV = MAXAV.ix[ku.koiname(koi),'maxAV']
    except KeyError:
        ra,dec = ku.radec(koi)
        maxAV = get_AV_infinity(ra,dec)
    return maxAV


def _getAV(k):
    return get_AV_infinity(*ku.radec(k))

def _generate_koi_maxAV_table(procs=1):
    kois = np.array(ku.DR25.index)
    pool = choose_pool(mpi=False, processes=procs)
    maxAV = pool.map(_getAV, kois)

    np.savetxt(KOI_MAXAV_FILE, np.array([kois, maxAV]).T, fmt='%s %.3f')


def koi_propdist(koi, prop):
    """
    """
    koi = ku.koiname(koi)
    kepid = KOIDATA.ix[koi, 'kepid']
    try:
        #first try cumulative table
        val = KOIDATA.ix[koi, prop]
        u1 = KOIDATA.ix[koi, prop+'_err1']
        u2 = KOIDATA.ix[koi, prop+'_err2']
    except KeyError:
        try:
            #try Huber table
            val = kicu.DATA.ix[kepid, prop]
            u1 = kicu.DATA.ix[kepid, prop+'_err1']
            u2 = kicu.DATA.ix[kepid, prop+'_err2']
        except KeyError:
            raise NoStellarPropError(koi)
    if np.isnan(val) or np.isnan(u2) or np.isnan(u1):
        raise MissingStellarPropError('{}: {} = ({},{},{})'.format(koi,
                                                                   prop,
                                                                   val,u1,u2))
    try:
        return dists.fit_doublegauss(val, -u2, u1)
    except:
        raise StellarPropError('{}: {} = ({},{},{})'.format(koi,
                                                            prop,
                                                            val,u1,u2))

class KOI_FPPCalculation(FPPCalculation):
    def __init__(self, koi, recalc=False,
                 use_JRowe=True, trsig_kws=None,
                 tag=None, starmodel_mcmc_kws=None,
                 **kwargs):

        koi = koiname(koi)

        #if saved popset exists, load
        folder = os.path.join(KOI_FPPDIR,koi)
        if tag is not None:
            folder += '_{}'.format(tag)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if trsig_kws is None:
            trsig_kws = {}

        #first check if pickled signal is there to be loaded
        trsigfile = os.path.join(folder,'trsig.pkl')
        if os.path.exists(trsigfile):
            with open(trsigfile, 'rb') as f:
                trsig = pickle.load(f)
        else:
            if use_JRowe:
                trsig = JRowe_KeplerTransitSignal(koi, **trsig_kws)
            else:
                trsig = KeplerTransitSignal(koi, **trsig_kws)

        popsetfile = os.path.join(folder,'popset.h5')
        if os.path.exists(popsetfile) and not recalc:
            popset = PopulationSet(popsetfile, **kwargs)

        else:
            koinum = koiname(koi, koinum=True)
            kepid = KOIDATA.ix[koi,'kepid']

            if 'mass' not in kwargs:
                kwargs['mass'] = koi_propdist(koi, 'mass')
            if 'radius' not in kwargs:
                kwargs['radius'] = koi_propdist(koi, 'radius')
            if 'feh' not in kwargs:
                kwargs['feh'] = koi_propdist(koi, 'feh')
            if 'age' not in kwargs:
                try:
                    kwargs['age'] = koi_propdist(koi, 'age')
                except:
                    kwargs['age'] = (9.7,0.1) #default age
            if 'Teff' not in kwargs:
                kwargs['Teff'] = kicu.DATA.ix[kepid,'teff']
            if 'logg' not in kwargs:
                kwargs['logg'] = kicu.DATA.ix[kepid,'logg']
            if 'rprs' not in kwargs:
                if use_JRowe:
                    kwargs['rprs'] = trsig.rowefit.ix['RD1','val']
                else:
                    kwargs['rprs'] = KOIDATA.ix[koi,'koi_ror']

            #if stellar properties are determined spectroscopically,
            # fit stellar model
            if 'starmodel' not in kwargs:
                if re.match('SPE', kicu.DATA.ix[kepid, 'teff_prov']):
                    logging.info('Spectroscopically determined stellar properties.')
                    #first, see if there already is a starmodel to load

                    #fit star model
                    Teff = kicu.DATA.ix[kepid, 'teff']
                    e_Teff = kicu.DATA.ix[kepid, 'teff_err1']
                    logg = kicu.DATA.ix[kepid, 'logg']
                    e_logg = kicu.DATA.ix[kepid, 'logg_err1']
                    feh = kicu.DATA.ix[kepid, 'feh']
                    e_feh = kicu.DATA.ix[kepid, 'feh_err1']
                    logging.info('fitting StarModel (Teff=({},{}), logg=({},{}), feh=({},{}))...'.format(Teff, e_Teff, logg, e_logg, feh, e_feh))

                    dar = Dartmouth_Isochrone()
                    starmodel = StarModel(dar, Teff=(Teff, e_Teff),
                                          logg=(logg, e_logg),
                                          feh=(feh, e_feh))
                    if starmodel_mcmc_kws is None:
                        starmodel_mcmc_kws = {}
                    starmodel.fit(**starmodel_mcmc_kws)
                    logging.info('Done.')
                    kwargs['starmodel'] = starmodel



            if 'mags' not in kwargs:
                kwargs['mags'] = ku.KICmags(koi)
            if 'ra' not in kwargs:
                kwargs['ra'], kwargs['dec'] = ku.radec(koi)
            if 'period' not in kwargs:
                kwargs['period'] = KOIDATA.ix[koi,'koi_period']

            if 'pl_kws' not in kwargs:
                kwargs['pl_kws'] = {}

            if 'fp_specific' not in kwargs['pl_kws']:
                rp = kwargs['radius'].mu * kwargs['rprs'] * RSUN/REARTH
                kwargs['pl_kws']['fp_specific'] = fp_fressin(rp)

            #trilegal_filename = os.path.join(folder,'starfield.h5')
            trilegal_filename = kepler_starfield_file(koi)
            popset = PopulationSet(trilegal_filename=trilegal_filename,
                                   **kwargs)
            #popset.save_hdf('{}/popset.h5'.format(folder), overwrite=True)


        lhoodcachefile = os.path.join(folder,'lhoodcache.dat')
        self.koi = koi
        FPPCalculation.__init__(self, trsig, popset,
                                folder=folder)
        self.save()
        self.apply_default_constraints()

    def apply_default_constraints(self):
        """Applies default secthresh & exclusion radius constraints
        """
        try:
            self.apply_secthresh(pipeline_weaksec(self.koi))
        except NoWeakSecondaryError:
            logging.warning('No secondary eclipse threshold set for {}'.format(self.koi))
        self.set_maxrad(default_r_exclusion(self.koi))


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

def jrowe_fit_old(koi):
    koinum = koiname(koi, star=True, koinum=True)
    folder = os.path.join(JROWE_DIR, 'koi{}.n'.format(koinum))

    num = np.round(koiname(koi,koinum=True) % 1 * 100)
    fitfile = os.path.join(folder, 'n{:.0f}.dat'.format(num))

    logging.debug('JRowe fitfile: {}'.format(fitfile))

    return pd.read_table(fitfile,index_col=0,usecols=(0,1,3),
                         names=['par','val','a','err','c'],
                         delimiter='\s+')

def jrowe_fit(koi):
    return jrowe_fit_old(koi)

    koi = koiname(koi)

    pars = ['RHO', 'EP1','PE1', 'BB1', 'RD1']
    vals = list(JROWE_DATA.ix[koi,['rhostar','T0','P','b','rdr']])

    old = jrowe_fit_old(koi)
    vals[1] = old.ix['EP1','val'] # hack to make sure epoch matches old photometry.

    return pd.DataFrame({'val':vals}, index=pars)

class JRowe_KeplerTransitSignal(KeplerTransitSignal):
    def __init__(self,koi,mcmc=True,maxslope=None,refit_mcmc=False,
                 **kwargs):
        self.folder = '%s/koi%i.n' % (JROWE_DIR,
                                      koiname(koi,star=True,
                                                 koinum=True))
        num = np.round(koiname(koi,koinum=True) % 1 * 100)

        self.lcfile = '%s/tremove.%i.dat' % (self.folder,num)
        if (not os.path.exists(self.lcfile)) or (os.stat(self.lcfile)[6]==0):
            kepid = ku.kepid(koi)
            self.lcfile = '{}/klc{:08.0f}.dct.dat'.format(self.folder,kepid)
            if not os.path.exists(self.lcfile):
                raise MissingKOIError('{} does not exist.'.format(self.lcfile))
            if os.stat(self.lcfile)[6]==0:
                raise EmptyPhotometryError('{} photometry file ({}) is empty'.format(koiname(koi),
                                                                                  self.lcfile))

        logging.debug('Reading photometry from {}'.format(self.lcfile))

        lc = pd.read_table(self.lcfile,names=['t','f','df'],
                                                  delimiter='\s+')

        logging.debug('{} points read from file.'.format(len(lc)))

        self.ttfile = '%s/koi%07.2f.tt' % (TTV_DIR, koiname(koi, koinum=True))
        if not os.path.exists(self.ttfile):
            self.ttfile = '%s/koi%07.2f.tt' % (self.folder,
                                               koiname(koi, koinum=True))
        self.has_ttvs = os.path.exists(self.ttfile)
        if self.has_ttvs:
            if os.stat(self.ttfile)[6]==0:
                self.has_ttvs = False
                logging.warning('TTV file exists for {}, but is empty.  No TTVs applied.'.format(koiname(koi)))
            else:
                logging.debug('Reading transit times from {}'.format(self.ttfile))
                tts = pd.read_table(self.ttfile,names=['tc','C-O','e_C-O'],
                                    delimiter='\s+',comment='#' )
                if tts['C-O'].std() < tts['e_C-O'].mean():
                    self.has_ttvs = False
                    logging.warning('TTV file exists for {}, but errors are too large.  No TTVs applied.'.format(koiname(koi)))

        #self.rowefitfile = '%s/n%i.dat' % (self.folder,num)

        #self.rowefit = pd.read_table(self.rowefitfile,index_col=0,usecols=(0,1,3),
        #                            names=['par','val','a','err','c'],
        #                            delimiter='\s+')
        self.rowefit = jrowe_fit(koi)


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

        logging.debug('Tdur = {:.2f}, P={:.4f}, ep={:.4f}'.format(self.Tdur,self.P,self.epoch))
        logging.debug('aR={0}, cosi={1}, RR={2}'.format(aR,cosi,RR))
        logging.debug('arcsin arg={}'.format(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5)))
        logging.debug('inside sqrt in arcsin arg={}'.format((((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))))
        logging.debug('best-fit impact parameter={:.2f}'.format(self.rowefit.ix['BB1','val']))

        lc['t'] += (2450000+0.5)
        lc['f'] += 1 # - self.rowefit.ix['ZPT','val']

        if self.has_ttvs:
            tts['tc'] += 2504900

        ts = pd.Series()
        fs = pd.Series()
        dfs = pd.Series()

        if self.has_ttvs:
            for t0 in tts['tc'] + tts['C-O']:
                t = lc['t'] - t0
                ok = np.absolute(t) < (2*self.Tdur)
                ts = ts.append(t[ok])
                fs = fs.append(lc['f'][ok])
                dfs = dfs.append(lc['df'][ok])
        else:
            center = self.epoch % self.P
            t = np.mod(lc['t'] - center + self.P/2,self.P) - self.P/2
            ok = np.absolute(t) < (2*self.Tdur)
            ts = t[ok]
            fs = lc['f'][ok]
            dfs = lc['df'][ok]

        logging.debug('{0}: has_ttvs is {1}'.format(koi,self.has_ttvs))
        logging.debug('{} light curve points used'.format(ok.sum()))

        if ok.sum()==0:
            logging.debug(t)
            logging.debug('no t points (above) with abs < {}'.format(2*self.Tdur))
            raise BadPhotometryError('No valid light curve points?')

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
            os.makedirs(folder)
        try:
            super(JRowe_KeplerTransitSignal,self).MCMC(savedir=folder,**kwargs)
        except IOError:
            shutil.rmtree(folder)
            os.makedirs(folder)
            super(JRowe_KeplerTransitSignal,self).MCMC(savedir=folder,**kwargs)

def use_property(kepid, prop):
    """Returns true if provenance of property is SPE or AST
    """
    try:
        prov = kicu.DATA.ix[kepid, '{}_prov'.format(prop)]
        return any([prov.startswith(s) for s in ['SPE', 'AST']])
    except KeyError:
        raise MissingStellarError('{} not in stellar table?'.format(kepid))


def star_config(koi, bands=['g','r','i','z','J','H','K'],
                unc=dict(g=0.05, r=0.05, i=0.05, z=0.05,
                         J=0.02, H=0.02, K=0.02), **kwargs):

    """returns star config object for given KOI
    """
    folder = os.path.join(KOI_FPPDIR, ku.koiname(koi))
    if not os.path.exists(folder):
        os.makedirs(folder)

    config = ConfigObj(os.path.join(folder,'star.ini'))

    koi = ku.koiname(koi)

    maxAV = koi_maxAV(koi)
    config['maxAV'] = maxAV

    mags = ku.KICmags(koi)
    for band in bands:
        if not np.isnan(mags[band]):
            config[band] = (mags[band], unc[band])
    config['Kepler'] = mags['Kepler']

    kepid = KOIDATA.ix[koi,'kepid']

    if use_property(kepid, 'teff'):
        teff, e_teff = (kicu.DATA.ix[kepid, 'teff'],
                          kicu.DATA.ix[kepid, 'teff_err1'])
        if not any(np.isnan([teff, e_teff])):
            config['Teff'] = (teff, e_teff)

    if use_property(kepid, 'logg'):
        logg, e_logg = (kicu.DATA.ix[kepid, 'logg'],
                          kicu.DATA.ix[kepid, 'logg_err1'])
        if not any(np.isnan([logg, e_logg])):
            config['logg'] = (logg, e_logg)

    if use_property(kepid, 'feh'):
        feh, e_feh = (kicu.DATA.ix[kepid, 'feh'],
                          kicu.DATA.ix[kepid, 'feh_err1'])
        if not any(np.isnan([feh, e_feh])):
            config['feh'] = (feh, e_feh)

    for kw,val in kwargs.items():
        config[kw] = val

    return config

def fpp_config(koi, **kwargs):
    """returns config object for given KOI
    """
    folder = os.path.join(KOI_FPPDIR, ku.koiname(koi))
    if not os.path.exists(folder):
        os.makedirs(folder)
    config = ConfigObj(os.path.join(folder,'fpp.ini'))

    koi = ku.koiname(koi)

    rowefit = jrowe_fit(koi)

    config['name'] = koi
    ra,dec = ku.radec(koi)
    config['ra'] = ra
    config['dec'] = dec
    config['rprs'] = rowefit.ix['RD1','val']
    config['period'] = rowefit.ix['PE1', 'val']

    config['starfield'] = kepler_starfield_file(koi)

    for kw,val in kwargs.items():
        config[kw] = val

    config['constraints'] = {}
    config['constraints']['maxrad'] = default_r_exclusion(koi)
    try:
        config['constraints']['secthresh'] = pipeline_weaksec(koi)
    except NoWeakSecondaryError:
        pass

    return config

def setup_fpp(koi, bands=['g','r','i','z','J','H','K'],
              unc=dict(g=0.05, r=0.05, i=0.05, z=0.05,
                       J=0.02, H=0.02, K=0.02),
              star_kws=None, fpp_kws=None, trsig_kws=None,
              trsig_overwrite=False,
              star_only=False, fpp_only=False):
    if star_kws is None:
        star_kws = {}
    if fpp_kws is None:
        fpp_kws = {}
    if trsig_kws is None:
        trsig_kws = {}

    if not star_only:
        #save transit signal
        folder = os.path.join(KOI_FPPDIR, ku.koiname(koi))
        trsig_file = os.path.join(folder,'trsig.pkl')
        if os.path.exists(trsig_file):
            if os.path.getsize(trsig_file)==0:
                os.remove(trsig_file)
        if not os.path.exists(trsig_file) or\
                trsig_overwrite:
            sig = JRowe_KeplerTransitSignal(koi, refit_mcmc=True,
                                            **trsig_kws)
            sig.save(os.path.join(folder,'trsig.pkl'))
        fpp = fpp_config(koi, **fpp_kws)
        fpp.write()

    if not fpp_only:
        star = star_config(koi, bands=bands, unc=unc, **star_kws)
        star.write()



###############Exceptions################

class BadPhotometryError(Exception):
    pass

class MissingKOIError(Exception):
    pass

class MissingStellarError(Exception):
    pass

class BadRoweFitError(Exception):
    pass

class EmptyPhotometryError(Exception):
    pass

class NoWeakSecondaryError(Exception):
    pass

class NoStellarPropError(Exception):
    pass

class MissingStellarPropError(Exception):
    pass

class StellarPropError(Exception):
    pass
