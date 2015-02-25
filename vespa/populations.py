from __future__ import print_function, division

import logging

import numpy as np
import os, os.path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from plotutils import setfig, plot2dhist
from hashutils import hashcombine

from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

from .transit_basic import occultquad, ldcoeffs, minimum_inclination
from .transit_basic import MAInterpolationFunction
from .fitebs import fitebs

from starutils.populations import StarPopulation, MultipleStarPopulation
from starutils.populations import ColormatchMultipleStarPopulation
from starutils.populations import Spectroscopic_MultipleStarPopulation
from starutils.populations import BGStarPopulation, BGStarPopulation_TRILEGAL
from starutils.populations import DARTMOUTH
from starutils.utils import draw_eccs, semimajor, withinroche
from starutils.utils import mult_masses, randpos_in_circle
from starutils.utils import fluxfrac, addmags
from starutils.utils import RAGHAVAN_LOGPERKDE

from starutils.constraints import UpperLimit

import simpledist.distributions as dists

from orbitutils.populations import OrbitPopulation_FromDF, TripleOrbitPopulation_FromDF

SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}
                        
INV_SHORT_MODELNAMES = {v:k for k,v in SHORT_MODELNAMES.iteritems()}

from astropy.units import Quantity
import astropy.units as u
import astropy.constants as const
AU = const.au.cgs.value
RSUN = const.R_sun.cgs.value
MSUN = const.M_sun.cgs.value
G = const.G.cgs.value
REARTH = const.R_earth.cgs.value
MEARTH = const.M_earth.cgs.value


class EclipsePopulation(StarPopulation):
    def __init__(self, stars=None, period=None, model='',
                 priorfactors=None, lhoodcachefile=None,
                 orbpop=None, prob=None,
                 cadence=0.020434028, #Kepler observing cadence, in days
                 **kwargs):
        """Base class for populations of eclipsing things.

        stars DataFrame must have the following columns:
        'M_1', 'M_2', 'R_1', 'R_2', 'u1_1', 'u1_2', 'u2_1', 'u2_2', and 
        either the P keyword argument provided or a `period column, as 
        well as the eclipse parameters: 'inc', 'ecc', 'w', 'dpri', 
        'dsec', 'b_sec', 'b_pri', 'fluxfrac_1', 'fluxfrac_2'

        For some functionality, also needs to have trapezoid fit 
        parameters in DataFrame
        
        if prob is not passed; should be able to calculated from given
        star/orbit properties.
        """
        
        self.period = period
        self.model = model
        if priorfactors is None:
            priorfactors = {}
        self.priorfactors = priorfactors
        self.prob = prob #calculate this if not provided?
        self.cadence = cadence
        self.lhoodcachefile = lhoodcachefile
        self.is_specific = False

        StarPopulation.__init__(self, stars=stars, orbpop=orbpop, 
                                name=model, **kwargs)
        
        if stars is not None:
            if len(self.stars)==0:
                raise EmptyPopulationError('Zero elements in {} population'.format(model))

        if 'slope' in self.stars:
            self._make_kde()

    def fit_trapezoids(self, MAfn=None, msg=None, **kwargs):
        logging.info('Fitting trapezoid models for {}...'.format(self.model))
        if MAfn is None:
            MAfn = MAInterpolationFunction(nzs=200,nps=400,pmin=0.007,pmax=1/0.007)
        if msg is None:
            msg = '{}: '.format(self.model)
        trapfit_df = fitebs(self.stars, MAfn=MAfn, msg=msg, cadence=self.cadence,
                            **kwargs)
        for col in trapfit_df.columns:
            self.stars[col] = trapfit_df[col]

        logging.info('Done.')
        self._make_kde()


    def apply_multicolor_transit(self, band, depth):
        raise NotImplementedError('multicolor transit not yet implemented')

    @property
    def eclipseprob(self):
        """
        """
        #TODO: incorporate eccentricity/omega for exact calculation?
        s = self.stars
        return ((s['radius_1'] + s['radius_2'])*RSUN / 
                (semimajor(s['P'],s['mass_1'] + s['mass_2'])*AU))

    @property
    def mean_eclipseprob(self):
        return self.eclipseprob.mean()

    @property
    def modelshort(self):
        try:
            name = SHORT_MODELNAMES[self.model]
            
            #add index if specific model is indexed
            if hasattr(self,'index'):
                name += '-{}'.format(self.index)

            return name

        except KeyError:
            raise KeyError('No short name for model: %s' % self.model)        

    @property
    def dilution_factor(self):
        return np.ones(len(self.stars))

    @property
    def depth(self):
        return self.dilution_factor * self.stars['depth']

    @property
    def secondary_depth(self):
        return self.dilution_factor * self.stars['secdepth']

    def constrain_secdepth(self, thresh):
        self.apply_constraint(UpperLimit(self.secondary_depth, thresh, name='secondary depth'))

    def apply_secthresh(self, *args, **kwargs):
        return self.constrain_secdepth(*args, **kwargs)

    def fluxfrac_eclipsing(self, band=None):
        pass

    def depth_in_band(self, band):
        pass

    @property
    def prior(self):
        prior = self.prob * self.selectfrac
        for f in self.priorfactors:
            prior *= self.priorfactors[f]
        return prior

    def add_priorfactor(self,**kwargs):
        for kw in kwargs:
            if kw in self.priorfactors:
                logging.error('%s already in prior factors for %s.  use change_prior function instead.' % (kw,self.model))
                continue
            else:
                self.priorfactors[kw] = kwargs[kw]
                logging.info('%s added to prior factors for %s' % (kw,self.model))

    def change_prior(self, **kwargs):
        for kw in kwargs:
            if kw in self.priorfactors:
                self.priorfactors[kw] = kwargs[kw]
                logging.info('{0} changed to {1} for {2} model'.format(kw,kwargs[kw],
                                                                       self.model))

    def _make_kde(self, use_sklearn=False, bandwidth=None, rtol=1e-6,
                  **kwargs):
        """Creates KDE objects for 3-d shape parameter distribution

        Uses scikit-learn KDE by default

        Keyword arguments passed to gaussian_kde
        """

        try:
            #define points that are ok to use
            ok = ((self.stars['slope'] > 0) & (self.stars['duration'] > 0) & 
                  (self.stars['duration'] < self.period) & (self.depth > 0) &
                  self.distok)
        except KeyError:
            logging.warning('Must do trapezoid fits before making KDE.')
            return
        
        if ok.sum() < 4:
            raise EmptyPopulationError('< 4 valid systems in population')

        deps = self.depth[ok]
        logdeps = np.log10(deps)
        durs = self.stars['duration'][ok]
        slopes = self.stars['slope'][ok]

        if use_sklearn:
            self.sklearn_kde = True
            logdeps_normed = (logdeps - logdeps.mean())/logdeps.std()
            durs_normed = (durs - durs.mean())/durs.std()
            slopes_normed = (slopes - slopes.mean())/slopes.std()

            #TODO: use sklearn preprocessing to replace below
            self.mean_logdepth = logdeps.mean()
            self.std_logdepth = logdeps.std()
            self.mean_dur = durs.mean()
            self.std_dur = durs.std()
            self.mean_slope = slopes.mean()
            self.std_slope = slopes.std()

            points = np.array([logdeps_normed, durs_normed, slopes_normed])

            #find best bandwidth.  For some reason this doesn't work?
            if bandwidth is None:
                grid = GridSearchCV(KernelDensity(rtol=rtol), 
                                    {'bandwidth':np.linspace(0.05,1,50)})
                grid.fit(points)
                self._best_bandwidth = grid.best_params_
                self.kde = grid.best_estimator_
            else:
                self.kde = KernelDensity(rtol=rtol, bandwidth=bandwidth).fit(points)
        else:
            self.sklearn_kde = False
            points = np.array([durs, logdeps, slopes])
            self.kde = gaussian_kde(points, **kwargs)
                
    def _density(self, logd, dur, slope):
        """
        """
        if self.sklearn_kde:
            #TODO: fix preprocessing
            pts = np.array([(logd - self.mean_logdepth)/self.std_logdepth,
                            (dur - self.mean_dur)/self.std_dur,
                            (slope - self.mean_slope)/self.std_slope])
            return self.kde.score_samples(pts)
        else:
            return self.kde(np.array([logd, dur, slope]))

    def lhood(self, trsig, recalc=False, cachefile=None):
        """Returns likelihood of transit signal
        """
        if cachefile is None:
            cachefile = self.lhoodcachefile
            if cachefile is None:
                cachefile = 'lhoodcache.dat'

        lhoodcache = loadcache(cachefile)
        key = hashcombine(self, trsig)
        if key in lhoodcache and not recalc:
            return lhoodcache[key] 
            
        if self.is_ruled_out:
            return 0

        lh = self.kde(trsig.kde.dataset).sum()

        fout = open(cachefile, 'a')
        fout.write('%i %g\n' % (key,lh))
        fout.close

        return lh
        
        
    def lhoodplot(self, trsig=None, fig=None, label='', plotsignal=False, 
                  piechart=True, figsize=None, logscale=True,
                  constraints='all', suptitle=None, Ltot=None,
                  maxdur=None, maxslope=None, inverse=False, 
                  colordict=None, cachefile=None, nbins=20,
                  dur_range=None, slope_range=None, depth_range=None,
                  **kwargs):
        setfig(fig, figsize=figsize)

        if trsig is not None:
            dep,ddep = trsig.logdepthfit
            dur,ddur = trsig.durfit
            slope,dslope = trsig.slopefit

            ddep = ddep.reshape((2,1))
            ddur = ddur.reshape((2,1))
            dslope = dslope.reshape((2,1))
            
            if dur_range is None:
                dur_range = (0,dur*2)
            if slope_range is None:
                slope_range = (2,slope*2)

        if constraints == 'all':
            mask = self.distok
        elif constraints == 'none':
            mask = np.ones(len(self.stars)).astype(bool)
        else:
            mask = np.ones(len(self.stars)).astype(bool)
            for c in constraints:
                if c not in self.distribution_skip:
                    mask &= self.constraints[c].ok

        if inverse:
            mask = ~mask

        if dur_range is None:
            dur_range = (self.stars[mask]['duration'].min(),
                         self.stars[mask]['duration'].max())
        if slope_range is None:
            slope_range = (2,self.stars[mask]['slope'].max())
        if depth_range is None:
            depth_range = (-5,-0.1)

        #This may mess with intended "inverse" behavior, probably?
        mask &= ((self.stars['duration'] > dur_range[0]) & 
                 (self.stars['duration'] < dur_range[1]))
        mask &= ((self.stars['duration'] > dur_range[0]) & 
                 (self.stars['duration'] < dur_range[1]))

        mask &= ((self.stars['slope'] > slope_range[0]) & 
                 (self.stars['slope'] < slope_range[1]))
        mask &= ((self.stars['slope'] > slope_range[0]) & 
                 (self.stars['slope'] < slope_range[1]))

        mask &= ((np.log10(self.depth) > depth_range[0]) & 
                 (np.log10(self.depth) < depth_range[1]))
        mask &= ((np.log10(self.depth) > depth_range[0]) & 
                 (np.log10(self.depth) < depth_range[1]))




        if piechart:
            a_pie = plt.axes([0.07, 0.5, 0.4, 0.5])
            self.constraint_piechart(fig=0, colordict=colordict)
            
        ax1 = plt.subplot(222)
        if not self.is_ruled_out:
            self.prophist2d('duration', 'depth', logy=True, fig=0,
                            mask=mask, interpolation='bicubic', 
                            logscale=logscale, nbins=nbins, **kwargs)
        if trsig is not None:
            plt.errorbar(dur,dep,xerr=ddur,yerr=ddep,color='w',marker='x',
                         ms=12,mew=3,lw=3,capsize=3,mec='w')  
            plt.errorbar(dur,dep,xerr=ddur,yerr=ddep,color='r',marker='x',
                         ms=10,mew=1.5)
        plt.ylabel(r'log($\delta$)')
        plt.xlabel('')
        plt.xlim(dur_range)
        plt.ylim(depth_range)
        yt = ax1.get_yticks()
        plt.yticks(yt[1:])
        xt = ax1.get_xticks()
        plt.xticks(xt[2:-1:2])

        ax3 = plt.subplot(223)
        if not self.is_ruled_out:
            self.prophist2d('depth', 'slope', logx=True, fig=0,
                            mask=mask, interpolation='bicubic', 
                            logscale=logscale, nbins=nbins, **kwargs)
        if trsig is not None:
            plt.errorbar(dep,slope,xerr=ddep,yerr=dslope,color='w',marker='x',
                         ms=12,mew=3,lw=3,capsize=3,mec='w')
            plt.errorbar(dep,slope,xerr=ddep,yerr=dslope,color='r',marker='x',
                         ms=10,mew=1.5)               
        plt.ylabel(r'$T/\tau$')
        plt.xlabel(r'log($\delta$)')
        plt.ylim(slope_range)
        plt.xlim(depth_range)
        yt = ax3.get_yticks()
        plt.yticks(yt[1:])

        ax4 = plt.subplot(224)
        if not self.is_ruled_out:
            self.prophist2d('duration', 'slope', fig=0,
                            mask=mask, interpolation='bicubic', 
                            logscale=logscale, nbins=nbins, **kwargs)
        if trsig is not None:
            plt.errorbar(dur,slope,xerr=ddur,yerr=dslope,color='w',marker='x',
                         ms=12,mew=3,lw=3,capsize=3,mec='w')   
            plt.errorbar(dur,slope,xerr=ddur,yerr=dslope,color='r',marker='x',
                         ms=10,mew=1.5)               
        plt.ylabel('')
        plt.xlabel(r'$T$ [days]')
        plt.ylim(slope_range)
        plt.xlim(dur_range)
        plt.xticks(xt[2:-1:2])
        plt.yticks(ax3.get_yticks())

        ticklabels = ax1.get_xticklabels() + ax4.get_yticklabels()
        plt.setp(ticklabels,visible=False)
        
        plt.subplots_adjust(hspace=0.001,wspace=0.001)

        if suptitle is None:
            suptitle = self.model
        plt.suptitle(suptitle,fontsize=20)        

        if Ltot is not None:
            lhood = self.lhood(trsig)
            plt.annotate('%s:\nProbability\nof scenario: %.3f' % (trsig.name,
                                                                  self.prior*lhood/Ltot),
                         xy=(0.5,0.5),ha='center',va='center',
                         bbox=dict(boxstyle='round',fc='w'),
                         xycoords='figure fraction',fontsize=15)
           


    @property
    def _properties(self):
        return ['period','model','priorfactors','prob','lhoodcachefile',
                'is_specific', 'cadence'] + \
            super(EclipsePopulation,self)._properties

    def load_hdf(self, filename, path=''): #perhaps this doesn't need to be written?
        StarPopulation.load_hdf(self, filename, path=path)
        try:
            self._make_kde()
        except NoTrapfitError:
            logging.warning('Trapezoid fit not done.')
        return self

class PlanetPopulation(EclipsePopulation):
    def __init__(self, filename=None, period=None, rprs=None,
                 mass=None, radius=None, Teff=None, logg=None,
                 starmodel=None,
                 band='Kepler', model='Planets', n=2e4,
                 fp_specific=0.01, u1=None, u2=None,
                 rbin_width=0.3,
                 MAfn=None, lhoodcachefile=None, **kwargs):
        """Population of Transiting Planets

        Mostly a copy of EBPopulation, with small modifications.

        For simplicity, primary star has only a radius and mass;
        the real properties don't matter at all.

        
                
        If file is passed, population is loaded from .h5 file.

        """

        self.period = period
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile
        self.rprs = rprs
        self.Teff = Teff
        self.logg = logg
        self.starmodel = starmodel
        
        if filename is not None:
            self.load_hdf(filename)
        elif radius is not None and mass is not None:
            # calculates eclipses 
            self.generate(rprs=rprs, mass=mass, radius=radius,
                          n=n, fp_specific=fp_specific, 
                          starmodel=starmodel,
                          rbin_width=rbin_width,
                          u1=u1, u2=u2, Teff=Teff, logg=logg,
                          MAfn=MAfn, **kwargs)

    def generate(self,rprs=None, mass=None, radius=None,
                n=2e4, fp_specific=0.01, u1=None, u2=None,
                 starmodel=None,
                Teff=None, logg=None, rbin_width=0.3,
                MAfn=None, **kwargs):
        """Generates transits
        """

        n = int(n)
        
        if starmodel is None:
            if type(mass) is type((1,)):
                mass = dists.Gaussian_Distribution(*mass)
            if isinstance(mass, dists.Distribution):
                mdist = mass
                mass = mdist.rvs(1e5)

            if type(radius) is type((1,)):
                radius = dists.Gaussian_Distribution(*radius)
            if isinstance(radius, dists.Distribution):
                rdist = radius
                radius = rdist.rvs(1e5)
        else:
            samples = starmodel.random_samples(1e5)
            mass = samples['mass'].values
            radius = samples['radius'].values
            Teff = samples['Teff'].mean()
            logg = samples['logg'].mean()

        if u1 is None or u2 is None:
            if Teff is None or logg is None:
                logging.warning('Teff, logg not provided; using solar limb darkening')
                u1 = 0.394; u2=0.296
            else:
                u1,u2 = ldcoeffs(Teff, logg)
            
        #use point estimate of rprs to construct planets in radius bin
        rp = self.rprs*radius.mean()
        rbin_min = (1-rbin_width)*rp
        rbin_max = (1+rbin_width)*rp
        radius_p = np.random.random(1e5)*(rbin_max - rbin_min) + rbin_min
        mass_p = (radius_p*RSUN/REARTH)**2.06 * MEARTH/MSUN #hokey, but doesn't matter

        stars = pd.DataFrame()
        #df_orbpop = pd.DataFrame() #for orbit population

        tot_prob = None; tot_dprob = None; prob_norm = None
        n_adapt = n
        while len(stars) < n:
            n_adapt = int(n_adapt)
            inds = np.random.randint(len(mass), size=n_adapt)
            
            #calculate eclipses.
            ecl_inds, df, (prob,dprob) = calculate_eclipses(mass[inds], mass_p[inds],
                                                        radius[inds], radius_p[inds],
                                                        15, np.inf, #arbitrary
                                                        u11s=u1, u21s=u2,
                                                        band=self.band, 
                                                        period=self.period, 
                                                        calc_mininc=True,
                                                        return_indices=True,
                                                        MAfn=MAfn)

            df['mass_A'] = mass[inds][ecl_inds]
            df['mass_B'] = mass_p[inds][ecl_inds]
            df['radius_A'] = radius[inds][ecl_inds]
            df['radius_B'] = radius_p[inds][ecl_inds]
            df['u1'] = u1 * np.ones_like(df['mass_A'])
            df['u2'] = u2 * np.ones_like(df['mass_A'])
            df['P'] = self.period * np.ones_like(df['mass_A'])
            
            ok = df['dpri']>0
            
            stars = pd.concat((stars, df[ok]))

            logging.info('{} Transiting planet systems generated (target {})'.format(len(stars),n))
            logging.debug('{} nans in stars[dpri]'.format(np.isnan(stars['dpri']).sum()))

            if tot_prob is None:
                prob_norm = (1/dprob**2)
                tot_prob = prob
                tot_dprob = dprob
            else:
                prob_norm = (1/tot_dprob**2 + 1/dprob**2)
                tot_prob = (tot_prob/tot_dprob**2 + prob/dprob**2)/prob_norm
                tot_dprob = 1/np.sqrt(prob_norm)

            n_adapt = min(int(1.2*(n-len(stars)) * n_adapt//len(df)), 5e4)
            n_adapt = max(n_adapt, 100)

        stars = stars.reset_index()
        stars.drop('index', axis=1, inplace=True)
        stars = stars.iloc[:n]

        stars['mass_1'] = stars['mass_A']
        stars['radius_1'] = stars['radius_A']
        stars['mass_2'] = stars['mass_B']
        stars['radius_2'] = stars['radius_B']

        #make OrbitPopulation?

        #finish below.
                
        priorfactors = {'fp_specific':fp_specific}

        EclipsePopulation.__init__(self, stars=stars,
                                   period=self.period, model=self.model,
                                   lhoodcachefile=self.lhoodcachefile,
                                   priorfactors=priorfactors, prob=tot_prob)
    @property
    def _properties(self):
        return ['rprs', 'Teff', 'logg'] + \
            super(PlanetPopulation, self)._properties

    
class EBPopulation(EclipsePopulation, ColormatchMultipleStarPopulation):
    def __init__(self, filename=None, period=None, mags=None, colors=['JK'],
                 mass=None, age=None, feh=None, starfield=None, colortol=0.1,
                 band='Kepler', model='EBs', f_binary=0.4, n=2e4,
                 MAfn=None, lhoodcachefile=None, starmodel=None,
                 **kwargs):
        """Population of EBs

        Mostly a copy of HEBPopulation, with small modifications.

        If file is passed, population is loaded from .h5 file.

        If file not passed, then a population will be generated.
        If mass, age, and feh are passed, then the primary of
        the population will be generated according to those distributions.
        If distributions are not passed, then populations will be generated
        according to provided starfield.

        mass is primary mass.  mass, age, and feh can be distributions
        (or tuples)

        kwargs passed to ``ColormatchMultipleStarPopulation`` 

        currently doesn't work if mags is None.
        """

        self.period = period
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile

        if filename is not None:
            self.load_hdf(filename)
        elif mags is not None or mass is not None:
            #generates stars from ColormatchMultipleStarPopulation
            # and eclipses using calculate_eclipses
            self.generate(mags=mags, colors=colors, colortol=colortol,
                          starfield=starfield, mass=mass,
                          age=age, feh=feh, n=n, MAfn=MAfn,
                          f_binary=f_binary, starmodel=starmodel,
                          **kwargs)

    def generate(self, mags, colors, starfield=None, colortol=0.1,
                 mass=None, age=None, feh=None, n=2e4,
                 MAfn=None, f_binary=0.4, starmodel=None,
                 **kwargs):
        """Generates stars and eclipses

        stars from ColormatchStellarPopulation; eclipses using calculate_eclipses
        """
        n = int(n)
        #if provided, period_long (self.period) 
        #  is the observed period of the eclipse
        pop_kwargs = {'mags':mags, 'colors':colors,
                      'colortol':colortol,
                      'starfield':starfield,
                      'period_long':self.period}

        if starmodel is not None:
            del pop_kwargs['colors']
            del pop_kwargs['colortol']
            del pop_kwargs['starfield']
            

        #insert additional arguments
        for kw,val in kwargs.iteritems():
            pop_kwargs[kw] = val

        stars = pd.DataFrame()
        df_orbpop = pd.DataFrame() #for orbit population

        tot_prob = None; tot_dprob = None; prob_norm = None
        n_adapt = n
        while len(stars) < n:
            n_adapt = int(n_adapt)

            if starmodel is not None:
                pop = Spectroscopic_MultipleStarPopulation(starmodel=starmodel,
                                                           f_triple=0,
                                                           f_binary=1,
                                                           n=n_adapt,
                                                           **pop_kwargs)
            else:
                pop = ColormatchMultipleStarPopulation(mA=mass, age=age, feh=feh,
                                                       f_triple=0, f_binary=1,
                                                       n=n_adapt, 
                                                       period_short=0,
                                                       **pop_kwargs)

            s = pop.stars.copy()

            #calculate limb-darkening coefficients
            u1A, u2A = ldcoeffs(s['Teff_A'], s['logg_A'])
            u1B, u2B = ldcoeffs(s['Teff_B'], s['logg_B'])


            #calculate eclipses.
            inds, df, (prob,dprob) = calculate_eclipses(s['mass_A'], s['mass_B'],
                                                        s['radius_A'], s['radius_B'],
                                                        s['{}_mag_A'.format(self.band)], 
                                                        s['{}_mag_B'.format(self.band)],
                                                        u11s=u1A, u21s=u2A,
                                                        u12s=u1B, u22s=u2B, 
                                                        band=self.band, 
                                                        period=self.period, 
                                                        calc_mininc=True,
                                                        return_indices=True,
                                                        MAfn=MAfn)

            s = s.iloc[inds].copy()
            s.reset_index(inplace=True)
            for col in df.columns:
                s[col] = df[col]
            stars = pd.concat((stars, s))

            new_df_orbpop = pop.orbpop.orbpop_long.dataframe.iloc[inds].copy()
            new_df_orbpop.reset_index(inplace=True)

            df_orbpop = pd.concat((df_orbpop, new_df_orbpop))

            logging.info('{} Eclipsing EB systems generated (target {})'.format(len(stars),n))
            logging.debug('{} nans in stars[dpri]'.format(np.isnan(stars['dpri']).sum()))
            logging.debug('{} nans in df[dpri]'.format(np.isnan(df['dpri']).sum()))

            if tot_prob is None:
                prob_norm = (1/dprob**2)
                tot_prob = prob
                tot_dprob = dprob
            else:
                prob_norm = (1/tot_dprob**2 + 1/dprob**2)
                tot_prob = (tot_prob/tot_dprob**2 + prob/dprob**2)/prob_norm
                tot_dprob = 1/np.sqrt(prob_norm)

            n_adapt = min(int(1.2*(n-len(stars)) * n_adapt//len(s)), 5e4)
            n_adapt = max(n_adapt, 100)

        stars = stars.iloc[:n]
        df_orbpop = df_orbpop.iloc[:n]
        orbpop = OrbitPopulation_FromDF(df_orbpop)            

        stars = stars.reset_index()
        stars.drop('index', axis=1, inplace=True)

        stars['mass_1'] = stars['mass_A']
        stars['radius_1'] = stars['radius_A']
        stars['mass_2'] = stars['mass_B']
        stars['radius_2'] = stars['radius_B']


        ColormatchMultipleStarPopulation.__init__(self, stars=stars,
                                                  orbpop=orbpop, 
                                                  f_triple=0, f_binary=f_binary,
                                                  period_short=0,
                                                  **pop_kwargs)

        #self.prob = tot_prob
        #self.dprob = tot_dprob #not really ever using this...?

        priorfactors = {'f_binary':f_binary}

        EclipsePopulation.__init__(self, stars=stars, orbpop=orbpop,
                                   period=self.period, model=self.model,
                                   lhoodcachefile=self.lhoodcachefile,
                                   priorfactors=priorfactors, prob=tot_prob)


class HEBPopulation(EclipsePopulation, ColormatchMultipleStarPopulation):
    def __init__(self, filename=None, period=None, mags=None, colors=['JK'], 
                 mass=None, age=None, feh=None, starfield=None, colortol=0.1,
                 band='Kepler', model='HEBs', f_triple=0.12, n=2e4,
                 MAfn=None, lhoodcachefile=None, starmodel=None, **kwargs):
        """Population of HEBs

        If file is passed, population is loaded from .h5 file.

        If file not passed, then a population will be generated.
        If mass, age, and feh are passed, then the primary of
        the population will be generated according to those distributions.
        If distributions are not passed, then populations will be generated
        according to provided starfield.

        mass is primary mass.  mass, age, and feh can be distributions
        (or tuples)

        kwargs passed to ``ColormatchMultipleStarPopulation`` 

        currently doesn't work if mags is None.
        """

        self.period = period
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile

        if filename is not None:
            self.load_hdf(filename)
        elif mags is not None or mass is not None:
            #generates stars from ColormatchMultipleStarPopulation
            # and eclipses using calculate_eclipses
            self.generate(mags=mags, colors=colors, colortol=colortol,
                          starfield=starfield, mass=mass,
                          age=age, feh=feh, n=n, MAfn=MAfn,
                          f_triple=f_triple, starmodel=starmodel,
                          **kwargs)
            

    @property
    def dilution_factor(self):
        magA = self.stars['{}_mag_A'.format(self.band)]
        magB = self.stars['{}_mag_B'.format(self.band)]
        magC = self.stars['{}_mag_C'.format(self.band)]
        return fluxfrac(addmags(magB,magC), magA)


    def generate(self, mags, colors, starfield=None, colortol=0.1,
                 mass=None, age=None, feh=None, n=2e4,
                 MAfn=None, f_triple=0.12, starmodel=None,
                 **kwargs):
        """Generates stars and eclipses

        stars from ColormatchStellarPopulation; eclipses using calculate_eclipses
        """
        n = int(n)
        #if provided, period_short (self.period) 
        #  is the observed period of the eclipse
        pop_kwargs = {'mags':mags, 'colors':colors,
                      'colortol':colortol,
                      'starfield':starfield,
                      'period_short':self.period}

        if starmodel is not None:
            del pop_kwargs['colors']
            del pop_kwargs['colortol']
            del pop_kwargs['starfield']


        #insert additionl arguments
        for kw,val in kwargs.iteritems():
            pop_kwargs[kw] = val

        stars = pd.DataFrame()
        df_long = pd.DataFrame() #orbit populations
        df_short = pd.DataFrame() #orbit populations

        tot_prob = None; tot_dprob = None; prob_norm = None
        n_adapt = n
        while len(stars) < n:
            n_adapt = int(n_adapt)

            if starmodel is not None:
                pop = Spectroscopic_MultipleStarPopulation(starmodel=starmodel,
                                                           f_triple=1,
                                                           n=n_adapt,
                                                           **pop_kwargs)
            else:
                pop = ColormatchMultipleStarPopulation(mA=mass, age=age, feh=feh,
                                                       f_triple=1,
                                                       n=n_adapt, **pop_kwargs)

            s = pop.stars.copy()

            #calculate limb-darkening coefficients
            u1A, u2A = ldcoeffs(s['Teff_A'], s['logg_A'])
            u1B, u2B = ldcoeffs(s['Teff_B'], s['logg_B'])
            u1C, u2C = ldcoeffs(s['Teff_C'], s['logg_C'])


            #calculate eclipses.  In the MultipleStarPopulation, stars '_B' and '_C'
            # are always the ones eclipsing each other.

            inds, df, (prob,dprob) = calculate_eclipses(s['mass_B'], s['mass_C'],
                                                        s['radius_B'], s['radius_C'],
                                                        s['{}_mag_B'.format(self.band)], 
                                                        s['{}_mag_C'.format(self.band)],
                                                        u11s=u1B, u21s=u2B,
                                                        u12s=u1C, u22s=u2C, 
                                                        band=self.band, 
                                                        period=self.period, 
                                                        calc_mininc=True,
                                                        return_indices=True,
                                                        MAfn=MAfn)
            s = s.iloc[inds].copy()
            s.reset_index(inplace=True)
            for col in df.columns:
                s[col] = df[col]
            stars = pd.concat((stars, s))

            new_df_long = pop.orbpop.orbpop_long.dataframe.iloc[inds].copy()
            new_df_long.reset_index(inplace=True)
            new_df_short = pop.orbpop.orbpop_short.dataframe.iloc[inds].copy()
            new_df_short.reset_index(inplace=True)

            df_long = pd.concat((df_long, new_df_long))
            df_short = pd.concat((df_short, new_df_short))

            logging.info('{} eclipsing HEB systems generated (target {})'.format(len(stars),n))
            logging.debug('{} nans in stars[dpri]'.format(np.isnan(stars['dpri']).sum()))
            logging.debug('{} nans in df[dpri]'.format(np.isnan(df['dpri']).sum()))

            if tot_prob is None:
                prob_norm = (1/dprob**2)
                tot_prob = prob
                tot_dprob = dprob
            else:
                prob_norm = (1/tot_dprob**2 + 1/dprob**2)
                tot_prob = (tot_prob/tot_dprob**2 + prob/dprob**2)/prob_norm
                tot_dprob = 1/np.sqrt(prob_norm)

            n_adapt = min(int(1.2*(n-len(stars)) * n_adapt//len(s)), 5e4)
            n_adapt = max(100, n_adapt)

        stars = stars.iloc[:n]
        df_long = df_long.iloc[:n]
        df_short = df_short.iloc[:n]
        orbpop = TripleOrbitPopulation_FromDF(df_long, df_short)            

        stars = stars.reset_index()
        stars.drop('index', axis=1, inplace=True)

        stars['mass_1'] = stars['mass_B']
        stars['radius_1'] = stars['radius_B']
        stars['mass_2'] = stars['mass_C']
        stars['radius_2'] = stars['radius_C']

        ColormatchMultipleStarPopulation.__init__(self, stars=stars,
                                                  orbpop=orbpop, 
                                                  f_triple=f_triple,
                                                  **pop_kwargs)

        #self.prob = tot_prob
        #self.dprob = tot_dprob #not really ever using this...?

        priorfactors = {'f_triple':f_triple}

        EclipsePopulation.__init__(self, stars=stars, orbpop=orbpop,
                                   period=self.period, model=self.model,
                                   lhoodcachefile=self.lhoodcachefile,
                                   priorfactors=priorfactors, prob=tot_prob)


    #@property
    #def _properties(self):
    #    #still unclear how this gets _properties
    #    # from both EclipsePopulation and ColormatchMultipleStarPopulation,
    #    # but it seems to...

    #    return super(HEBPopulation,self)._properties
            


class BEBPopulation(EclipsePopulation, MultipleStarPopulation,
                    BGStarPopulation):
    def __init__(self, filename=None, period=None, mags=None,
                 ra=None, dec=None, trilegal_filename=None,
                 n=2e4, ichrone=DARTMOUTH, band='Kepler',
                 MAfn=None, lhoodcachefile=None,
                 maxrad=10, f_binary=0.4, model='BEBs', **kwargs):
        """

        Filename is for loading population from HDF

        trilegal_filename holds BG star population

        maxrad in arcsec

        isochrone is Dartmouth, by default (in starutils)
        """

        self.period = period
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile
        self.mags = mags
        
        if filename is not None:
            self.load_hdf(filename)

        elif trilegal_filename is not None or (ra is not None
                                               and dec is not None):
            self.generate(trilegal_filename,
                          ra=ra, dec=dec, mags=mags,
                          n=n, ichrone=ichrone, MAfn=MAfn,
                          maxrad=maxrad, f_binary=f_binary, **kwargs)

    @property
    def prior(self):
        return (super(BEBPopulation, self).prior * 
                self.density.to('arcsec^-2').value * #sky density
                np.pi*(self.maxrad.to('arcsec').value)**2) # sky area
        

    @property
    def dilution_factor(self):
        if self.mags is None:
            return super(BEBPopulation, self).dilution_factor
        else:
            b = self.band
            return fluxfrac(self.stars['{}_mag'.format(b)], self.mags[b])

    def generate(self, trilegal_filename, ra=None, dec=None,
                 n=2e4, ichrone=DARTMOUTH, MAfn=None,
                 mags=None, maxrad=None, f_binary=0.4, **kwargs):

        n = int(n)
        
        #generate/load BG primary stars from TRILEGAL simulation
        bgpop = BGStarPopulation_TRILEGAL(trilegal_filename,
                                        ra=ra, dec=dec, mags=mags,
                                        maxrad=maxrad, **kwargs)

        # Make sure that
        # properties of stars are within allowable range for isochrone.
        # This is a bit hacky, admitted.
        mass = bgpop.stars['m_ini'].values
        age = bgpop.stars['logAge'].values
        feh = bgpop.stars['[M/H]'].values

        pct = 0.05 #pct distance from "edges" of ichrone interpolation
        mass[mass < ichrone.minmass*(1+pct)] = ichrone.minmass*(1+pct)
        mass[mass > ichrone.maxmass*(1-pct)] = ichrone.maxmass*(1-pct)
        age[age < ichrone.minage*(1+pct)] = ichrone.minage*(1+pct)
        age[age > ichrone.maxage*(1-pct)] = ichrone.maxage*(1-pct)
        feh[feh < ichrone.minfeh+0.05] = ichrone.minfeh+0.05
        feh[feh > ichrone.maxfeh-0.05] = ichrone.maxfeh-0.05

        distance = bgpop.stars['distance'].values

        #Generate binary population to draw eclipses from
        pop = MultipleStarPopulation(mA=mass, age=age, feh=feh,
                                            f_triple=0, f_binary=1,
                                            distance=distance,
                                            ichrone=ichrone)
        
        all_stars = pop.stars.dropna(subset=['mass_A'])
        all_stars.reset_index(inplace=True)
        
        #generate eclipses
        stars = pd.DataFrame()
        df_orbpop = pd.DataFrame()
        tot_prob = None; tot_dprob=None; prob_norm=None
                
        n_adapt = n
        while len(stars) < n:
            n_adapt = int(n_adapt)
            inds = np.random.randint(len(all_stars), size=n_adapt)

            s = all_stars.iloc[inds]  
            
            #calculate limb-darkening coefficients
            u1A, u2A = ldcoeffs(s['Teff_A'], s['logg_A'])
            u1B, u2B = ldcoeffs(s['Teff_B'], s['logg_B'])

            inds, df, (prob,dprob) = calculate_eclipses(s['mass_A'], s['mass_B'],
                                                        s['radius_A'], s['radius_B'],
                                                        s['{}_mag_A'.format(self.band)], 
                                                        s['{}_mag_B'.format(self.band)],
                                                        u11s=u1A, u21s=u2A,
                                                        u12s=u1B, u22s=u2B, 
                                                        band=self.band, 
                                                        period=self.period, 
                                                        calc_mininc=True,
                                                        return_indices=True,
                                                        MAfn=MAfn)
            s = s.iloc[inds].copy()
            s.reset_index(inplace=True)
            for col in df.columns:
                s[col] = df[col]
            stars = pd.concat((stars, s))

            #new_df_orbpop = pop.orbpop.orbpop_long.dataframe.iloc[inds].copy()
            #new_df_orbpop.reset_index(inplace=True)

            #df_orbpop = pd.concat((df_orbpop, new_df_orbpop))

            logging.info('{} BEB systems generated (target {})'.format(len(stars),n))
            #logging.debug('{} nans in stars[dpri]'.format(np.isnan(stars['dpri']).sum()))
            #logging.debug('{} nans in df[dpri]'.format(np.isnan(df['dpri']).sum()))

            if tot_prob is None:
                prob_norm = (1/dprob**2)
                tot_prob = prob
                tot_dprob = dprob
            else:
                prob_norm = (1/tot_dprob**2 + 1/dprob**2)
                tot_prob = (tot_prob/tot_dprob**2 + prob/dprob**2)/prob_norm
                tot_dprob = 1/np.sqrt(prob_norm)

            n_adapt = min(int(1.2*(n-len(stars)) * n_adapt//len(s)), 5e5)
            #logging.debug('n_adapt = {}'.format(n_adapt))
            n_adapt = max(n_adapt, 100)
            n_adapt = int(n_adapt)
            
        stars = stars.iloc[:n]
        #df_orbpop = df_orbpop.iloc[:n]
        #orbpop = OrbitPopulation_FromDF(df_orbpop)            

        if 'level_0' in stars:
            stars.drop('level_0', axis=1, inplace=True) #dunno where this came from
        stars = stars.reset_index()
        stars.drop('index', axis=1, inplace=True)

        stars['mass_1'] = stars['mass_A']
        stars['radius_1'] = stars['radius_A']
        stars['mass_2'] = stars['mass_B']
        stars['radius_2'] = stars['radius_B']
            
        MultipleStarPopulation.__init__(self, stars=stars,
                                        #orbpop=orbpop,
                                        f_triple=0, f_binary=f_binary,
                                        period_long=self.period)

        priorfactors = {'f_binary':f_binary}

        #attributes needed for BGStarPopulation
        self.density = bgpop.density
        self.trilegal_args = bgpop.trilegal_args
        self._maxrad = bgpop._maxrad

        #create an OrbitPopulation here?
        
        EclipsePopulation.__init__(self, stars=stars, #orbpop=orbpop,
                                   period=self.period, model=self.model,
                                   lhoodcachefile=self.lhoodcachefile,
                                   priorfactors=priorfactors, prob=tot_prob)

        #add Rsky property
        self.stars['Rsky'] = randpos_in_circle(len(self.stars), 
                                               self._maxrad, return_rad=True)

    @property
    def _properties(self):
        return ['density','trilegal_args','mags'] + \
          super(BEBPopulation, self)._properties

            
class PopulationSet(object):
    def __init__(self, poplist=None, ra=None, dec=None,
                 period=None, mags=None, n=2e4, 
                 mass=None, age=None, feh=None, 
                 radius=None, rprs=None,
                 MAfn=None, colors=None,
                 trilegal_filename=None,
                 Teff=None, logg=None, savefile=None,
                 starmodel=None,
                 heb_kws=None, eb_kws=None, 
                 beb_kws=None, pl_kws=None,
                 hide_exceptions=False):
        """

        Poplist can be a list of EclipsePopulations, a string (filename),
        or nothing, in which case the populations will be generated

        mass, radius, age, and feh can be two-element tuples or Distributions

        rprs, Teff, logg should be single values.

        """

        #if string is passed, load from file
        if poplist is None:
            self.generate(ra, dec, period, mags,
                          n=n, mass=mass, age=age, feh=feh,
                          MAfn=MAfn, colors=colors, radius=radius,
                          trilegal_filename=trilegal_filename,
                          Teff=Teff, logg=logg, rprs=rprs,
                          savefile=savefile, starmodel=starmodel,
                          heb_kws=heb_kws, eb_kws=eb_kws, 
                          beb_kws=beb_kws, pl_kws=pl_kws,
                          hide_exceptions=hide_exceptions)
            
        elif type(poplist)==type(''):
            self.load_hdf(poplist)
        else:
            self.poplist = poplist

    def generate(self, ra, dec, period, mags,
                 n=2e4, mass=None, age=None, feh=None, radius=None,
                 MAfn=None, colors=None, Teff=None, logg=None,
                 rprs=None, trilegal_filename=None, starmodel=None,
                 heb_kws=None, eb_kws=None, 
                 beb_kws=None, pl_kws=None, savefile=None,
                 hide_exceptions=False):
        """
        """
        if colors is None:
            colors = ['JK','HK']
        if MAfn is None:
            MAfn = MAInterpolationFunction(pmin=0.007, pmax=1/0.007, nzs=200, nps=400)

        if beb_kws is None:
            beb_kws = {}
        if heb_kws is None:
            heb_kws = {}
        if eb_kws is None:
            eb_kws = {}
        if pl_kws is None:
            pl_kws = {}

        try:
            bebpop = BEBPopulation(trilegal_filename=trilegal_filename,
                                   ra=ra, dec=dec, period=period, 
                                   mags=mags, MAfn=MAfn, n=n, **beb_kws)
            bebpop.fit_trapezoids(MAfn=MAfn)
            if savefile is not None:
                bebpop.save_hdf(savefile, 'beb')
        except:
            logging.error('Error generating BEB population.')
            if not hide_exceptions:
                raise

        try:
            hebpop = HEBPopulation(mass=mass, age=age, feh=feh, 
                                   colors=colors, period=period,
                                   starmodel=starmodel,
                                   mags=mags, MAfn=MAfn, n=n, **heb_kws)
            hebpop.fit_trapezoids(MAfn=MAfn)
            if savefile is not None:
                hebpop.save_hdf(savefile, 'heb')
        except:
            logging.error('Error generating HEB population.')
            if not hide_exceptions:
                raise

        try:
            ebpop = EBPopulation(mass=mass, age=age, feh=feh, 
                                 colors=colors, period=period,
                                 starmodel=starmodel,
                                 mags=mags, MAfn=MAfn, n=n, **eb_kws)
            ebpop.fit_trapezoids(MAfn=MAfn)
            if savefile is not None:
                ebpop.save_hdf(savefile, 'eb')
        except:
            logging.error('Error generating EB population.')
            if not hide_exceptions:
                raise

        try:
            plpop = PlanetPopulation(mass=mass, radius=radius,
                                     period=period, rprs=rprs,
                                     Teff=Teff, logg=logg,
                                     starmodel=starmodel,
                                     MAfn=MAfn, n=n, **pl_kws)
            plpop.fit_trapezoids(MAfn=MAfn)
            if savefile is not None:
                plpop.save_hdf(savefile, 'pl')
        except:
            logging.error('Error generating Planet population.')
            if not hide_exceptions:
                raise

        self.poplist = [hebpop, ebpop, bebpop, plpop]

    @property
    def constraints(self):
        cs = []
        for pop in self.poplist:
            cs += [c for c in pop.constraints]
        return list(set(cs))

    @property
    def modelnames(self):
        return [pop.model for pop in self.poplist]

    @property
    def shortmodelnames(self):
        return [pop.modelshort for pop in self.poplist]

    def save_hdf(self, filename, path='', overwrite=False):
        if os.path.exists(filename) and overwrite:
            os.remove(filename)

        for pop in self.poplist:
            name = pop.modelshort
            pop.save_hdf(filename, path='{}/{}'.format(path,name), append=True)

    def load_hdf(self, filename, path=''):
        store = pd.HDFStore(filename)
        models = []
        types = []
        for k in store.keys():
            m = re.search('/(\S+)/stars', k)
            if m:
                models.append(m.group(1))
                types.append(store.get_storer(m.group(0)).attrs.poptype)
        poplist = []
        for m,t in zip(models,types):
            poplist.append(t().load_hdf(filename, path='{}/{}'.format(path,m)))
        store.close()

        PopulationSet.__init__(self, poplist) #how to deal with saved constraints?
        return self

    def add_population(self,pop):
        if pop.model in self.modelnames:
            raise ValueError('%s model already in PopulationSet.' % pop.model)
        self.modelnames.append(pop.model)
        self.shortmodelnames.append(pop.modelshort)
        self.poplist.append(pop)
        self.apply_dmaglim()

    def remove_population(self,pop):
        iremove=None
        for i in range(len(self.poplist)):
            if self.modelnames[i]==self.poplist[i].model:
                iremove=i
        if iremove is not None:
            self.modelnames.pop(i)
            self.shortmodelnames.pop(i)
            self.poplist.pop(i)

    def __hash__(self):
        key = 0
        for pop in self.poplist:
            key = hashcombine(key,pop)
        return key

    def __getitem__(self,name):
        name = name.lower()
        if name in ['pl','pls']:
            name = 'planets'
        elif name in ['eb','ebs']:
            name = 'ebs'
        elif name in ['heb','hebs']:
            name = 'hebs'
        elif name in ['beb','bebs','bgeb','bgebs']:
            name = 'bebs'
        elif name in ['bpl','bgpl','bpls','bgpls']:
            name = 'blended planets'
        elif name in ['sbeb','sbgeb','sbebs','sbgebs']:
            name = 'specific beb'
        elif name in ['sheb','shebs']:
            name = 'specific heb'
        for pop in self.poplist:
            if name==pop.model.lower():
                return pop
        raise ValueError('%s not in modelnames: %s' % (name,self.modelnames))

    @property
    def colordict(self):
        d = {}
        i=0
        n = len(self.constraints)
        for c in self.constraints:
            #self.colordict[c] = colors[i % 6]
            d[c] = cm.jet(1.*i/n)
            i+=1
        return d

    @property
    def priorfactors(self):
        priorfactors = {}
        for pop in self.poplist:
            for f in pop.priorfactors:
                if f in priorfactors:
                    if pop.priorfactors[f] != priorfactors[f]:
                        raise ValueError('prior factor %s is inconsistent!' % f)
                else:
                    priorfactors[f] = pop.priorfactors[f]
        return priorfactors
        

    def change_prior(self,**kwargs):
        for kw,val in kwargs.iteritems():
            if kw=='area':
                logging.warning('cannot change area in this way--use change_maxrad instead')
                continue
            for pop in self.poplist:
                k = {kw:val}
                pop.change_prior(**k)

    def apply_multicolor_transit(self,band,depth):
        if '{} band transit'.format(band) not in self.constraints:
            self.constraints.append('{} band transit'.format(band))
        for pop in self.poplist:
            pop.apply_multicolor_transit(band,depth)

    def set_maxrad(self,newrad):
        if not isinstance(newrad, Quantity):
            newrad = newrad * u.arcsec
        #if 'Rsky' not in self.constraints:
        #    self.constraints.append('Rsky')
        for pop in self.poplist:
            if not pop.is_specific:
                try: 
                    pop.maxrad = newrad
                except AttributeError:
                    pass

    def apply_dmaglim(self,dmaglim=None):
        if 'bright blend limit' not in self.constraints:
            self.constraints.append('bright blend limit')
        for pop in self.poplist:
            if not hasattr(pop,'dmaglim') or pop.is_specific:
                continue
            if dmaglim is None:
                dmag = pop.dmaglim
            else:
                dmag = dmaglim
            pop.set_dmaglim(dmag)
        self.dmaglim = dmaglim

    def apply_trend_constraint(self,limit,dt):
        if 'RV monitoring' not in self.constraints:
            self.constraints.append('RV monitoring')
        for pop in self.poplist:
            if not hasattr(pop,'dRV'):
                continue
            pop.apply_trend_constraint(limit,dt)
        self.trend_limit = limit
        self.trend_dt = dt

    def apply_secthresh(self,secthresh):
        if 'secondary depth' not in self.constraints:
            self.constraints.append('secondary depth')
        for pop in self.poplist:
            pop.apply_secthresh(secthresh)
        self.secthresh = secthresh

    def constrain_property(self,prop,**kwargs):
        if prop not in self.constraints:
            self.constraints.append(prop)
        for pop in self.poplist:
            try:
                pop.constrain_property(prop,**kwargs)
            except AttributeError:
                logging.info('%s model does not have property stars.%s (constraint not applied)' % (pop.model,prop))

    def replace_constraint(self,name,**kwargs):
        for pop in self.poplist:
            pop.replace_constraint(name,**kwargs)
        if name not in self.constraints:
            self.constraints.append(name)

    def remove_constraint(self,*names):
        for name in names:
            for pop in self.poplist:
                if name in pop.constraints:
                    pop.remove_constraint(name)
                else:
                    logging.info('%s model does not have %s constraint' % (pop.model,name))
            if name in self.constraints:
                self.constraints.remove(name)
                    
    def apply_cc(self, cc, **kwargs):
        if type(cc)==type(''):
            pass
        if cc.name not in self.constraints:
            self.constraints.append(cc.name)
        for pop in self.poplist:
            if not pop.is_specific:
                try:
                    pop.apply_cc(cc, **kwargs)
                except AttributeError:
                    logging.info('%s cc not applied to %s model' % (cc.name,pop.model))

    def apply_vcc(self,vcc):
        if 'secondary spectrum' not in self.constraints:
            self.constraints.append('secondary spectrum')
        for pop in self.poplist:
            if not pop.is_specific:
                try:
                    pop.apply_vcc(vcc)
                except:
                    logging.info('VCC constraint not applied to %s model' % (pop.model))




############ Utility Functions ##############
    
def calculate_eclipses(M1s, M2s, R1s, R2s, mag1s, mag2s,
                       u11s=0.394, u21s=0.296, u12s=0.394, u22s=0.296,
                       Ps=None, period=None, logperkde=RAGHAVAN_LOGPERKDE,
                       incs=None, eccs=None, band='i',
                       mininc=None, maxecc=0.97, verbose=False,
                       return_probability_only=False, return_indices=False,
                       calc_mininc=True, MAfn=None):
    """Returns random eclipse parameters for provided inputs

    If single period desired, pass 'period' keyword.

    M1s, M2s, R1s, R2s must be array_like
    """
    if MAfn is None:
        logging.warning('MAInterpolationFunction not passed, so generating one...')
        MAfn = MAInterpolationFunction(nzs=200,nps=400,pmin=0.007,pmax=1/0.007)

    M1s = np.atleast_1d(M1s)
    M2s = np.atleast_1d(M2s)
    R1s = np.atleast_1d(R1s)
    R2s = np.atleast_1d(R2s)

    nbad = (np.isnan(M1s) | np.isnan(M2s) | np.isnan(R1s) | np.isnan(R2s)).sum()
    if nbad > 0:
        logging.warning('{} M1s are nan'.format(np.isnan(M1s).sum()))
        logging.warning('{} M2s are nan'.format(np.isnan(M2s).sum()))
        logging.warning('{} R1s are nan'.format(np.isnan(R1s).sum()))
        logging.warning('{} R2s are nan'.format(np.isnan(R2s).sum()))

    mag1s = mag1s * np.ones_like(M1s)
    mag2s = mag2s * np.ones_like(M1s)
    u11s = u11s * np.ones_like(M1s)
    u21s = u21s * np.ones_like(M1s)
    u12s = u12s * np.ones_like(M1s)
    u22s = u22s * np.ones_like(M1s)

    n = np.size(M1s)

    #a bit clunky here, but works.
    simPs = False
    if period:
        Ps = np.ones(n)*period
    else:
        if Ps is None:
            Ps = 10**(logperkde.rvs(n))
            simPs = True
    simeccs = False
    if eccs is None:
        if not simPs and period is not None:
            eccs = draw_eccs(n,period,maxecc=maxecc)
        else:
            eccs = draw_eccs(n,Ps,maxecc=maxecc)
        simeccs = True

    bad_Ps = np.isnan(Ps)
    if bad_Ps.sum()>0:
        logging.warning('{} nan periods.  why?'.format(bad_Ps.sum()))
    bad_eccs = np.isnan(eccs)
    if bad_eccs.sum()>0:
        logging.warning('{} nan eccentricities.  why?'.format(bad_eccs.sum()))

    semimajors = semimajor(Ps, M1s+M2s)*AU #in AU

    #check to see if there are simulated instances that are
    # too close; i.e. periastron sends secondary within roche 
    # lobe of primary
    tooclose = withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
    ntooclose = tooclose.sum()
    tries = 0
    maxtries=5
    if simPs:
        while ntooclose > 0:
            lastntooclose=ntooclose
            Ps[tooclose] = 10**(logperkde.rvs(ntooclose))
            if simeccs:
                eccs[tooclose] = draw_eccs(ntooclose,Ps[tooclose])
            semimajors[tooclose] = semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
            tooclose = withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
            ntooclose = tooclose.sum()
            if ntooclose==lastntooclose:   #prevent infinite loop
                tries += 1
                if tries > maxtries:
                    logging.info('{} binaries are "too close"; gave up trying to fix.'.format(ntooclose))
                    break                       
    else:
        while ntooclose > 0:
            lastntooclose=ntooclose
            if simeccs:
                eccs[tooclose] = draw_eccs(ntooclose,Ps[tooclose])
            semimajors[tooclose] = semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
            #wtooclose = where(semimajors*(1-eccs) < 2*(R1s+R2s)*RSUN)
            tooclose = withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
            ntooclose = tooclose.sum()
            if ntooclose==lastntooclose:   #prevent infinite loop
                tries += 1
                if tries > maxtries:
                    logging.info('{} binaries are "too close"; gave up trying to fix.'.format(ntooclose))
                    break                       

    #randomize inclinations, either full range, or within restricted range
    if mininc is None and calc_mininc:
        mininc = minimum_inclination(Ps, M1s, M2s, R1s, R2s)

    if incs is None:
        if mininc is None:
            incs = np.arccos(np.random.random(n)) #random inclinations in radians
        else:
            incs = np.arccos(np.random.random(n)*np.cos(mininc*np.pi/180))
    if mininc:
        prob = np.cos(mininc*np.pi/180)
    else:
        prob = 1

    logging.debug('initial probability given mininc starting at {}'.format(prob))

    ws = np.random.random(n)*2*np.pi

    switched = (R2s > R1s)
    R_large = switched*R2s + ~switched*R1s
    R_small = switched*R1s + ~switched*R2s


    b_tras = semimajors*np.cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 + eccs*np.sin(ws))
    b_occs = semimajors*np.cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 - eccs*np.sin(ws))

    b_tras[tooclose] = np.inf
    b_occs[tooclose] = np.inf

    ks = R_small/R_large
    Rtots = (R_small + R_large)/R_large
    tra = (b_tras < Rtots)
    occ = (b_occs < Rtots)
    nany = (tra | occ).sum()
    peb = nany/float(n)
    prob *= peb
    if return_probability_only:
        return prob,prob*np.sqrt(nany)/n


    i = (tra | occ)
    wany = np.where(i)
    P,M1,M2,R1,R2,mag1,mag2,inc,ecc,w = Ps[i],M1s[i],M2s[i],R1s[i],R2s[i],\
        mag1s[i],mag2s[i],incs[i]*180/np.pi,eccs[i],ws[i]*180/np.pi
    a = semimajors[i]  #in cm already
    b_tra = b_tras[i]
    b_occ = b_occs[i]
    u11 = u11s[i]
    u21 = u21s[i]
    u12 = u12s[i]
    u22 = u22s[i]
   
    
    switched = (R2 > R1)
    R_large = switched*R2 + ~switched*R1
    R_small = switched*R1 + ~switched*R2
    k = R_small/R_large
    
    #calculate durations
    T14_tra = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1+k)**2 - b_tra**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180)) #*24*60
    T23_tra = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1-k)**2 - b_tra**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180)) #*24*60
    T14_occ = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1+k)**2 - b_occ**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1-ecc*np.sin(w*np.pi/180)) #*24*60
    T23_occ = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1-k)**2 - b_occ**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1-ecc*np.sin(w*np.pi/180)) #*24*60
    
    bad = (np.isnan(T14_tra) & np.isnan(T14_occ))
    if bad.sum() > 0:
        logging.error('Something snuck through with no eclipses!')
        logging.error('k: {}'.format(k[bad]))
        logging.error('b_tra: {}'.format(b_tra[bad]))
        logging.error('b_occ: {}'.format(b_occ[bad]))
        logging.error('T14_tra: {}'.format(T14_tra[bad]))
        logging.error('T14_occ: {}'.format(T14_occ[bad]))
        logging.error('under sqrt (tra): {}'.format((1+k[bad])**2 - b_tra[bad]**2))
        logging.error('under sqrt (occ): {}'.format((1+k[bad])**2 - b_occ[bad]**2))
        logging.error('eccsq: {}'.format(ecc[bad]**2))
        logging.error('a in Rsun: {}'.format(a[bad]/RSUN))
        logging.error('R_large: {}'.format(R_large[bad]))
        logging.error('R_small: {}'.format(R_small[bad]))
        logging.error('P: {}'.format(P[bad]))
        logging.error('total M: {}'.format(M1[bad]+M2[bad]))

    T14_tra[(np.isnan(T14_tra))] = 0
    T23_tra[(np.isnan(T23_tra))] = 0
    T14_occ[(np.isnan(T14_occ))] = 0
    T23_occ[(np.isnan(T23_occ))] = 0

    #calling mandel-agol
    ftra = MAfn(k,b_tra,u11,u21)
    focc = MAfn(1/k,b_occ/k,u12,u22)

    #fix those with k or 1/k out of range of MAFN....or do it in MAfn eventually?
    wtrabad = np.where((k < MAfn.pmin) | (k > MAfn.pmax))
    woccbad = np.where((1/k < MAfn.pmin) | (1/k > MAfn.pmax))
    for ind in wtrabad[0]:
        ftra[ind] = occultquad(b_tra[ind],u11[ind],u21[ind],k[ind])
    for ind in woccbad[0]:
        focc[ind] = occultquad(b_occ[ind]/k[ind],u12[ind],u22[ind],1/k[ind])

    F1 = 10**(-0.4*mag1) + switched*10**(-0.4*mag2)
    F2 = 10**(-0.4*mag2) + switched*10**(-0.4*mag1)

    dtra = 1-(F2 + F1*ftra)/(F1+F2)
    docc = 1-(F1 + F2*focc)/(F1+F2)

    totmag = -2.5*np.log10(F1+F2)

    #wswitched = where(switched)
    dtra[switched],docc[switched] = (docc[switched],dtra[switched])
    T14_tra[switched],T14_occ[switched] = (T14_occ[switched],T14_tra[switched])
    T23_tra[switched],T23_occ[switched] = (T23_occ[switched],T23_tra[switched])
    b_tra[switched],b_occ[switched] = (b_occ[switched],b_tra[switched])
    #mag1[wswitched],mag2[wswitched] = (mag2[wswitched],mag1[wswitched])
    F1[switched],F2[switched] = (F2[switched],F1[switched])
    u11[switched],u12[switched] = (u12[switched],u11[switched])
    u21[switched],u22[switched] = (u22[switched],u21[switched])

    dtra[(np.isnan(dtra))] = 0
    docc[(np.isnan(docc))] = 0

    if np.any(np.isnan(ecc)):
        logging.warning('{} nans in eccentricity.  why?'.format(np.isnan(ecc).sum()))

    df =  pd.DataFrame({'{}_mag_tot'.format(band) : totmag,
                        'P':P, 'ecc':ecc, 'inc':inc, 'w':w,
                        'dpri':dtra, 'dsec':docc,
                        'T14_pri':T14_tra, 'T23_pri':T23_tra,
                        'T14_sec':T14_occ, 'T23_sec':T23_occ,
                        'b_pri':b_tra, 'b_sec':b_occ,
                        '{}_mag_1'.format(band) : mag1,
                        '{}_mag_2'.format(band) : mag2,
                        'fluxfrac_1':F1/(F1+F2),
                        'fluxfrac_2':F2/(F1+F2),
                        'switched':switched,
                        'u1_1':u11, 'u2_1':u21, 'u1_2':u12, 'u2_2':u22})

    df.reset_index(inplace=True)

    logging.debug('final prob: {}'.format(prob))

    if return_indices:
        return wany, df, (prob, prob*np.sqrt(nany)/n)
    else:
        return df, (prob, prob*np.sqrt(nany)/n)



#########################
###### Utility functions

def loadcache(cachefile):
    """  
    """
    cache = {}
    if os.path.exists(cachefile):
        for line in open(cachefile):
            line = line.split()
            if len(line)==2:
                try:
                    cache[int(line[0])] = float(line[1])
                except:
                    pass
    return cache
    

####### Exceptions

class EmptyPopulationError(Exception):
    pass

class NoTrapfitError(Exception):
    pass
