from __future__ import print_function, division

import logging
import os, os.path
import re
import math
import copy

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from scipy.stats import gaussian_kde
    from scipy.integrate import quad
else:
    np, pd, plt, cm = (None, None, None, None)
    gaussian_kde, quad = (None, None)

try:
    from sklearn.neighbors import KernelDensity
    from sklearn.grid_search import GridSearchCV
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import LeaveOneOut
except ImportError:
    logging.warning('sklearn not available')
    KernelDensity = None
    GridSearchCV = None

if not on_rtd:
    from isochrones import StarModel, get_ichrone
else:
    class StarModel(object):
        pass
#from transit import Central, System, Body

from .transit_basic import occultquad, ldcoeffs, minimum_inclination
from .transit_basic import MAInterpolationFunction
from .transit_basic import eclipse_pars
from .transit_basic import eclipse, eclipse_tt, NoEclipseError, NoFitError
from .transit_basic import MAXSLOPE
from .fitebs import fitebs

from .plotutils import setfig, plot2dhist
from .hashutils import hashcombine

from .stars.populations import StarPopulation, MultipleStarPopulation
from .stars.populations import BGStarPopulation, BGStarPopulation_TRILEGAL
from .stars.populations import Observed_BinaryPopulation, Observed_TriplePopulation
# from .stars.populations import DARTMOUTH
from .stars.utils import draw_eccs, semimajor, withinroche
from .stars.utils import mult_masses, randpos_in_circle
from .stars.utils import fluxfrac, addmags
from .stars.utils import RAGHAVAN_LOGPERKDE

from .stars.constraints import UpperLimit

try:
    import simpledist.distributions as dists
except ImportError:
    logging.warning('simpledist not available')
    dists = None

try:
    from progressbar import Percentage,Bar,RotatingMarker,ETA,ProgressBar
    pbar_ok = True
except ImportError:
    pbar_ok = False


from .orbits.populations import OrbitPopulation, TripleOrbitPopulation

SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'EBs (Double Period)':'eb_Px2',
                    'HEBs (Double Period)':'heb_Px2',
                    'BEBs (Double Period)':'beb_Px2',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}

INV_SHORT_MODELNAMES = {v:k for k,v in SHORT_MODELNAMES.items()}

DEFAULT_MODELS = ['beb','heb','eb',
                  'beb_Px2', 'heb_Px2','eb_Px2',
                  'pl']


if not on_rtd:
    from astropy.units import Quantity
    import astropy.units as u
    import astropy.constants as const
    AU = const.au.cgs.value
    RSUN = const.R_sun.cgs.value
    MSUN = const.M_sun.cgs.value
    G = const.G.cgs.value
    REARTH = const.R_earth.cgs.value
    MEARTH = const.M_earth.cgs.value
else:
    Quantity = None
    u = None
    const = None
    AU, RSUN, MSUN, G, REARTH, MEARTH = (None, None, None, None, None, None)


class EclipsePopulation(StarPopulation):
    """Base class for populations of eclipsing things.

    This is the base class for populations of various scenarios
    that could explain a tranist signal; that is,
    astrophysical false positives or transiting planets.

    Once set up properly, :func:`EclipsePopulation.fit_trapezoids`
    can be used to fit the trapezoidal shape parameters, after
    which the likelihood of a transit signal under the model
    may be calculated.

    Subclasses :class:`vespa.stars.StarPopulation`, which enables
    all the functionality of observational constraints.

    if prob is not passed; should be able to calculated from given
    star/orbit properties.

    As with :class:`vespa.stars.StarPopulation`, any subclass must be able
    to be initialized with no arguments passed, in order for
    :func:`vespa.stars.StarPopulation.load_hdf` to work properly.

    :param stars:
        ``DataFrame`` with star properties.  Must contain
        ``M_1, M_2, R_1, R_2, u1_1, u1_2, u2_1, u2_2``.
        Also, either the ``period`` keyword argument must be provided
        or a ``period`` column should be in ``stars``.
        ``stars`` must also have the eclipse parameters:
        `'inc, ecc, w, dpri, dsec, b_sec, b_pri, fluxfrac_1, fluxfrac_2``.

    :param period: (optional)
        Orbital period.  If not provided, then ``stars`` must
        have period column.

    :param model: (optional)
        Name of the model.

    :param priorfactors: (optional)
        Multiplicative factors that quantify the model prior
        for this particular model; e.g. ``f_binary``, etc.

    :param lhoodcachefile: (optional)
        File where likelihood calculation cache is written.

    :param orbpop: (optional)
        Orbit population.
    :type orbpop:
        :class:`orbits.OrbitPopulation` or
        :class:`orbits.TripleOrbitPopulation`

    :param prob: (optional)
        Averaged eclipse probability of scenario instances.
        If not provided, this should be calculated,
        though this is not implemented yet.

    :param cadence: (optional)
        Observing cadence, in days.  Defaults to *Kepler* value.

    :param **kwargs:
        Additional keyword arguments passed to
        :class:`vespa.stars.StarPopulation`.

    """

    def __init__(self, stars=None, period=None, model='',
                 priorfactors=None, lhoodcachefile=None,
                 orbpop=None, prob=None,
                 cadence=1626./86400, #Kepler observing cadence, in days
                 **kwargs):


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

    def fit_trapezoids(self, MAfn=None, msg=None, use_pbar=True, **kwargs):
        """
        Fit trapezoid shape to each eclipse in population

        For each instance in the population, first the correct,
        physical Mandel-Agol transit shape is simulated,
        and then this curve is fit with a trapezoid model

        :param MAfn:
            :class:`transit_basic.MAInterpolationFunction` object.
            If not passed, then one with default parameters will
            be created.

        :param msg:
            Message to be displayed for progressbar output.

        :param **kwargs:
            Additional keyword arguments passed to :func:`fitebs.fitebs`.

        """
        logging.info('Fitting trapezoid models for {}...'.format(self.model))

        if msg is None:
            msg = '{}: '.format(self.model)

        n = len(self.stars)
        deps, durs, slopes = (np.zeros(n), np.zeros(n), np.zeros(n))
        secs = np.zeros(n, dtype=bool)
        dsec = np.zeros(n)

        if use_pbar and pbar_ok:
            widgets = [msg+'fitting shape parameters for %i systems: ' % n,Percentage(),
                       ' ',Bar(marker=RotatingMarker()),' ',ETA()]
            pbar = ProgressBar(widgets=widgets,maxval=n)
            pbar.start()

        for i in range(n):
            logging.debug('Fitting star {}'.format(i))
            pri = (self.stars['dpri'][i] > self.stars['dsec'][i] or
                   np.isnan(self.stars['dsec'][i]))
            sec = not pri
            secs[i] = sec
            if sec:
                dsec[i] = self.stars['dpri'][i]
            else:
                dsec[i] = self.stars['dsec'][i]

            try:
                trap_pars = self.eclipse_trapfit(i, secondary=sec, **kwargs)

            except NoEclipseError:
                logging.error('No eclipse registered for star {}'.format(i))
                trap_pars = (np.nan, np.nan, np.nan)
            except NoFitError:
                logging.error('Fit did not converge for star {}'.format(i))
                trap_pars = (np.nan, np.nan, np.nan)
            except KeyboardInterrupt:
                raise
            except:
                logging.error('Unknown error for star {}'.format(i))
                trap_pars = (np.nan, np.nan, np.nan)

            if use_pbar and pbar_ok:
                pbar.update(i)
            durs[i], deps[i], slopes[i] = trap_pars

        logging.info('Done.')

        self.stars['depth'] = deps
        self.stars['duration'] = durs
        self.stars['slope'] = slopes
        self.stars['secdepth'] = dsec
        self.stars['secondary'] = secs

        self._make_kde()

    @property
    def eclipse_features(self):
        stars = self.stars
        ok = (stars.depth > 0).values
        stars = stars[ok]
        texp = self.cadence

        # Define features
        sec = stars.secondary
        pri = ~sec
        P = stars.P
        T14 = sec*stars.T14_sec + pri*stars.T14_pri
        T23 = sec*stars.T23_sec + pri*stars.T23_pri
        T14 += texp
        T23 = np.clip(T23 - texp, 0, T14)
        tau = (T14 - T23)/2.
        k = (sec*(stars.radius_A/stars.radius_B) +
             ~sec*(stars.radius_B/stars.radius_A))
        b = sec*(stars.b_sec/k) + pri*stars.b_pri
        logd = np.log10(sec*stars.dsec + pri*stars.dpri)
        u1 = sec*stars.u1_2 + pri*stars.u1_1
        u2 = sec*stars.u2_2 + pri*stars.u2_1
        #fluxfrac = sec*stars.fluxfrac_2 + pri*stars.fluxfrac_1
        dilution = self.dilution_factor[ok]

        X = np.array([P,T14,tau,k,b,logd,u1,u2,dilution,sec]).T
        return X

    @property
    def eclipse_targets(self):
        ok = (self.stars.depth > 0).values
        stars = self.stars[ok]
        duration = np.array(stars.duration)
        logdepth = np.array(np.log10(stars.depth))
        slope = np.array(stars.slope)
        return duration, logdepth, slope

    def apply_multicolor_transit(self, band, depth):
        raise NotImplementedError('multicolor transit not yet implemented')

    @property
    def eclipseprob(self):
        """
        Array of eclipse probabilities.
        """
        #TODO: incorporate eccentricity/omega for exact calculation?
        s = self.stars
        return ((s['radius_1'] + s['radius_2'])*RSUN /
                (semimajor(s['P'],s['mass_1'] + s['mass_2'])*AU))

    @property
    def mean_eclipseprob(self):
        """Mean eclipse probability for population
        """
        return self.eclipseprob.mean()

    @property
    def modelshort(self):
        """
        Short version of model name

        Dictionary defined in ``populations.py``::

            SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}


        """
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
        """
        Multiplicative factor (<1) that converts true depth to diluted depth.
        """
        return np.ones(len(self.stars))

    @property
    def depth(self):
        """
        Observed primary depth (fitted undiluted depth * dilution factor)
        """
        return self.dilution_factor * self.stars['depth']

    @property
    def secondary_depth(self):
        """
        Observed secondary depth (fitted undiluted sec. depth * dilution factor)
        """
        return self.dilution_factor * self.stars['secdepth']

    def constrain_secdepth(self, thresh):
        """
        Constrain the observed secondary depth to be less than a given value

        :param thresh:
            Maximum allowed fractional depth for diluted secondary
            eclipse depth

        """
        self.apply_constraint(UpperLimit(self.secondary_depth, thresh, name='secondary depth'))

    def apply_secthresh(self, *args, **kwargs):
        """Another name for constrain_secdepth
        """
        return self.constrain_secdepth(*args, **kwargs)

    def fluxfrac_eclipsing(self, band=None):
        """Stub for future multicolor transit implementation
        """
        pass

    def depth_in_band(self, band):
        """Stub for future multicolor transit implementation
        """
        pass

    @property
    def prior(self):
        """
        Model prior for particular model.

        Product of eclipse probability (``self.prob``),
        the fraction of scenario that is allowed by the various
        constraints (``self.selectfrac``), and all additional
        factors in ``self.priorfactors``.

        """
        prior = self.prob * self.selectfrac
        for f in self.priorfactors:
            prior *= self.priorfactors[f]
        return prior

    def add_priorfactor(self,**kwargs):
        """Adds given values to priorfactors

        If given keyword exists already, error will be raised
        to use :func:`EclipsePopulation.change_prior` instead.
        """
        for kw in kwargs:
            if kw in self.priorfactors:
                logging.error('%s already in prior factors for %s.  use change_prior function instead.' % (kw,self.model))
                continue
            else:
                self.priorfactors[kw] = kwargs[kw]
                logging.info('%s added to prior factors for %s' % (kw,self.model))

    def change_prior(self, **kwargs):
        """
        Changes existing priorfactors.

        If given keyword isn't already in priorfactors,
        then will be ignored.
        """
        for kw in kwargs:
            if kw in self.priorfactors:
                self.priorfactors[kw] = kwargs[kw]
                logging.info('{0} changed to {1} for {2} model'.format(kw,kwargs[kw],
                                                                       self.model))

    def _make_kde(self, use_sklearn=False, bandwidth=None, rtol=1e-6,
                  sig_clip=50, no_sig_clip=False, cov_all=True,
                  **kwargs):
        """Creates KDE objects for 3-d shape parameter distribution

        KDE represents likelihood as function of trapezoidal
        shape parameters (log(delta), T, T/tau).

        Uses :class:`scipy.stats.gaussian_kde`` KDE by default;
        Scikit-learn KDE implementation tested a bit, but not
        fully implemented.

        :param use_sklearn:
            Whether to use scikit-learn implementation of KDE.
            Not yet fully implemented, so this should stay ``False``.

        :param bandwidth, rtol:
            Parameters for sklearn KDE.

        :param **kwargs:
            Additional keyword arguments passed to
            :class:`scipy.stats.gaussian_kde``.

        """

        try:
            #define points that are ok to use
            first_ok = ((self.stars['slope'] > 0) &
                        (self.stars['duration'] > 0) &
                        (self.stars['duration'] < self.period) &
                        (self.depth > 0))
        except KeyError:
            logging.warning('Must do trapezoid fits before making KDE.')
            return

        self.empty = False
        if first_ok.sum() < 4:
            logging.warning('Empty population ({}): < 4 valid systems! Cannot calculate lhood.'.format(self.model))
            self.is_empty = True #will cause is_ruled_out to be true as well.
            return
            #raise EmptyPopulationError('< 4 valid systems in population')

        logdeps = np.log10(np.ma.array(self.depth, mask=~first_ok))
        durs = np.ma.array(self.stars['duration'], mask=~first_ok)
        slopes = np.ma.array(self.stars['slope'], mask=~first_ok)

        #Now sigma-clip those points that passed first cuts
        ok = np.ones(len(logdeps), dtype=bool)
        for x in [logdeps, durs, slopes]:
            med = np.ma.median(x)
            mad = np.ma.median((x - med).__abs__())
            after_clip = np.ma.masked_where((x - med).__abs__() / mad > sig_clip, x)
            ok &= ~after_clip.mask

        second_ok = ok & first_ok
        assert np.allclose(second_ok, ok)

        # Before making KDE for real, first calculate
        #  covariance and inv_cov of uncut data, to use
        #  when it's cut, too.

        points = np.ma.array([logdeps,
                              durs,
                              slopes], mask=np.row_stack((~second_ok, ~second_ok, ~second_ok)))

        points = points.compress(~points.mask[0],axis=1).data
        #from numpy.linalg import LinAlgError
        
        try:
          from scipy import linalg
          kde = gaussian_kde(points) #backward compatibility?
          inv = linalg.inv(kde._data_covariance)
          #print(np.vstack(points), np.shape(np.vstack(points)))
        except np.linalg.linalg.LinAlgError:
          print(points, np.shape(points))
        cov_all = kde._data_covariance
        icov_all = kde._data_inv_cov
        factor = kde.factor

        # OK, now cut the data for constraints & proceed

        ok = second_ok & self.distok

        points = np.ma.array([durs,
                              logdeps,
                              slopes], mask=np.row_stack((~ok, ~ok, ~ok)))
        points = points.compress(~points.mask[0],axis=1)
        logdeps = points.data[1]
        durs = points.data[0]
        slopes = points.data[2]

        if ok.sum() < 4 and not self.empty:
            logging.warning('Empty population ({}): < 4 valid systems! Cannot calculate lhood.'.format(self.model))
            self.is_empty = True
            return
            #raise EmptyPopulationError('< 4 valid systems in population')


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
            try:
              points_skl = normalize(np.transpose([durs, logdeps, slopes]))
            except ValueError:
              from nose.tools import set_trace; set_trace()
              set_trace()
            #assert np.allclose(points_pre, points_skl)

            #find best bandwidth.  For some reason this doesn't work?
            if bandwidth is None:
                bandwidths = np.linspace(0.05,1,100)
                grid = GridSearchCV(KernelDensity(kernel='gaussian'),\
                    {'bandwidth': bandwidths},\
                    cv=3)
                grid.fit(points_skl)
                self._best_bandwidth = grid.best_params_
                self.kde = grid.best_estimator_
            else:
                self.kde = KernelDensity(rtol=rtol, bandwidth=bandwidth).fit(points_skl)
        else:
            self.sklearn_kde = False
            #Yangyang: method 1
            points = (points+1e-07*np.random.uniform(-1.0, 1.0, np.shape(points))).data
            self.kde = gaussian_kde(points, **kwargs) #backward compatibility?

            # Reset covariance based on uncut data
            self.kde._data_covariance = cov_all
            self.kde._data_inv_cov = icov_all
            self.kde._compute_covariance()


    def _density(self, dataset):
        """
        Evaluate KDE at given points.

        Prepares data according to whether sklearn or scipy
        KDE in use.

        :param log, dur, slope:
            Trapezoidal shape parameters.
        """
        if self.sklearn_kde:
            #TODO: fix preprocessing
            #Yangyang's modification(method2):
            #pts = np.array([(logd - self.mean_logdepth)/self.std_logdepth,
            #                (dur - self.mean_dur)/self.std_dur,
            #                (slope - self.mean_slope)/self.std_slope])
            pts = normalize(dataset.T)#(#sample, #features)to make consistent with scipy method, besides their density is in log, then...
            return np.exp(self.kde.score_samples(pts))
        else:
            return self.kde(dataset)

    def lhood(self, trsig, recalc=False, cachefile=None):
        """Returns likelihood of transit signal

        Returns sum of ``trsig`` MCMC samples evaluated
        at ``self.kde``.

        :param trsig:
            :class:`vespa.TransitSignal` object.

        :param recalc: (optional)
            Whether to recalculate likelihood (if calculation
            is cached).

        :param cachefile: (optional)
            File that holds likelihood calculation cache.

        """
        if not hasattr(self,'kde'):
            self._make_kde()

        if cachefile is None:
            cachefile = self.lhoodcachefile
            if cachefile is None:
                cachefile = 'lhoodcache.dat'

        lhoodcache = _loadcache(cachefile)
        key = hashcombine(self, trsig)
        if key in lhoodcache and not recalc:
            return lhoodcache[key]

        if self.is_ruled_out:
            return 0

        N = trsig.kde.dataset.shape[1]
        lh = np.sum(self._density(trsig.kde.dataset)) / N

        with open(cachefile, 'a') as fout:
            fout.write('%i %g\n' % (key, lh))

        return lh


    def lhoodplot(self, trsig=None, fig=None,
                  piechart=True, figsize=None, logscale=True,
                  constraints='all', suptitle=None, Ltot=None,
                  maxdur=None, maxslope=None, inverse=False,
                  colordict=None, cachefile=None, nbins=20,
                  dur_range=None, slope_range=None, depth_range=None,
                  recalc=False,**kwargs):
        """
        Makes plot of likelihood density function, optionally with transit signal

        If ``trsig`` not passed, then just density plot of the likelidhoo
        will be made; if it is passed, then it will be plotted
        over the density plot.

        :param trsig: (optional)
            :class:`vespa.TransitSignal` object.

        :param fig: (optional)
            Argument for :func:`plotutils.setfig`.

        :param piechart: (optional)
            Whether to include a plot of the piechart that describes
            the effect of the constraints on the population.

        :param figsize: (optional)
            Passed to :func:`plotutils.setfig`.

        :param logscale: (optional)
            If ``True``, then shading will be based on the log-histogram
            (thus showing more detail at low density).  Passed to
            :func:`vespa.stars.StarPopulation.prophist2d`.

        :param constraints: (``'all', 'none'`` or ``list``; optional)
            Which constraints to apply in making plot.  Picking
            specific constraints allows you to visualize in more
            detail what the effect of a constraint is.

        :param suptitle: (optional)
            Title for the figure.

        :param Ltot: (optional)
            Total of ``prior * likelihood`` for all models.  If this is
            passed, then "Probability of scenario" gets a text box
            in the middle.

        :param inverse: (optional)
            Intended to allow showing only the instances that are
            ruled out, rather than those that remain.  Not sure if this
            works anymore.

        :param colordict: (optional)
            Dictionary to define colors of constraints to be used
            in pie chart.  Intended to unify constraint colors among
            different models.

        :param cachefile: (optional)
            Likelihood calculation cache file.

        :param nbins: (optional)
            Number of bins with which to make the 2D histogram plot;
            passed to :func:`vespa.stars.StarPopulation.prophist2d`.

        :param dur_range, slope_range, depth_range: (optional)
            Define ranges of plots.

        :param **kwargs:
            Additional keyword arguments passed to
            :func:`vespa.stars.StarPopulation.prophist2d`.

        """

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
            lhood = self.lhood(trsig, recalc=recalc)
            plt.annotate('%s:\nProbability\nof scenario: %.3f' % (trsig.name,
                                                                  self.prior*lhood/Ltot),
                         xy=(0.5,0.5),ha='center',va='center',
                         bbox=dict(boxstyle='round',fc='w'),
                         xycoords='figure fraction',fontsize=15)

    def eclipse_pars(self, i, secondary=False):
        s = self.stars.iloc[i]
        P = s['P']

        #p0, b, aR = eclipse_pars(P, s['mass_1'], s['mass_2'],
        #                         s['radius_1'], s['radius_2'],
        #                         ecc=s['ecc'], inc=s['inc'],
        #                         w=s['w'])

        p0 = s['radius_2']/s['radius_1']
        aR = semimajor(P, s['mass_1']+s['mass_2'])*AU/(s['radius_1']*RSUN)
        if secondary:
            mu1, mu2 = s[['u1_2', 'u2_2']]
            b = s['b_sec']
            frac = s['fluxfrac_2']
        else:
            mu1, mu2 = s[['u1_1', 'u2_1']]
            b = s['b_pri']
            frac = s['fluxfrac_1']

        return dict(P=P, p0=p0, b=b, aR=aR, frac=frac, u1=mu1, u2=mu2,
                    ecc=s['ecc'], w=s['w'])

    def eclipse(self, i, secondary=False, **kwargs):
        pars = self.eclipse_pars(i, secondary=secondary)

        for k,v in pars.items():
            kwargs[k] = v

        return eclipse(sec=secondary, **kwargs)

    def eclipse_trapfit(self, i, secondary=False, **kwargs):
        pars = self.eclipse_pars(i, secondary=secondary)

        for k,v in pars.items():
            kwargs[k] = v
        kwargs['cadence'] = self.cadence

        return eclipse_tt(sec=secondary, **kwargs)

    def eclipse_new(self, i, secondary=False, npoints=200, width=3,
                texp=None):
        """
        Returns times and fluxes of eclipse i (centered at t=0)
        """
        texp = self.cadence
        s = self.stars.iloc[i]

        e = s['ecc']
        P = s['P']
        if secondary:
            mu1, mu2 = s[['u1_2', 'u2_2']]
            w = np.mod(np.deg2rad(s['w']) + np.pi, 2*np.pi)
            mass_central, radius_central = s[['mass_2','radius_2']]
            mass_body, radius_body = s[['mass_1','radius_1']]
            b = s['b_sec'] * s['radius_1']/s['radius_2']
            frac = s['fluxfrac_2']
        else:
            mu1, mu2 = s[['u1_1', 'u2_1']]
            w = np.deg2rad(s['w'])
            mass_central, radius_central = s[['mass_1','radius_1']]
            mass_body, radius_body = s[['mass_2','radius_2']]
            b = s['b_pri']
            frac = s['fluxfrac_1']


        central_kwargs = dict(mass=mass_central, radius=radius_central,
                              mu1=mu1, mu2=mu2)
        central = Central(**central_kwargs)

        body_kwargs = dict(radius=radius_body, mass=mass_body, b=b,
                           period=P, e=e, omega=w)
        body = Body(**body_kwargs)

        logging.debug('central: {}'.format(central_kwargs))
        logging.debug('body: {}'.format(body_kwargs))

        s = System(central)
        s.add_body(body)

        # As of now, body.duration returns strictly circular duration
        dur = body.duration

        logging.debug('duration: {}'.format(dur))

        ts = np.linspace(-width/2*dur, width/2*dur, npoints)
        fs = s.light_curve(ts, texp=texp)
        fs = 1 - frac*(1-fs)
        return ts, fs

    @property
    def _properties(self):
        return ['period','model','priorfactors','prob','lhoodcachefile',
                'is_specific', 'cadence'] + \
            super(EclipsePopulation,self)._properties

    @classmethod
    def load_hdf(cls, filename, path=''): #perhaps this doesn't need to be written?
        """
        Loads EclipsePopulation from HDF file

        Also runs :func:`EclipsePopulation._make_kde` if it can.

        :param filename:
            HDF file

        :param path: (optional)
            Path within HDF file

        """

        new = StarPopulation.load_hdf(filename, path=path)

        #setup lazy loading of starmodel if present
        try:
            with pd.HDFStore(filename) as store:
                if '{}/starmodel'.format(path) in store:
                    new._starmodel = None
                    new._starmodel_file = filename
                    new._starmodel_path = '{}/starmodel'.format(path)
        except:
            pass

        try:
            new._make_kde()
        except NoTrapfitError:
            logging.warning('Trapezoid fit not done.')
        return new


    @property
    def starmodel(self):
        if not hasattr(self, '_starmodel'):
            raise AttributeError('{} does not have starmodel.'.format(self))

        if (hasattr(self, '_starmodel_file') and hasattr(self, '_starmodel_path')):
            self._starmodel = StarModel.load_hdf(self._starmodel_file,
                                                 path=self._starmodel_path)

        return self._starmodel

    def resample(self):
        """
        Returns a copy of population with stars resampled (with replacement).

        Used in bootstrap estimate of FPP uncertainty.

        TODO: check to make sure constraints properly copied!
        """
        new = copy.deepcopy(self)
        N = len(new.stars)
        inds = np.random.randint(N, size=N)

        # Resample stars
        new.stars = new.stars.iloc[inds].reset_index()

        # Resample constraints
        if hasattr(new, '_constraints'):
            for c in new._constraints:
                new._constraints[c] = new._constraints[c].resample(inds)

        new._make_kde()
        return new


class EclipsePopulation_Px2(EclipsePopulation):
    def apply_secthresh(self, *args, **kwargs):
        logging.warning('Secondary depth cut should not be used on a double-period scenario!')

    @property
    def depth_difference(self):
        return np.absolute(self.depth - self.secondary_depth)

    def constrain_oddeven(self, diff):
        self.apply_constraint(UpperLimit(self.depth_difference, diff, name='odd-even'))

class PlanetPopulation(EclipsePopulation):
    """Population of Transiting Planets

    Subclass of :class:`EclipsePopulation`.  This is mostly
    a copy of :class:`EBPopulation`, with small modifications.

    Star properties may be defined either with either a
    :class:`isochrones.StarModel` or by defining just its
    ``mass`` and ``radius`` (and ``Teff`` and ``logg`` if
    desired to set limb darkening coefficients appropriately).

    :param period:
        Period of signal.

    :param rprs:
        Point-estimate of Rp/Rs radius ratio.

    :param mass, radius: (optional)
        Mass and radius of host star.  If defined, must be
        either tuples of form ``(value, error)`` or
        :class:`simpledist.Distribution` objects.

    :param Teff, logg: (optional)
        Teff and logg point estimates for host star.
        These are used only for calculating limb darkening
        coefficients.

    :param starmodel: (optional)
        The preferred way to define the properties of the
        host star.  If MCMC has been run on this model,
        then samples are just read off; if it hasn't,
        then it will run it.
    :type starmodel:
        :class:`isochrones.StarModel`

    :param band: (optional)
        Photometric band in which eclipse is detected.

    :param model: (optional)
        Name of the model.

    :param n: (optional)
        Number of instances to simulate.  Default = ``2e4``.

    :param fp_specific: (optional)
        "Specific occurrence rate" for this type of planets;
        that is, the planet occurrence rate integrated
        from ``(1-rbin_width)x`` to ``(1+rbin_width)x`` this planet radius.  This
        goes into the ``priorfactor`` for this model.

    :param u1, u2: (optional)
        Limb darkening parameters.  If not provided, then
        calculated based on ``Teff, logg`` or just
        defaulted to solar values.

    :param rbin_width: (optional)
        Fractional width of rbin for ``fp_specific``.

    :param MAfn: (optional)
        :class:`transit_basic.MAInterpolationFunction` object.
        If not passed, then one with default parameters will
        be created.

    :param lhoodcachefile: (optional)
        Likelihood calculation cache file.

    """

    def __init__(self, period=None,
                 cadence=1626./86400, #Kepler observing cadence, in days
                 rprs=None,
                 mass=None, radius=None, Teff=None, logg=None,
                 starmodel=None,
                 band='Kepler', model='Planets', n=2e4,
                 fp_specific=None, u1=None, u2=None,
                 rbin_width=0.3,
                 MAfn=None, lhoodcachefile=None):

        self.period = period
        self.cadence = cadence
        self.n = n
        self.model = model
        self.band = band
        self.rprs = rprs
        self.Teff = Teff
        self.logg = logg
        self._starmodel = starmodel

        if radius is not None and mass is not None or starmodel is not None:
            # calculates eclipses
            logging.debug('generating planet population...')
            self.generate(rprs=rprs, mass=mass, radius=radius,
                          n=n, fp_specific=fp_specific,
                          starmodel=starmodel,
                          rbin_width=rbin_width,
                          u1=u1, u2=u2, Teff=Teff, logg=logg,
                          MAfn=MAfn,lhoodcachefile=lhoodcachefile)

    def generate(self,rprs=None, mass=None, radius=None,
                n=2e4, fp_specific=0.01, u1=None, u2=None,
                 starmodel=None,
                Teff=None, logg=None, rbin_width=0.3,
                MAfn=None, lhoodcachefile=None):
        """Generates Population

        All arguments defined in ``__init__``.
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
            mass = samples['mass_0_0'].values
            radius = samples['radius_0_0'].values
            Teff = samples['Teff_0_0'].mean()
            logg = samples['logg_0_0'].mean()

        logging.debug('star mass: {}'.format(mass))
        logging.debug('star radius: {}'.format(radius))
        logging.debug('Teff: {}'.format(Teff))
        logging.debug('logg: {}'.format(logg))

        if u1 is None or u2 is None:
            if Teff is None or logg is None:
                logging.warning('Teff, logg not provided; using solar limb darkening')
                u1 = 0.394; u2=0.296
            else:
                u1,u2 = ldcoeffs(Teff, logg)

        #use point estimate of rprs to construct planets in radius bin
        #rp = self.rprs*np.median(radius)
        #rbin_min = (1-rbin_width)*rp
        #rbin_max = (1+rbin_width)*rp

        rprs_bin_min = (1-rbin_width)*self.rprs
        rprs_bin_max = (1+rbin_width)*self.rprs

        radius_p = radius * (np.random.random(int(1e5))*(rprs_bin_max - rprs_bin_min) + rprs_bin_min)
        mass_p = (radius_p*RSUN/REARTH)**2.06 * MEARTH/MSUN #hokey, but doesn't matter

        logging.debug('planet radius: {}'.format(radius_p))

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

            ok = (df['dpri']>0) & (df['T14_pri'] > 0)

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

        if fp_specific is None:
            rp = stars['radius_2'].mean() * RSUN/REARTH
            fp_specific = fp_fressin(rp)

        priorfactors = {'fp_specific':fp_specific}

        self._starmodel = starmodel

        EclipsePopulation.__init__(self, stars=stars,
                                   period=self.period, cadence=self.cadence,
                                   model=self.model,
                                   priorfactors=priorfactors, prob=tot_prob,
                                   lhoodcachefile=lhoodcachefile)
    @property
    def _properties(self):
        return ['rprs', 'Teff', 'logg'] + \
            super(PlanetPopulation, self)._properties

    def save_hdf(self, filename, path='', **kwargs):
        super(PlanetPopulation, self).save_hdf(filename, path=path, **kwargs)
        self.starmodel.save_hdf(filename, path='{}/starmodel'.format(path), append=True)

    #@classmethod
    #def load_hdf(cls, filename, path=''):
    #    pop = super(PlanetPopulation, cls).load_hdf(filename, path=path)
    #    pop.starmodel = StarModel.load_hdf(filename,
    #                                       path='{}/starmodel'.format(path))
    #    return pop

class EBPopulation(EclipsePopulation, Observed_BinaryPopulation):
    """Population of Eclipsing Binaries (undiluted)

    Eclipsing Binary (EB) population is generated by fitting
    a two-star model to the observed properties of the system
    (photometric and/or spectroscopic), using
    :class:`isochrones.starmodel.BinaryStarModel`.


    Inherits from :class:`EclipsePopulation` and
    :class:`stars.Observed_BinaryPopulation`.

    :param period:
        Orbital period

    :param mags:
        Observed apparent magnitudes.  Won't work if this is
        ``None``, which is the default.
    :type mags:
        ``dict``

    :param Teff,logg,feh:
        Spectroscopic properties of primary, if measured, in ``(value, err)`` format.

    :param starmodel: (optional)
        Must be a BinaryStarModel.
        If MCMC has been run on this model,
        then samples are just read off; if it hasn't,
        then it will run it.
    :type starmodel:
        :class:`isochrones.BinaryStarModel`

    :param band: (optional)
        Photometric bandpass in which transit signal is observed.

    :param model:  (optional)
        Name of model.

    :param f_binary: (optional)
        Binary fraction to be assumed.  Will be one of the ``priorfactors``.

    :param n: (optional)
        Number of instances to simulate.  Default = 2e4.

    :param MAfn: (optional)
        :class:`transit_basic.MAInterpolationFunction` object.
        If not passed, then one with default parameters will
        be created.

    :param lhoodcachefile: (optional)
        Likelihood calculation cache file.

    """

    def __init__(self, period=None,
                 cadence=1626./86400, #Kepler observing cadence, in days
                 mags=None, mag_errs=None,
                 Teff=None, logg=None, feh=None,
                 starmodel=None,
                 band='Kepler', model='EBs', f_binary=0.4, n=2e4,
                 MAfn=None, lhoodcachefile=None, **kwargs):

        self.period = period
        self.cadence = cadence
        self.n = n
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile

        if mags is not None or starmodel is not None:
            self.generate(mags=mags, n=n, MAfn=MAfn, mag_errs=mag_errs,
                          f_binary=f_binary, starmodel=starmodel,
                          **kwargs)

    def generate(self, mags, n=2e4, mag_errs=None,
                 Teff=None, logg=None, feh=None,
                 MAfn=None, f_binary=0.4, starmodel=None,
                 **kwargs):
        """Generates stars and eclipses

        All arguments previously defined.
        """
        n = int(n)


        #create master population from which to create eclipses
        pop = Observed_BinaryPopulation(mags=mags, mag_errs=mag_errs,
                                        Teff=Teff,
                                        logg=logg, feh=feh,
                                        starmodel=starmodel,
                                        period=self.period,
                                        n=2*n)

        all_stars = pop.stars

        #start with empty; will concatenate onto
        stars = pd.DataFrame()
        df_orbpop = pd.DataFrame()


        #calculate eclipses

        if MAfn is None:
            MAfn = MAInterpolationFunction(pmin=0.007, pmax=1/0.007, nzs=200, nps=400)

        tot_prob = None; tot_dprob = None; prob_norm = None
        n_adapt = n
        while len(stars) < n:
            n_adapt = int(n_adapt)
            inds = np.random.randint(len(all_stars), size=n_adapt)

            s = all_stars.iloc[inds]

            #calculate limb-darkening coefficients
            u1A, u2A = ldcoeffs(s['Teff_A'], s['logg_A'])
            u1B, u2B = ldcoeffs(s['Teff_B'], s['logg_B'])

            cur_orbpop_df = pop.orbpop.dataframe.iloc[inds].copy()

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

            new_df_orbpop = cur_orbpop_df.iloc[inds].copy()
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
        orbpop = OrbitPopulation.from_df(df_orbpop)

        stars = stars.reset_index()
        stars.drop('index', axis=1, inplace=True)

        stars['mass_1'] = stars['mass_A']
        stars['radius_1'] = stars['radius_A']
        stars['mass_2'] = stars['mass_B']
        stars['radius_2'] = stars['radius_B']

        ## Why does this make it go on infinite loop??
        #Observed_BinaryPopulation.__init__(self, stars=stars, orbpop=orbpop,
        #                                   mags=mags, mag_errs=mag_errs,
        #                                   Teff=Teff, logg=logg, feh=feh,
        #                                   starmodel=starmodel)
        ###########

        self.mags = mags
        self.mag_errs = mag_errs
        self.Teff = Teff
        self.logg = logg
        self.feh = feh
        self._starmodel = pop.starmodel

        priorfactors = {'f_binary':f_binary}

        EclipsePopulation.__init__(self, stars=stars, orbpop=orbpop,
                                   period=self.period, cadence=self.cadence,
                                   model=self.model,
                                   priorfactors=priorfactors, prob=tot_prob,
                                   lhoodcachefile=self.lhoodcachefile)

class EBPopulation_Px2(EclipsePopulation_Px2, EBPopulation):
    def __init__(self, period=None, model='EBs (Double Period)',
                 **kwargs):
        try:
            period *= 2
        except:
            pass

        EBPopulation.__init__(self, period=period, model=model,
                              **kwargs)

class HEBPopulation(EclipsePopulation, Observed_TriplePopulation):
    """Population of Hierarchical Eclipsing Binaries

    Hierarchical Eclipsing Binary (HEB) population is generated
    by fitting
    a two-star model to the observed properties of the system
    (photometric and/or spectroscopic), using
    :class:`isochrones.starmodel.BinaryStarModel`.

    by

    Inherits from :class:`EclipsePopulation` and
    :class:`stars.Observed_TriplePopulation`.

    :param period:
        Orbital period

    :param mags,mag_errs:
        Observed apparent magnitudes; uncertainties optional.  If
        uncertainties not provided, :class:`Observed_TriplePopulation`
        will default to uncertainties in all bands of 0.05 mag.
    :type mags:
        ``dict``

    :param Teff,logg,feh:
        Spectroscopic properties of primary, if measured, in ``(value, err)`` format.

    :param starmodel: (optional)
        Must be a BinaryStarModel.
        If MCMC has been run on this model,
        then samples are just read off; if it hasn't,
        then it will run it.
    :type starmodel:
        :class:`isochrones.BinaryStarModel`

    :param band: (optional)
        Photometric bandpass in which transit signal is observed.

    :param model:  (optional)
        Name of model.

    :param f_binary: (optional)
        Binary fraction to be assumed.  Will be one of the ``priorfactors``.

    :param n: (optional)
        Number of instances to simulate.  Default = 2e4.

    :param MAfn: (optional)
        :class:`transit_basic.MAInterpolationFunction` object.
        If not passed, then one with default parameters will
        be created.

    :param lhoodcachefile: (optional)
        Likelihood calculation cache file.

    """

    def __init__(self, period=None,
                 cadence=1626./86400, #Kepler observing cadence, in days
                 mags=None, mag_errs=None,
                 Teff=None, logg=None, feh=None,
                 starmodel=None,
                 band='Kepler', model='HEBs', f_triple=0.12, n=2e4,
                 MAfn=None, lhoodcachefile=None, **kwargs):

        self.period = period
        self.cadence = cadence
        self.n = n
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile

        if mags is not None or starmodel is not None:
            self.generate(mags=mags, n=n, MAfn=MAfn, mag_errs=mag_errs,
                          f_triple=f_triple, starmodel=starmodel,
                          **kwargs)

    def generate(self, mags, n=2e4, mag_errs=None,
                 Teff=None, logg=None, feh=None,
                 MAfn=None, f_triple=0.12, starmodel=None,
                 **kwargs):
        """Generates stars and eclipses

        All arguments previously defined.
        """
        n = int(n)


        #create master population from which to create eclipses
        pop = Observed_TriplePopulation(mags=mags, mag_errs=mag_errs,
                                        Teff=Teff,
                                        logg=logg, feh=feh,
                                        starmodel=starmodel,
                                        period=self.period,
                                        n=2*n)

        all_stars = pop.stars

        #start with empty; will concatenate onto
        stars = pd.DataFrame()
        df_orbpop_short = pd.DataFrame()
        df_orbpop_long = pd.DataFrame()


        #calculate eclipses

        if MAfn is None:
            MAfn = MAInterpolationFunction(pmin=0.007, pmax=1/0.007, nzs=200, nps=400)

        tot_prob = None; tot_dprob = None; prob_norm = None
        n_adapt = n
        while len(stars) < n:
            n_adapt = int(n_adapt)
            inds = np.random.randint(len(all_stars), size=n_adapt)

            s = all_stars.iloc[inds]

            #calculate limb-darkening coefficients
            u1A, u2A = ldcoeffs(s['Teff_A'], s['logg_A'])
            u1B, u2B = ldcoeffs(s['Teff_B'], s['logg_B'])
            u1C, u2C = ldcoeffs(s['Teff_C'], s['logg_C'])

            cur_orbpop_short_df = pop.orbpop.orbpop_short.dataframe.iloc[inds].copy()
            cur_orbpop_long_df = pop.orbpop.orbpop_long.dataframe.iloc[inds].copy()

            #calculate eclipses.
            inds, df, (prob,dprob) = calculate_eclipses(s['mass_B'], s['mass_C'],
                                                        s['radius_B'], s['radius_C'],
                                                        s['{}_mag_B'.format(self.band)],
                                                        s['{}_mag_C'.format(self.band)],
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

            new_df_orbpop_short = cur_orbpop_short_df.iloc[inds].copy()
            new_df_orbpop_short.reset_index(inplace=True)

            new_df_orbpop_long = cur_orbpop_long_df.iloc[inds].copy()
            new_df_orbpop_long.reset_index(inplace=True)

            df_orbpop_short = pd.concat((df_orbpop_short, new_df_orbpop_short))
            df_orbpop_long = pd.concat((df_orbpop_long, new_df_orbpop_long))

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
            n_adapt = max(n_adapt, 100)

        stars = stars.iloc[:n]
        df_orbpop_short = df_orbpop_short.iloc[:n]
        df_orbpop_long = df_orbpop_long.iloc[:n]
        orbpop = TripleOrbitPopulation.from_df(df_orbpop_long, df_orbpop_short)

        stars = stars.reset_index()
        stars.drop('index', axis=1, inplace=True)

        stars['mass_1'] = stars['mass_B']
        stars['radius_1'] = stars['radius_B']
        stars['mass_2'] = stars['mass_C']
        stars['radius_2'] = stars['radius_C']

        ## Why does this make it go on infinite loop??
        #Observed_TriplePopulation.__init__(self, stars=stars, orbpop=orbpop,
        #                                   mags=mags, mag_errs=mag_errs,
        #                                   Teff=Teff, logg=logg, feh=feh,
        #                                   starmodel=starmodel)
        #############

        self.mags = mags
        self.mag_errs = mag_errs
        self.Teff = Teff
        self.logg = logg
        self.feh = feh
        self._starmodel = pop.starmodel

        priorfactors = {'f_triple':f_triple}

        EclipsePopulation.__init__(self, stars=stars, orbpop=orbpop,
                                   period=self.period, cadence=self.cadence,
                                   model=self.model,
                                   priorfactors=priorfactors, prob=tot_prob,
                                   lhoodcachefile=self.lhoodcachefile)

class HEBPopulation_Px2(EclipsePopulation_Px2, HEBPopulation):
    def __init__(self, period=None, model='HEBs (Double Period)',
                 **kwargs):
        try:
            period *= 2
        except TypeError:
            pass

        HEBPopulation.__init__(self, period=period, model=model,
                               **kwargs)

class BEBPopulation(EclipsePopulation, MultipleStarPopulation,
                    BGStarPopulation):
    """
    Population of "Background" eclipsing binaries (BEBs)

    :param period:
        Orbital period.

    :param mags:
        Observed apparent magnitudes of target (foreground)
        star.  Must have at least magnitude in band
        that eclipse is measured in (``band`` argument).
    :type mags:
        ``dict``

    :param ra,dec: (optional)
        Coordinates of star (to simulate field star population).
        If ``trilegal_filename`` not provided, then TRILEGAL
        simulation will be generated.

    :param trilegal_filename:
        Name of file that contains TRILEGAL field star
        simulation to use.  Should always be provided
        if population is to be generated.  If file
        does not exist, then TRILEGAL simulation
        will be saved as this filename (use .h5 extension).

    :param n: (optional)
        Size of simulation.  Default is 2e4.

    :param ichrone: (optional)
        :class:`isochrones.Isochrone` object to use
        to generate stellar models.

    :param band: (optional)
        Photometric bandpass in which eclipse signal is observed.

    :param maxrad: (optional)
        Maximum radius [arcsec] from target star to assign to BG stars.

    :param f_binary: (optional)
        Assumed binary fraction.  Will be part of ``priorfactors``.

    :param model: (optional)
        Model name.

    :param MAfn: (optional)
        :class:`transit_basic.MAInterpolationFunction` object.
        If not passed, then one with default parameters will
        be created.

    :param lhoodcachefile: (optional)
        Likelihood calculation cache file.

    :param **kwargs:
        Additional keyword arguments passed to
        :class:`stars.BGStarPopulation_TRILEGAL`.


    """
    def __init__(self, period=None,
                 cadence=1626./86400, #Kepler observing cadence, in days
                 mags=None,
                 ra=None, dec=None, trilegal_filename=None,
                 n=2e4, ichrone='mist', band='Kepler',
                 maxrad=10, f_binary=0.4, model='BEBs',
                 MAfn=None, lhoodcachefile=None,
                 **kwargs):
        self.period = period
        self.cadence = cadence
        self.n = n
        self.model = model
        self.band = band
        self.lhoodcachefile = lhoodcachefile
        self.mags = mags

        if trilegal_filename is not None or (ra is not None
                                            and dec is not None):
            if self.band not in self.mags:
                raise ValueError('{} band must be in mags.'.format(self.band))

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
                 n=2e4, ichrone='mist', MAfn=None,
                 mags=None, maxrad=None, f_binary=0.4, **kwargs):
        """
        Generate population.
        """
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

        ichrone = get_ichrone(ichrone)

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
                                   period=self.period, cadence=self.cadence,
                                   model=self.model,
                                   lhoodcachefile=self.lhoodcachefile,
                                   priorfactors=priorfactors, prob=tot_prob)

        #add Rsky property
        self.stars['Rsky'] = randpos_in_circle(len(self.stars),
                                               self._maxrad, return_rad=True)

    @property
    def _properties(self):
        return ['density','trilegal_args','mags'] + \
          super(BEBPopulation, self)._properties


class BEBPopulation_Px2(EclipsePopulation_Px2, BEBPopulation):
    def __init__(self, period=None, model='BEBs (Double Period)',
                 **kwargs):
        try:
            period *= 2
        except TypeError:
            pass

        BEBPopulation.__init__(self, period=period, model=model,
                               **kwargs)

class PopulationSet(object):
    """
    A set of EclipsePopulations used to calculate a transit signal FPP

    This can be initialized with a list of :class:`EclipsePopulation` objects
    that have been pre-generated, or it can be passed the arguments required
    to generate the default list of :class:`EclipsePopulation`s.

    :param poplist:
        Can be either a list of :class:`EclipsePopulation` objects,
        a filename (in which case a saved :class:`PopulationSet`
        will be loaded), or ``None``, in which case the populations
        will be generated.

    :param period:
        Orbital period of signal.

    :param mags:
        Observed magnitudes of target star.
    :type mags:
        ``dict``

    :param n:
        Size of simulations.  Default is 2e4.

    :param ra, dec: (optional)
        Target star position; passed to :class:`BEBPopulation`.

    :param trilegal_filename:
        Passed to :class:`BEBPopulation`.

    :param mass, age, feh, radius: (optional)
        Properties of target star.  Either in ``(value, error)`` form
        or as :class:`simpledist.Distribution` objects.  Not necessary
        if ``starmodel`` is passed.

    :param starmodel: (optional)
        The preferred way to define the properties of the
        host star.  If MCMC has been run on this model,
        then samples are just read off; if it hasn't,
        then it will run it.
    :type starmodel:
        :class:`isochrones.StarModel`

    :param rprs:
        R_planet/R_star.  Single-value estimate.

    :param MAfn: (optional)
        :class:`transit_basic.MAInterpolationFunction` object.
        If not passed, then one with default parameters will
        be created.

    :param colors: (optional)
        Colors to use to constrain multiple star populations;
        passed to :class:`EBPopulation` and :class:`HEBPopulation`.
        Default will be ['JK', 'HK']

    :param Teff, logg: (optional)
        If ``starmodel`` not provided, then these can be used
        (single values only) in order for :class:`PlanetPopulation`
        to use the right limb darkening parameters.

    :param savefile: (optional)
        HDF file in which to save :class:`PopulationSet`.

    :param heb_kws, eb_kws, beb_kws, pl_kws: (optional)
        Keyword arguments to pass on to respective
        :class:`EclipsePopulation` constructors.

    :param hide_exceptions: (optional)
        If ``True``, then exceptions generated during
        population simulations will be passed, not raised.

    :param fit_trap: (optional)
        If ``True``, then population generation will also
        call :func:`EclipsePopulation.fit_trapezoids` for each
        model population.

    :param do_only: (optional)
        Can be defined in order to make only a subset of populations.
        List or tuple should contain modelname shortcuts
        (e.g., 'beb', 'heb', 'eb', or 'pl').


    """
    def __init__(self, poplist=None,
                 period=None,
                 cadence=1626./86400, #Kepler observing cadence, in days
                 mags=None, n=2e4,
                 ra=None, dec=None, trilegal_filename=None,
                 Teff=None, logg=None, feh=None,
                 starmodel=None,
                 binary_starmodel=None,
                 triple_starmodel=None,
                 rprs=None,
                 MAfn=None,
                 savefile=None,
                 heb_kws=None, eb_kws=None,
                 beb_kws=None, pl_kws=None,
                 hide_exceptions=False,
                 fit_trap=True, do_only=None):
        #if string is passed, load from file
        if poplist is None:
            self.generate(ra, dec, period, cadence, mags,
                          n=n, MAfn=MAfn,
                          trilegal_filename=trilegal_filename,
                          Teff=Teff, logg=logg, feh=feh,
                          rprs=rprs,
                          savefile=savefile, starmodel=starmodel,
                          binary_starmodel=binary_starmodel,
                          triple_starmodel=triple_starmodel,
                          heb_kws=heb_kws, eb_kws=eb_kws,
                          beb_kws=beb_kws, pl_kws=pl_kws,
                          hide_exceptions=hide_exceptions,
                          fit_trap=fit_trap,
                          do_only=do_only)

        elif type(poplist)==type(''):
            self = PopulationSet.load_hdf(poplist)
        else:
            self.poplist = poplist

    def generate(self, ra, dec, period, cadence, mags,
                 n=2e4, Teff=None, logg=None, feh=None,
                 MAfn=None,
                 rprs=None, trilegal_filename=None,
                 starmodel=None,
                 binary_starmodel=None, triple_starmodel=None,
                 heb_kws=None, eb_kws=None,
                 beb_kws=None, pl_kws=None, savefile=None,
                 hide_exceptions=False, fit_trap=True,
                 do_only=None):
        """
        Generates PopulationSet.
        """
        do_all = False
        if do_only is None:
            do_all = True
            do_only = DEFAULT_MODELS

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

        if 'heb' in do_only:
            try:
                hebpop = HEBPopulation(mags=mags,
                                       Teff=Teff, logg=logg, feh=feh,
                                       period=period, cadence=cadence,
                                       starmodel=triple_starmodel,
                                       starfield=trilegal_filename,
                                       MAfn=MAfn, n=n, **heb_kws)
                if fit_trap:
                    hebpop.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    if do_all:
                        hebpop.save_hdf(savefile, 'heb', overwrite=True)
                    else:
                        hebpop.save_hdf(savefile, 'heb', append=True)
            except:
                logging.error('Error generating HEB population.')
                if not hide_exceptions:
                    raise

        if 'heb_Px2' in do_only:
            try:
                hebpop_Px2 = HEBPopulation_Px2(mags=mags,
                                       Teff=Teff, logg=logg, feh=feh,
                                       period=period, cadence=cadence,
                                       starmodel=triple_starmodel,
                                       starfield=trilegal_filename,
                                       MAfn=MAfn, n=n, **heb_kws)
                if fit_trap:
                    hebpop_Px2.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    if do_all:
                        hebpop_Px2.save_hdf(savefile, 'heb_Px2', overwrite=True)
                    else:
                        hebpop_Px2.save_hdf(savefile, 'heb_Px2', append=True)
            except:
                logging.error('Error generating HEB_Px2 population.')
                if not hide_exceptions:
                    raise

        if 'eb' in do_only:
            try:
                ebpop = EBPopulation(mags=mags,
                                     Teff=Teff, logg=logg, feh=feh,
                                     period=period, cadence=cadence,
                                     starmodel=binary_starmodel,
                                     starfield=trilegal_filename,
                                     MAfn=MAfn, n=n, **eb_kws)
                if fit_trap:
                    ebpop.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    ebpop.save_hdf(savefile, 'eb', append=True)
            except:
                logging.error('Error generating EB population.')
                if not hide_exceptions:
                    raise

        if 'eb_Px2' in do_only:
            try:
                ebpop_Px2 = EBPopulation_Px2(mags=mags,
                                     Teff=Teff, logg=logg, feh=feh,
                                     period=period, cadence=cadence,
                                     starmodel=binary_starmodel,
                                     starfield=trilegal_filename,
                                     MAfn=MAfn, n=n, **eb_kws)
                if fit_trap:
                    ebpop_Px2.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    ebpop_Px2.save_hdf(savefile, 'eb_Px2', append=True)
            except:
                logging.error('Error generating EB_Px2 population.')
                if not hide_exceptions:
                    raise

        if 'beb' in do_only:
            try:
                bebpop = BEBPopulation(trilegal_filename=trilegal_filename,
                                       ra=ra, dec=dec, period=period, cadence=cadence,
                                       mags=mags, MAfn=MAfn, n=n, **beb_kws)
                if fit_trap:
                    bebpop.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    bebpop.save_hdf(savefile, 'beb', append=True)
            except:
                logging.error('Error generating BEB population.')
                if not hide_exceptions:
                    raise

        if 'beb_Px2' in do_only:
            try:
                bebpop_Px2 = BEBPopulation_Px2(trilegal_filename=trilegal_filename,
                                       ra=ra, dec=dec, period=period, cadence=cadence,
                                       mags=mags, MAfn=MAfn, n=n, **beb_kws)
                if fit_trap:
                    bebpop_Px2.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    bebpop_Px2.save_hdf(savefile, 'beb_Px2', append=True)
            except:
                logging.error('Error generating BEB_Px2 population.')
                if not hide_exceptions:
                    raise

        if 'pl' in do_only:
            try:
                plpop = PlanetPopulation(period=period, cadence=cadence,
                                         rprs=rprs,
                                         starmodel=starmodel,
                                         MAfn=MAfn, n=n, **pl_kws)

                if fit_trap:
                    plpop.fit_trapezoids(MAfn=MAfn)
                if savefile is not None:
                    plpop.save_hdf(savefile, 'pl', append=True)
            except:
                logging.error('Error generating Planet population.')
                if not hide_exceptions:
                    raise

        if not do_all and savefile is not None:
            hebpop = HEBPopulation.load_hdf(savefile, 'heb')
            hebpop_Px2 = HEBPopulation.load_hdf(savefile, 'heb_Px2')
            ebpop = EBPopulation.load_hdf(savefile, 'eb')
            ebpop_Px2 = EBPopulation.load_hdf(savefile, 'eb_Px2')
            bebpop = BEBPopulation.load_hdf(savefile, 'beb')
            bebpop_Px2 = BEBPopulation.load_hdf(savefile, 'beb_Px2')
            plpop = PlanetPopulation.load_hdf(savefile, 'pl')


        self.poplist = [hebpop, hebpop_Px2,
                        ebpop, ebpop_Px2,
                        bebpop, bebpop_Px2, plpop]

    @property
    def constraints(self):
        """
        Unique list of constraints among all populations in set.
        """
        cs = []
        for pop in self.poplist:
            cs += [c for c in pop.constraints]
        return list(set(cs))

    @property
    def modelnames(self):
        """
        List of model names
        """
        return [pop.model for pop in self.poplist]

    @property
    def shortmodelnames(self):
        """
        List of short modelnames.
        """
        return [pop.modelshort for pop in self.poplist]

    def save_hdf(self, filename, path='', overwrite=False):
        """
        Saves PopulationSet to HDF file.
        """
        if os.path.exists(filename) and overwrite:
            os.remove(filename)

        for pop in self.poplist:
            name = pop.modelshort
            pop.save_hdf(filename, path='{}/{}'.format(path,name), append=True)

    @classmethod
    def load_hdf(cls, filename, path=''):
        """
        Loads PopulationSet from file
        """
        with pd.HDFStore(filename) as store:
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

        return cls(poplist) #how to deal with saved constraints?
        #PopulationSet.__init__(self, poplist) #how to deal with saved constraints?
        #return self

    def add_population(self,pop):
        """Adds population to PopulationSet
        """
        if pop.model in self.modelnames:
            raise ValueError('%s model already in PopulationSet.' % pop.model)
        self.modelnames.append(pop.model)
        self.shortmodelnames.append(pop.modelshort)
        self.poplist.append(pop)
        #self.apply_dmaglim()

    def remove_population(self,pop):
        """Removes population from PopulationSet
        """
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
        elif name in ['eb_Px2', 'ebs_Px2', 'eb_px2', 'ebs_Px2']:
            name = 'ebs (double period)'
        elif name in ['heb_Px2', 'hebs_Px2', 'heb_px2', 'hebs_px2']:
            name = 'hebs (double period)'
        elif name in ['beb_Px2', 'bebs_Px2', 'beb_px2', 'bebs_px2']:
            name = 'bebs (double period)'
        for pop in self.poplist:
            if name==pop.model.lower():
                return pop
        raise ValueError('%s not in modelnames: %s' % (name,self.modelnames))

    @property
    def colordict(self):
        """
        Dictionary holding colors that correspond to constraints.
        """
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
        """Combinartion of priorfactors from all populations
        """
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
        """Changes prior factor(s) in all populations
        """
        for kw,val in kwargs.items():
            if kw=='area':
                logging.warning('cannot change area in this way--use change_maxrad instead')
                continue
            for pop in self.poplist:
                k = {kw:val}
                pop.change_prior(**k)

    def apply_multicolor_transit(self,band,depth):
        """
        Applies constraint corresponding to measuring transit in different band

        This is not implemented yet.
        """
        if '{} band transit'.format(band) not in self.constraints:
            self.constraints.append('{} band transit'.format(band))
        for pop in self.poplist:
            pop.apply_multicolor_transit(band,depth)

    def set_maxrad(self,newrad):
        """
        Sets max allowed radius in populations.

        Doesn't operate via the :class:`stars.Constraint`
        protocol; rather just rescales the sky positions
        for the background objects and recalculates
        sky area, etc.

        """
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
        """
        Applies a constraint that sets the maximum brightness for non-target star

        :func:`stars.StarPopulation.set_dmaglim` not yet implemented.

        """
        raise NotImplementedError
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

    def apply_trend_constraint(self, limit, dt, **kwargs):
        """
        Applies constraint corresponding to RV trend non-detection to each population

        See :func:`stars.StarPopulation.apply_trend_constraint`;
        all arguments passed to that function for each population.

        """
        if 'RV monitoring' not in self.constraints:
            self.constraints.append('RV monitoring')
        for pop in self.poplist:
            if not hasattr(pop,'dRV'):
                continue
            pop.apply_trend_constraint(limit, dt, **kwargs)
        self.trend_limit = limit
        self.trend_dt = dt

    def apply_secthresh(self, secthresh, **kwargs):
        """Applies secondary depth constraint to each population

        See :func:`EclipsePopulation.apply_secthresh`;
        all arguments passed to that function for each population.

        """

        if 'secondary depth' not in self.constraints:
            self.constraints.append('secondary depth')
        for pop in self.poplist:
            if not isinstance(pop, EclipsePopulation_Px2):
                pop.apply_secthresh(secthresh, **kwargs)
        self.secthresh = secthresh

    def constrain_oddeven(self, diff, **kwargs):
        """Constrains the difference b/w primary and secondary to be < diff
        """
        if 'odd-even' not in self.constraints:
            self.constraints.append('odd-even')
        for pop in self.poplist:
            if isinstance(pop, EclipsePopulation_Px2):
                pop.constrain_oddeven(diff, **kwargs)
        self.oddeven_diff = diff



    def constrain_property(self,prop,**kwargs):
        """
        Constrains property for each population

        See :func:`vespa.stars.StarPopulation.constrain_property`;
        all arguments passed to that function for each population.

        """
        if prop not in self.constraints:
            self.constraints.append(prop)
        for pop in self.poplist:
            try:
                pop.constrain_property(prop,**kwargs)
            except AttributeError:
                logging.info('%s model does not have property stars.%s (constraint not applied)' % (pop.model,prop))

    def replace_constraint(self,name,**kwargs):
        """
        Replaces removed constraint in each population.

        See :func:`vespa.stars.StarPopulation.replace_constraint`

        """

        for pop in self.poplist:
            pop.replace_constraint(name,**kwargs)
        if name not in self.constraints:
            self.constraints.append(name)

    def remove_constraint(self,*names):
        """
        Removes constraint from each population

        See :func:`vespa.stars.StarPopulation.remove_constraint

        """
        for name in names:
            for pop in self.poplist:
                if name in pop.constraints:
                    pop.remove_constraint(name)
                else:
                    logging.info('%s model does not have %s constraint' % (pop.model,name))
            if name in self.constraints:
                self.constraints.remove(name)

    def apply_cc(self, cc, **kwargs):
        """
        Applies contrast curve constraint to each population

        See :func:`vespa.stars.StarPopulation.apply_cc`;
        all arguments passed to that function for each population.

        """
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
        """
        Applies velocity contrast curve constraint to each population

        See :func:`vespa.stars.StarPopulation.apply_vcc`;
        all arguments passed to that function for each population.

        """
        if 'secondary spectrum' not in self.constraints:
            self.constraints.append('secondary spectrum')
        for pop in self.poplist:
            if not pop.is_specific:
                try:
                    pop.apply_vcc(vcc)
                except:
                    logging.info('VCC constraint not applied to %s model' % (pop.model))

    def resample(self):
        new = copy.deepcopy(self)
        new_poplist = [pop.resample() for pop in new.poplist]
        new.poplist = new_poplist
        return new



############ Utility Functions ##############

def calculate_eclipses(M1s, M2s, R1s, R2s, mag1s, mag2s,
                       u11s=0.394, u21s=0.296, u12s=0.394, u22s=0.296,
                       Ps=None, period=None, logperkde=RAGHAVAN_LOGPERKDE,
                       incs=None, eccs=None,
                       mininc=None, calc_mininc=True,
                       maxecc=0.97, ecc_fn=draw_eccs,
                       band='Kepler',
                       return_probability_only=False, return_indices=True,
                       MAfn=None):
    """Returns random eclipse parameters for provided inputs


    :param M1s, M2s, R1s, R2s, mag1s, mag2s: (array-like)
        Primary and secondary properties (mass, radius, magnitude)

    :param u11s, u21s, u12s, u22s: (optional)
        Limb darkening parameters (u11 = u1 for star 1, u21 = u2 for star 1, etc.)

    :param Ps: (array-like, optional)
        Orbital periods; same size as ``M1s``, etc.
        If only a single period is desired, use ``period``.

    :param period: (optional)
        Orbital period; use this keyword if only a single period is desired.

    :param logperkde: (optional)
        If neither ``Ps`` nor ``period`` is provided, then periods will be
        randomly generated according to this log-period distribution.
        Default is taken from the Raghavan (2010) period distribution.

    :param incs, eccs: (optional)
        Inclinations and eccentricities.  If not passed, they will be generated.
        Eccentricities will be generated according to ``ecc_fn``; inclinations
        will be randomly generated out to ``mininc``.

    :param mininc: (optional)
        Minimum inclination to generate.  Useful if you want to enhance
        efficiency by only generating mostly eclipsing, instead of mostly
        non-eclipsing systems.  If not provided and ``calc_mininc`` is
        ``True``, then this will be calculated based on inputs.

    :param calc_mininc: (optional)
        Whether to calculate ``mininc`` based on inputs.  If truly isotropic
        inclinations are desired, set this to ``False``.

    :param maxecc: (optional)
        Maximum eccentricity to generate.

    :param ecc_fn: (callable, optional)
        Orbital eccentricity generating function.  Must return ``n`` orbital
        eccentricities generated according to provided period(s)::

            eccs = ecc_fn(n,Ps)

        Defaults to :func:`stars.utils.draw_eccs`.

    :param band: (optional)
        Photometric bandpass in which eclipse is observed.

    :param return_probability_only: (optional)
        If ``True``, then will return only the average eclipse probability
        of population.

    :param return_indices: (optional)
        If ``True``, returns the indices of the original input arrays
        that the output ``DataFrame`` corresponds to.  **This behavior
        will/should be changed to just return a ``DataFrame`` of the same
        length as inputs...**

    :param MAfn: (optional)
        :class:`transit_basic.MAInterpolationFunction` object.
        If not passed, then one with default parameters will
        be created.

    :return:
        * [``wany``: indices describing which of the original input
          arrays the output ``DataFrame`` corresponds to.
        * ``df``: ``DataFrame`` with the following columns:
          ``[{band}_mag_tot, P, ecc, inc, w, dpri, dsec,
             T14_pri, T23_pri, T14_sec, T23_sec, b_pri,
             b_sec, {band}_mag_1, {band}_mag_2, fluxfrac_1,
             fluxfrac_2, switched, u1_1, u2_1, u1_2, u2_2]``.
             **N.B. that this will be shorter than your input arrays,
             because not everything will eclipse; this behavior
             will likely be changed in the future because it's confusing.**
        * ``(prob, dprob)`` Eclipse probability with Poisson uncertainty

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
            eccs = ecc_fn(n,period,maxecc=maxecc)
        else:
            eccs = ecc_fn(n,Ps,maxecc=maxecc)
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


class ArtificialPopulation(EclipsePopulation):
    """ A population with contrived likelihood function

    prior : The model prior for this population
    lhoodfn : a normalized PDF of (duration, log(depth), slope)

    must define prior, _lhoodfn

    """
    #def __init__(self, prior, lhoodfn):
    #    self._prior = prior
    #    self._lhoodfn = lhoodfn

    @property
    def prior(self):
        return self._prior

    def lhood(self, trsig, **kwargs):
        N = trsig.kde.dataset.shape[1]
        lh = self._lhoodfn(trsig.kde.dataset).sum() / N
        return lh

    @property
    def priorfactors(self):
        return {}

    def resample(self):
        return copy.deepcopy(self)

class BoxyModel(ArtificialPopulation):
    max_slope = MAXSLOPE
    logd_range = (-5,0)
    dur_range = (0,2)
    model='boxy'
    modelshort='boxy'

    def __init__(self, prior, min_slope):
        self._prior = prior
        self.min_slope = min_slope

    def _lhoodfn(self, x):
        level = 1./((self.logd_range[1]-self.logd_range[0])*
                    (self.dur_range[1]-self.dur_range[0])*
                    (self.max_slope-self.min_slope))
        return level*(x[2,:] > self.min_slope)


class LongModel(ArtificialPopulation):
    slope_range = (2,15)
    logd_range = (0,5)
    max_dur = 2.
    model='long'
    modelshort='long'

    def __init__(self, prior, min_dur):
        self._prior = prior
        self.min_dur = min_dur

    def _lhoodfn(self, x):
        level = 1./((self.logd_range[1]-self.logd_range[0])*
                    (self.slope_range[1]-self.slope_range[0])*
                    (self.max_dur-self.min_dur))
        return level*(x[0,:] > self.min_dur)

#####################
###### Utility functions

def fp_fressin(rp,dr=None):
    if dr is None:
        dr = rp*0.3
    fp = quad(fressin_occurrence,rp-dr,rp+dr)[0]
    return max(fp, 0.001) #to avoid zero

def fressin_occurrence(rp):
    """Occurrence rates per bin from Fressin+ (2013)
    """
    rp = np.atleast_1d(rp)

    sq2 = np.sqrt(2)
    bins = np.array([1/sq2,1,sq2,2,2*sq2,
                     4,4*sq2,8,8*sq2,
                     16,16*sq2])
    rates = np.array([0,0.155,0.155,0.165,0.17,0.065,0.02,0.01,0.012,0.01,0.002,0])

    return rates[np.digitize(rp,bins)]


def _loadcache(cachefile):
    """ Returns a dictionary resulting from reading a likelihood cachefile
    """
    cache = {}
    if os.path.exists(cachefile):
        with open(cachefile) as f:
            for line in f:
                line = line.split()
                if len(line) == 2:
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
