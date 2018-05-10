from __future__ import division, print_function

import os,os.path
import logging
import pickle

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy.random as rand
    from scipy.stats import gaussian_kde
    import corner
    from emcee.autocorr import integrated_time, AutocorrError
    from astropy.io import ascii
else:
    np, pd, plt, rand = (None, None, None, None)
    gaussian_kde = None



try:
    import corner
except ImportError:
    pass

if not on_rtd:
    from .plotutils import setfig
    from .hashutils import hashcombine, hasharray
    from .transit_basic import traptransit, fit_traptransit, traptransit_MCMC, MAXSLOPE
    from .statutils import kdeconf, qstd, conf_interval
else:
    MAXSLOPE = None

def load_pkl(filename):
    with open(filename, 'rb') as fin:
        return pickle.load(fin)

class TransitSignal(object):
    """A phased-folded transit signal.

    Epoch of the transit at 0, 'continuum' set at 1.

    :param ts, fs, dfs:
        Times (days from mid-transit), fluxes (relative to 1),
        flux uncertainties.  dfs optional

    :param P:
        Orbital period.

    :param p0: (optional)
        Initial guess for least-squares trapezoid fit.
        If not provided, then some decent guess will be made
        (which is better on made-up data than real...)

    :param name: (optional)
        Name of the signal.

    :param maxslope: (optional)
        Upper limit to use for "slope" parameter (T/tau)
        in the MCMC fitting of signal.  Default is 15.


    .. note:: The implementation of this object can use some refactoring;
         as it is directly translated from some older code.  As
         such, not all methods/attributes are well documented.


    """
    def __init__(self,ts,fs,dfs=None,P=None,p0=None,name='',maxslope=MAXSLOPE):

        ts = np.atleast_1d(ts)
        fs = np.atleast_1d(fs)

        inds = ts.argsort()
        self.ts = ts[inds]
        self.fs = fs[inds]
        self.name = name
        self.P = P

        self.maxslope = maxslope
        if type(P) == type(np.array([1])):
            self.P = P[0]

        #set default best-guess trapezoid parameters
        if p0 is None:
            depth = 1 - fs.min()
            duration = (fs < (1-0.01*depth)).sum()/float(len(fs)) * (ts[-1] - ts[0])
            tc0 = ts[fs.argmin()]
            p0 = np.array([duration,depth,5.,tc0])

        tfit = fit_traptransit(ts,fs,p0)

        if dfs is None:
            dfs = (self.fs - traptransit(self.ts,tfit)).std()
        if np.size(dfs)==1:
            dfs = np.ones(len(self.ts))*dfs
        self.dfs = dfs

        self.dur,self.depth,self.slope,self.center = tfit
        self.trapfit = tfit

        logging.debug('trapezoidal leastsq fit: {}'.format(self.trapfit))

        self.hasMCMC = False

    @classmethod
    def from_ascii(cls, filename, **kwargs):
        table = ascii.read(filename).to_pandas()
        if len(table.columns)==3:
            return cls(table.iloc[:, 0].values, table.iloc[:, 1].values, table.iloc[:, 2].values,
                        **kwargs)
        elif len(table.columns)==2:
            return cls(table.iloc[:, 0].values, table.iloc[:, 1].values,
                        **kwargs)


    def save_hdf(self, filename, path=''):
        """
        Save transitsignal info using HDF...not yet implemented.

        .. note::

           Refactoring plan is to re-write saving to use HDF
           instead of pickle.
        """
        raise NotImplementedError


    def triangle(self, **kwargs):
        pts = np.array([self.logdeps, self.durs, self.slopes]).T
        fig = corner.corner(pts, labels=['log (Depth)',
                                           'Duration', 'T/tau'], **kwargs)
        return fig

    def save_pkl(self, filename):
        """
        Pickles TransitSignal.
        """
        with open(filename, 'wb') as fout:
            pickle.dump(self, fout)

    #eventually make this save_hdf
    def save(self, filename):
        """
        Calls save_pkl function.
        """
        self.save_pkl(filename)

    def __eq__(self,other):
        return hash(self) == hash(other)

    def __hash__(self):
        key =  hashcombine(hasharray(self.ts),
                           hasharray(self.fs),
                           self.P,
                           self.maxslope)
        if self.hasMCMC:
            key = hashcombine(key, hasharray(self.slopes),
                              hasharray(self.durs),
                              hasharray(self.logdeps))
        return key

    def plot(self, fig=None, plot_trap=False, name=False, trap_color='g',
             trap_kwargs=None, **kwargs):
        """
        Makes a simple plot of signal

        :param fig: (optional)
            Argument for :func:`plotutils.setfig`.

        :param plot_trap: (optional)
            Whether to plot the (best-fit least-sq) trapezoid fit.

        :param name: (optional)
            Whether to annotate plot with the name of the signal;
            can be ``True`` (in which case ``self.name`` will be
            used), or any arbitrary string.

        :param trap_color: (optional)
            Color of trapezoid fit line.

        :param trap_kwargs: (optional)
            Keyword arguments to pass to trapezoid fit line.

        :param **kwargs: (optional)
            Additional keyword arguments passed to ``plt.plot``.


        """

        setfig(fig)

        plt.plot(self.ts,self.fs,'.',**kwargs)

        if plot_trap and hasattr(self,'trapfit'):
            if trap_kwargs is None:
                trap_kwargs = {}
            plt.plot(self.ts, traptransit(self.ts,self.trapfit),
                     color=trap_color, **trap_kwargs)

        if name is not None:
            if type(name)==type(''):
                text = name
            else:
                text = self.name
            plt.annotate(text,xy=(0.1,0.1),xycoords='axes fraction',fontsize=22)

        if hasattr(self,'depthfit') and not np.isnan(self.depthfit[0]):
            lo = 1 - 3*self.depthfit[0]
            hi = 1 + 2*self.depthfit[0]
        else:
            lo = 1
            hi = 1

        sig = qstd(self.fs,0.005)
        hi = max(hi,self.fs.mean() + 7*sig)
        lo = min(lo,self.fs.mean() - 7*sig)
        logging.debug('lo={}, hi={}'.format(lo,hi))
        plt.ylim((lo,hi))
        plt.xlabel('time [days]')
        plt.ylabel('Relative flux')

    def MCMC(self, niter=500, nburn=200, nwalkers=200, threads=1,
             fit_partial=False, width=3, savedir=None, refit=False,
             thin=10, conf=0.95, maxslope=MAXSLOPE, debug=False, p0=None):
        """
        Fit transit signal to trapezoid model using MCMC

        .. note:: As currently implemented, this method creates a
            bunch of attributes relevant to the MCMC fit; I plan
            to refactor this to define those attributes as properties
            so as not to have their creation hidden away here.  I plan
            to refactor how this works.
        """
        if fit_partial:
            wok = np.where((np.absolute(self.ts-self.center) < (width*self.dur)) &
                           ~np.isnan(self.fs))
        else:
            wok = np.where(~np.isnan(self.fs))

        if savedir is not None:
            if not os.path.exists(savedir):
                os.mkdir(savedir)

        alreadydone = True
        alreadydone &= savedir is not None
        alreadydone &= os.path.exists('%s/ts.npy' % savedir)
        alreadydone &= os.path.exists('%s/fs.npy' % savedir)

        if savedir is not None and alreadydone:
            ts_done = np.load('%s/ts.npy' % savedir)
            fs_done = np.load('%s/fs.npy' % savedir)
            alreadydone &= np.all(ts_done == self.ts[wok])
            alreadydone &= np.all(fs_done == self.fs[wok])

        if alreadydone and not refit:
            logging.info('MCMC fit already done for %s.  Loading chains.' % self.name)
            Ts = np.load('%s/duration_chain.npy' % savedir)
            ds = np.load('%s/depth_chain.npy' % savedir)
            slopes = np.load('%s/slope_chain.npy' % savedir)
            tcs = np.load('%s/tc_chain.npy' % savedir)
        else:
            logging.info('Fitting data to trapezoid shape with MCMC for %s....' % self.name)
            if p0 is None:
                p0 = self.trapfit.copy()
                p0[0] = np.absolute(p0[0])
                if p0[2] < 2:
                    p0[2] = 2.01
                if p0[1] < 0:
                    p0[1] = 1e-5
            logging.debug('p0 for MCMC = {}'.format(p0))
            sampler = traptransit_MCMC(self.ts[wok],self.fs[wok],self.dfs[wok],
                                        niter=niter,nburn=nburn,nwalkers=nwalkers,
                                        threads=threads,p0=p0,return_sampler=True,
                                        maxslope=maxslope)

            Ts,ds,slopes,tcs = (sampler.flatchain[:,0],sampler.flatchain[:,1],
                                sampler.flatchain[:,2],sampler.flatchain[:,3])

            self.sampler = sampler
            if savedir is not None:
                np.save('%s/duration_chain.npy' % savedir,Ts)
                np.save('%s/depth_chain.npy' % savedir,ds)
                np.save('%s/slope_chain.npy' % savedir,slopes)
                np.save('%s/tc_chain.npy' % savedir,tcs)
                np.save('%s/ts.npy' % savedir,self.ts[wok])
                np.save('%s/fs.npy' % savedir,self.fs[wok])

        if debug:
            print(Ts)
            print(ds)
            print(slopes)
            print(tcs)

        N = len(Ts)
        try:
            self.Ts_acor = integrated_time(Ts)
            self.ds_acor = integrated_time(ds)
            self.slopes_acor = integrated_time(slopes)
            self.tcs_acor = integrated_time(tcs)
            self.fit_converged = True
        except AutocorrError:
            self.fit_converged = False


        ok = (Ts > 0) & (ds > 0) & (slopes > 0) & (slopes < self.maxslope)
        logging.debug('trapezoidal fit has {} good sample points'.format(ok.sum()))
        if ok.sum()==0:
            if (Ts > 0).sum()==0:
                #logging.debug('{} points with Ts > 0'.format((Ts > 0).sum()))
                logging.debug('{}'.format(Ts))
                raise MCMCError('{}: 0 points with Ts > 0'.format(self.name))
            if (ds > 0).sum()==0:
                #logging.debug('{} points with ds > 0'.format((ds > 0).sum()))
                logging.debug('{}'.format(ds))
                raise MCMCError('{}: 0 points with ds > 0'.format(self.name))
            if (slopes > 0).sum()==0:
                #logging.debug('{} points with slopes > 0'.format((slopes > 0).sum()))
                logging.debug('{}'.format(slopes))
                raise MCMCError('{}: 0 points with slopes > 0'.format(self.name))
            if (slopes < self.maxslope).sum()==0:
                #logging.debug('{} points with slopes < maxslope ({})'.format((slopes < self.maxslope).sum(),self.maxslope))
                logging.debug('{}'.format(slopes))
                raise MCMCError('{} points with slopes < maxslope ({})'.format((slopes < self.maxslope).sum(),self.maxslope))


        durs,deps,logdeps,slopes = (Ts[ok],ds[ok],np.log10(ds[ok]),
                                              slopes[ok])


        inds = (np.arange(len(durs)/thin)*thin).astype(int)
        durs,deps,logdeps,slopes = (durs[inds],deps[inds],logdeps[inds],
                                              slopes[inds])

        self.durs,self.deps,self.logdeps,self.slopes = (durs,deps,logdeps,slopes)

        self._make_kde(conf=conf)

        self.hasMCMC = True

    def corner(self, outfile=None, plot_contours=False, **kwargs):
        fig = corner.corner(self.kde.dataset.T, labels=['Duration', 'log(depth)', 'T/tau'],
                            plot_contours=False, **kwargs)

        if outfile is not None:
            fig.savefig(outfile)

        return fig

    def _make_kde(self, conf=0.95):

        self.durkde = gaussian_kde(self.durs)
        self.depthkde = gaussian_kde(self.deps)
        self.slopekde = gaussian_kde(self.slopes)
        self.logdepthkde = gaussian_kde(self.logdeps)


        if self.fit_converged:
            try:
                durconf = kdeconf(self.durkde,conf)
                depconf = kdeconf(self.depthkde,conf)
                logdepconf = kdeconf(self.logdepthkde,conf)
                slopeconf = kdeconf(self.slopekde,conf)
            except:
                raise
                raise MCMCError('Error generating confidence intervals...fit must not have worked.')

            durmed = np.median(self.durs)
            depmed = np.median(self.deps)
            logdepmed = np.median(self.logdeps)
            slopemed = np.median(self.slopes)

            self.durfit = (durmed,np.array([durmed-durconf[0],durconf[1]-durmed]))
            self.depthfit = (depmed,np.array([depmed-depconf[0],depconf[1]-depmed]))
            self.logdepthfit = (logdepmed,np.array([logdepmed-logdepconf[0],logdepconf[1]-logdepmed]))
            self.slopefit = (slopemed,np.array([slopemed-slopeconf[0],slopeconf[1]-slopemed]))

        else:
            self.durfit = (np.nan,(np.nan,np.nan))
            self.depthfit = (np.nan,(np.nan,np.nan))
            self.logdepthfit = (np.nan,(np.nan,np.nan))
            self.slopefit = (np.nan,(np.nan,np.nan))


        points = np.array([self.durs,self.logdeps,self.slopes])
        self.kde = gaussian_kde(points)


class TransitSignal_FromSamples(TransitSignal):
    """Use this if all you have is the trapezoid-fit samples
    """
    def __init__(self, period, durs, depths, slopes,
                 name='', **kwargs):
        self.period = period
        self.durs = durs
        self.deps = depths
        self.logdeps = np.log10(depths)
        self.slopes = slopes
        self.hasMCMC = True
        self.fit_converged = True #better be
        self._make_kde()
        self.name = name

    def MCMC(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        pass

    def __hash__(self):
        return hashcombine(self.period, hasharray(self.durs),
                           hasharray(self.deps),
                           hasharray(self.slopes))

class TransitSignal_DF(TransitSignal):
    def __init__(self, df, columns=['t','f','e_f'], **kwargs):
        t_col, f_col, e_f_col = columns
        t = df[t_col]
        f = df[f_col]
        if e_f_col in df:
            e_f = df[e_f_col]
        else:
            e_f = None
        TransitSignal.__init__(self, t, f, e_f, **kwargs)

class TransitSignal_ASCII(TransitSignal):
    def __init__(self, filename, cols=(0,1), err_col=2, **kwargs):
        t, f = np.loadtxt(filename, usecols=cols, unpack=True)
        try:
            e_f = np.loadtxt(filename, usecols=(err_col,))
        except:
            e_f = None
        TransitSignal.__init__(self, t, f, e_f, **kwargs)


############# Exceptions ##############3

class MCMCError(Exception):
    pass
