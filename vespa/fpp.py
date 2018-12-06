from __future__ import print_function, division

import os, os.path, re
import logging
from six import string_types
try:
    import cPickle as pickle
except ImportError:
    import pickle


try:
    from configobj import ConfigObj
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    ConfigObj, np, plt, cm = (None, None, None, None)


from .populations import PopulationSet, DEFAULT_MODELS, ArtificialPopulation
from .transitsignal import TransitSignal, MCMCError

from .stars.contrastcurve import ContrastCurveFromFile
from .stars.contrastcurve import VelocityContrastCurve

from .plotutils import setfig
from .hashutils import hashcombine

try:
    from isochrones import StarModel, BinaryStarModel, TripleStarModel
except ImportError:
    StarModel = None
# from .stars.populations import DARTMOUTH

try:
    from tables.exceptions import HDF5ExtError
except ImportError:
    HDF5ExtError = None

class FPPCalculation(object):
    """
    An object to organize an FPP calculation.

    May be created in one of three ways:

    * Manually building a
      :class:`TransitSignal` and a :class:`PopulationSet`
      and then calling the constructor,
    * Loading from a folder in which the correct data
      files have been saved, using :func:`FPPCalculation.load`, or
    * Reading from a config file, using :func:`FPPCalculation.from_ini`


    :param trsig:
        :class:`TransitSignal` object representing the signal
        being modeled.

    :param popset:
        :class:`PopulationSet` object representing the set
        of models being considered as an explanation for
        the signal.

    :param folder: (optional)
        Folder where likelihood cache, results file, plots, etc.
        are written by default.

    """
    def __init__(self, trsig, popset, folder='.'):
        self.trsig = trsig
        self.name = trsig.name
        self.popset = popset
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        if not self.trsig.hasMCMC:
            logging.warning('TransitSignal trapezoid shape has not been fit with MCMC.')
        else:
            if not self.trsig.fit_converged:
                raise MCMCError('TransitSignal trapezoid fit not converged ({}).'.format(self.trsig.name))

        lhoodcachefile = os.path.join(self.folder,'lhoodcache.dat')

        self.lhoodcachefile = lhoodcachefile
        for pop in self.popset.poplist:
            pop.lhoodcachefile = lhoodcachefile

    @classmethod
    def from_ini(cls, folder, ini_file='fpp.ini', ichrone='mist', recalc=False,
                refit_trap=False, **kwargs):
        """
        To enable simple usage, initializes a FPPCalculation from a .ini file

        By default, a file called ``fpp.ini`` will be looked for in the
        current folder.  Also present must be a ``star.ini`` file that
        contains the observed properties of the target star.

        ``fpp.ini`` must be of the following form::

            name = k2oi
            ra = 11:30:14.510
            dec = +07:35:18.21

            period = 32.988 #days
            rprs = 0.0534   #Rp/Rstar
            photfile = lc_k2oi.csv

            [constraints]
            maxrad = 10 #exclusion radius [arcsec]
            secthresh = 0.001 #maximum allowed secondary signal depth

            #This variable defines contrast curves
            #ccfiles = Keck_J.cc, Lick_J.cc

        Photfile must be a text file with columns ``(days_from_midtransit,
        flux, flux_err)``.  Both whitespace- and comma-delimited
        will be tried, using ``np.loadtxt``.  Photfile need not be there
        if there is a pickled :class:`TransitSignal` saved in the same
        directory as ``ini_file``, named ``trsig.pkl`` (or another name
        as defined by ``trsig`` keyword in ``.ini`` file).

        ``star.ini`` should look something like the following::

            B = 15.005, 0.06
            V = 13.496, 0.05
            g = 14.223, 0.05
            r = 12.858, 0.04
            i = 11.661, 0.08
            J = 9.763, 0.03
            H = 9.135, 0.03
            K = 8.899, 0.02
            W1 = 8.769, 0.023
            W2 = 8.668, 0.02
            W3 = 8.552, 0.025
            Kepler = 12.473

            #Teff = 3503, 80
            #feh = 0.09, 0.09
            #logg = 4.89, 0.1

        Any star properties can be defined; if errors are included
        then they will be used in the :class:`isochrones.StarModel`
        MCMC fit.
        Spectroscopic parameters (``Teff, feh, logg``) are optional.
        If included, then they will also be included in
        :class:`isochrones.StarModel` fit.  A magnitude for the
        band in which the transit signal is observed (e.g., ``Kepler``)
        is required, though need not have associated uncertainty.


        :param folder:
            Folder to find configuration files.

        :param ini_file:
            Input configuration file.

        :param star_ini_file:
            Input config file for :class:`isochrones.StarModel` fits.

        :param recalc:
            Whether to re-calculate :class:`PopulationSet`, if a
            ``popset.h5`` file is already present

        :param **kwargs:
            Keyword arguments passed to :class:`PopulationSet`.

        Creates:

            * ``trsig.pkl``: the pickled :class:`vespa.TransitSignal` object.
            * ``starfield.h5``: the TRILEGAL field star simulation
            * ``starmodel.h5``: the :class:`isochrones.StarModel` fit
            * ``popset.h5``: the :class:`vespa.PopulationSet` object
              representing the model population simulations.

        Raises
        ------
        RuntimeError :
            If single, double, and triple starmodels are
            not computed, then raises with admonition to run
            `starfit --all`.

        AttributeError :
            If `trsig.pkl` not present in folder, and
            `photfile` is not defined in config file.

        """

        # Check if all starmodel fits are done.
        # If not, tell user to run 'starfit --all'
        config = ConfigObj(os.path.join(folder, ini_file))

        # Load required entries from ini_file
        try:
            name = config['name']
            ra, dec = config['ra'], config['dec']
            period = float(config['period'])
            rprs = float(config['rprs'])
        except KeyError as err:
            raise KeyError('Missing required element of ini file: {}'.format(err))

        try:
            cadence = float(config['cadence'])
        except KeyError:
            logging.warning('Cadence not provided in fpp.ini; defaulting to Kepler cadence.')
            logging.warning('If this is not a Kepler target, please set cadence (in days).')
            cadence = 1626.0/86400 # Default to Kepler cadence

        try:
            band = config['band']
        except KeyError:
            logging.warning('Transit band not provided in fpp.ini; defaulting to "Kepler."')
            logging.warning('If this is not a Kepler target, please specify filter.')
            band = 'Kepler' # Default to Kepler

        def fullpath(filename):
            if os.path.isabs(filename):
                return filename
            else:
                return os.path.join(folder, filename)

        # Non-required entries with default values
        popset_file = fullpath(config.get('popset', 'popset.h5'))
        starfield_file = fullpath(config.get('starfield', 'starfield.h5'))
        trsig_file = fullpath(config.get('trsig', 'trsig.pkl'))

        # Check for StarModel fits
        starmodel_basename = config.get('starmodel_basename',
                                        '{}_starmodel'.format(ichrone))
        single_starmodel_file = os.path.join(folder,'{}_single.h5'.format(starmodel_basename))
        binary_starmodel_file = os.path.join(folder,'{}_binary.h5'.format(starmodel_basename))
        triple_starmodel_file = os.path.join(folder,'{}_triple.h5'.format(starmodel_basename))

        try:
            single_starmodel = StarModel.load_hdf(single_starmodel_file)
            binary_starmodel = StarModel.load_hdf(binary_starmodel_file)
            triple_starmodel = StarModel.load_hdf(triple_starmodel_file)
        except Exception as e:
            print(e)
            raise RuntimeError('Cannot load StarModels.  ' +
                               'Please run `starfit --all {}`.'.format(folder))

        # Create (or load) TransitSignal
        if os.path.exists(trsig_file):
            logging.info('Loading transit signal from {}...'.format(trsig_file))
            with open(trsig_file, 'rb') as f:
                trsig = pickle.load(f)
        else:
            try:
                photfile = fullpath(config['photfile'])
            except KeyError:
                raise AttributeError('If transit pickle file (trsig.pkl) ' +
                                     'not present, "photfile" must be' +
                                     'defined.')

            trsig = TransitSignal.from_ascii(photfile, P=period, name=name)
            if not trsig.hasMCMC or refit_trap:
                logging.info('Fitting transitsignal with MCMC...')
                trsig.MCMC()
                trsig.save(trsig_file)

        # Create (or load) PopulationSet
        do_only = DEFAULT_MODELS
        if os.path.exists(popset_file):
            if recalc:
                os.remove(popset_file)
            else:
                with pd.HDFStore(popset_file) as store:
                    do_only = [m for m in DEFAULT_MODELS if m not in store]

        # Check that properties of saved population match requested
        try:
            popset = PopulationSet.load_hdf(popset_file)
            for pop in popset.poplist:
                if pop.cadence != cadence:
                    raise ValueError('Requested cadence ({}) '.format(cadence) +
                                    'does not match stored {})! Set recalc=True.'.format(pop.cadence))
        except:
            raise

        if do_only:
            logging.info('Generating {} models for PopulationSet...'.format(do_only))
        else:
            logging.info('Populations ({}) already generated.'.format(DEFAULT_MODELS))

        popset = PopulationSet(period=period, cadence=cadence,
                               band=band,
                               mags=single_starmodel.mags,
                               ra=ra, dec=dec,
                               trilegal_filename=starfield_file, # Maybe change parameter name?
                               starmodel=single_starmodel,
                               binary_starmodel=binary_starmodel,
                               triple_starmodel=triple_starmodel,
                               rprs=rprs, do_only=do_only,
                               savefile=popset_file, **kwargs)


        fpp = cls(trsig, popset, folder=folder)


        #############
        # Apply constraints

        # Exclusion radius
        maxrad = float(config['constraints']['maxrad'])
        fpp.set_maxrad(maxrad)
        if 'secthresh' in config['constraints']:
            secthresh = float(config['constraints']['secthresh'])
            if not np.isnan(secthresh):
                fpp.apply_secthresh(secthresh)

        # Odd-even constraint
        diff = 3 * np.max(trsig.depthfit[1])
        fpp.constrain_oddeven(diff)

        #apply contrast curve constraints if present
        if 'ccfiles' in config['constraints']:
            ccfiles = config['constraints']['ccfiles']
            if isinstance(ccfiles, string_types):
                ccfiles = [ccfiles]
            for ccfile in ccfiles:
                if not os.path.isabs(ccfile):
                    ccfile = os.path.join(folder, ccfile)
                m = re.search('(\w+)_(\w+)\.cc',os.path.basename(ccfile))
                if not m:
                    logging.warning('Invalid CC filename ({}); '.format(ccfile) +
                                     'skipping.')
                    continue
                else:
                    band = m.group(2)
                    inst = m.group(1)
                    name = '{} {}-band'.format(inst, band)
                    cc = ContrastCurveFromFile(ccfile, band, name=name)
                    fpp.apply_cc(cc)

        #apply "velocity contrast curve" if present
        if 'vcc' in config['constraints']:
            dv = float(config['constraints']['vcc'][0])
            dmag = float(config['constraints']['vcc'][1])
            vcc = VelocityContrastCurve(dv, dmag)
            fpp.apply_vcc(vcc)

        return fpp

    def __getattr__(self, attr):
        if attr != 'popset':
            return getattr(self.popset, attr)

    def save(self, overwrite=True):
        """
        Saves PopulationSet and TransitSignal.

        Shouldn't need to use this if you're using
        :func:`FPPCalculation.from_ini`.

        Saves :class`PopulationSet` to ``[folder]/popset.h5]``
        and :class:`TransitSignal` to ``[folder]/trsig.pkl``.

        :param overwrite: (optional)
            Whether to overwrite existing files.

        """
        self.save_popset(overwrite=overwrite)
        self.save_signal()

    @classmethod
    def load(cls, folder):
        """
        Loads PopulationSet from folder

        ``popset.h5`` and ``trsig.pkl`` must exist in folder.

        :param folder:
            Folder from which to load.
        """
        popset = PopulationSet.load_hdf(os.path.join(folder,'popset.h5'))
        sigfile = os.path.join(folder,'trsig.pkl')
        with open(sigfile, 'rb') as f:
            trsig = pickle.load(f)
        return cls(trsig, popset, folder=folder)

    def FPPplots(self, folder=None, format='png', tag=None, **kwargs):
        """
        Make FPP diagnostic plots

        Makes likelihood "fuzz plot" for each model, a FPP summary figure,
        a plot of the :class:`TransitSignal`, and writes a ``results.txt``
        file.

        :param folder: (optional)
            Destination folder for plots/``results.txt``.  Default
            is ``self.folder``.

        :param format: (optional)
            Desired format of figures.  e.g. ``png``, ``pdf``...

        :param tag: (optional)
            If this is provided (string), then filenames will have
            ``_[tag]`` appended to the filename, before the extension.

        :param **kwargs:
            Additional keyword arguments passed to :func:`PopulationSet.lhoodplots`.

        """
        if folder is None:
            folder = self.folder

        self.write_results(folder=folder)
        self.lhoodplots(folder=folder,figformat=format,tag=tag,**kwargs)
        self.FPPsummary(folder=folder,saveplot=True,figformat=format,tag=tag)
        self.plotsignal(folder=folder,saveplot=True,figformat=format)

    def plotsignal(self,fig=None,saveplot=True,folder=None,figformat='png',**kwargs):
        """
        Plots TransitSignal

        Calls :func:`TransitSignal.plot`, saves to provided folder.

        :param fig: (optional)
            Argument for :func:`plotutils.setfig`.

        :param saveplot: (optional)
            Whether to save figure.

        :param folder: (optional)
            Folder to which to save plot

        :param figformat: (optional)
            Desired format for figure.

        :param **kwargs:
            Additional keyword arguments passed to :func:`TransitSignal.plot`.
        """
        if folder is None:
            folder = self.folder

        self.trsig.plot(plot_trap=True,fig=fig,**kwargs)
        if saveplot:
            plt.savefig('%s/signal.%s' % (folder,figformat))
            plt.close()

    def write_results(self,folder=None, filename='results.txt', to_file=True):
        """
        Writes text file of calculation summary.

        :param folder: (optional)
            Folder to which to write ``results.txt``.

        :param filename:
            Filename to write.  Default=``results.txt``.

        :param to_file:
            If True, then writes file.  Otherwise just return header, line.

        :returns:
            Header string, line
        """
        if folder is None:
            folder = self.folder
        if to_file:
            fout = open(os.path.join(folder,filename), 'w')
        header = ''
        for m in self.popset.shortmodelnames:
            header += 'lhood_{0} L_{0} pr_{0} '.format(m)
        header += 'fpV fp FPP'
        if to_file:
            fout.write('{} \n'.format(header))
        Ls = {}
        Ltot = 0
        lhoods = {}
        for model in self.popset.modelnames:
            lhoods[model] = self.lhood(model)
            Ls[model] = self.prior(model)*lhoods[model]
            Ltot += Ls[model]

        line = ''
        for model in self.popset.modelnames:
            line += '%.2e %.2e %.2e ' % (lhoods[model], Ls[model], Ls[model]/Ltot)
        line += '%.3g %.3f %.2e' % (self.fpV(),self.priorfactors['fp_specific'],self.FPP())

        if to_file:
            fout.write(line+'\n')
            fout.close()

        return header, line

    def save_popset(self,filename='popset.h5',**kwargs):
        """Saves the PopulationSet

        Calls :func:`PopulationSet.save_hdf`.
        """
        self.popset.save_hdf(os.path.join(self.folder,filename))

    def save_signal(self,filename=None):
        """
        Saves TransitSignal.

        Calls :func:`TransitSignal.save`; default filename is
        ``trsig.pkl`` in ``self.folder``.
        """
        if filename is None:
            filename = os.path.join(self.folder,'trsig.pkl')
        self.trsig.save(filename)

    def FPPsummary(self,fig=None,figsize=(10,8),saveplot=False,folder='.',
                   starinfo=True,siginfo=True,
                   priorinfo=True,constraintinfo=True,
                   tag=None,simple=False,figformat='png'):
        """
        Makes FPP summary plot

        .. note::

            This is due for updates/improvements.

        :param fig, figsize: (optional)
            Arguments for :func:`plotutils.setfig`.

        :param saveplot: (optional)
            Whether to save figure.  Default is ``False``.

        :param folder: (optional)
            Folder to which to save plot; default is current working dir.

        :param figformat: (optional)
            Desired format of saved figure.

        """
        if simple:
            starinfo = False
            siginfo = False
            priorinfo = False
            constraintinfo = False

        setfig(fig,figsize=figsize)
        # three pie charts
        priors = []; lhoods = []; Ls = []
        for model in self.popset.modelnames:
            priors.append(self.prior(model))
            lhoods.append(self.lhood(model))
            if np.isnan(priors[-1]):
                raise ValueError('{} prior is nan; priorfactors={}'.format(model,
                                                                           self.priorfactors))
            Ls.append(priors[-1]*lhoods[-1])
        priors = np.array(priors)
        lhoods = np.array(lhoods)

        Ls = np.array(Ls)
        logging.debug('modelnames={}'.format(self.popset.modelnames))
        logging.debug('priors={}'.format(priors))
        logging.debug('lhoods={}'.format(lhoods))

        #colors = ['b','g','r','m','c']
        nmodels = len(self.popset.modelnames)
        colors = [cm.jet(1.*i/nmodels) for i in range(nmodels)]
        legendprop = {'size':8}

        ax1 = plt.axes([0.15,0.45,0.35,0.43])
        try:
            plt.pie(priors/priors.sum(),colors=colors)
            labels = []
            for i,model in enumerate(self.popset.modelnames):
                labels.append('%s: %.1e' % (model,priors[i]))
            plt.legend(labels,bbox_to_anchor=(-0.25,-0.1),loc='lower left',prop=legendprop)
            plt.title('Priors')
        except:
            msg = 'Error calculating priors.\n'
            for i,mod in enumerate(self.popset.shortmodelnames):
                msg += '%s: %.1e' % (model,priors[i])
            plt.annotate(msg, xy=(0.5,0.5), xycoords='axes fraction')


        ax2 = plt.axes([0.5,0.45,0.35,0.43])
        try:
            plt.pie(lhoods/lhoods.sum(),colors=colors)
            labels = []
            for i,model in enumerate(self.popset.modelnames):
                labels.append('%s: %.1e' % (model,lhoods[i]))
            plt.legend(labels,bbox_to_anchor=(1.25,-0.1),loc='lower right',prop=legendprop)
            plt.title('Likelihoods')
        except:
            msg = 'Error calculating lhoods.\n'
            for i,mod in enumerate(self.popset.shortmodelnames):
                msg += '%s: %.1e' % (model,lhoods[i])
            plt.annotate(msg, xy=(0.5,0.5), xycoords='axes fraction')

        ax3 = plt.axes([0.3,0.03,0.4,0.5])
        try:
            plt.pie(Ls/Ls.sum(),colors=colors)
            labels = []
            #for i,model in enumerate(['eb','heb','bgeb','bgpl','pl']):
            for i,model in enumerate(self.popset.modelnames):
                labels.append('%s: %.3f' % (model,Ls[i]/Ls.sum()))
            plt.legend(labels,bbox_to_anchor=(1.6,0.44),loc='right',prop={'size':10},shadow=True)
            plt.annotate('Final Probability',xy=(0.5,-0.01),ha='center',xycoords='axes fraction',fontsize=18)
        except:
            msg = 'Error calculating final probabilities.\n'
            plt.annotate(msg, xy=(0.5,0.5), xycoords='axes fraction')


        """
        #starpars = 'Star parameters used\nin simulations'
        starpars = ''
        if 'M' in self['heb'].stars.keywords and 'DM_P' in self.keywords:
            starpars += '\n$M/M_\odot = %.2f^{+%.2f}_{-%.2f}$' % (self['M'],self['DM_P'],self['DM_N'])
        else:
            starpars += '\n$(M/M_\odot = %.2f \pm %.2f)$' % (self['M'],0)  #this might not always be right?

        if 'DR_P' in self.keywords:
            starpars += '\n$R/R_\odot = %.2f^{+%.2f}_{-%.2f}$' % (self['R'],self['DR_P'],self['DR_N'])
        else:
            starpars += '\n$R/R_\odot = %.2f \pm %.2f$' % (self['R'],self['DR'])

        if 'FEH' in self.keywords:
            if 'DFEH_P' in self.keywords:
                starpars += '\n$[Fe/H] = %.2f^{+%.2f}_{-%.2f}$' % (self['FEH'],self['DFEH_P'],self['DFEH_N'])
            else:
                starpars += '\n$[Fe/H] = %.2f \pm %.2f$' % (self['FEH'],self['DFEH'])
        for kw in self.keywords:
            if re.search('-',kw):
                try:
                    starpars += '\n$%s = %.2f (%.2f)$ ' % (kw,self[kw],self['COLORTOL'])
                except TypeError:
                    starpars += '\n$%s = %s (%.2f)$ ' % (kw,self[kw],self['COLORTOL'])

        #if 'J-K' in self.keywords:
        #    starpars += '\n$J-K = %.2f (%.2f)$ ' % (self['J-K'],self['COLORTOL'])
        #if 'G-R' in self.keywords:
        #    starpars += '\n$g-r = %.2f (%.2f)$' % (self['G-R'],self['COLORTOL'])
        if starinfo:
            plt.annotate(starpars,xy=(0.03,0.91),xycoords='figure fraction',va='top')

        #p.annotate('Star',xy=(0.04,0.92),xycoords='figure fraction',va='top')

        priorpars = r'$f_{b,short} = %.2f$  $f_{trip} = %.2f$' % (self.priorfactors['fB']*self.priorfactors['f_Pshort'],
                                                                self.priorfactors['ftrip'])
        if 'ALPHA' in self.priorfactors:
            priorpars += '\n'+r'$f_{pl,bg} = %.2f$  $\alpha_{pl,bg} = %.1f$' % (self.priorfactors['fp'],self['ALPHA'])
        else:
            priorpars += '\n'+r'$f_{pl,bg} = %.2f$  $\alpha_1,\alpha_2,r_b = %.1f,%.1f,%.1f$' % \
                         (self.priorfactors['fp'],self['bgpl'].stars.keywords['ALPHA1'],
                          self['bgpl'].stars.keywords['ALPHA2'],
                          self['bgpl'].stars.keywords['RBREAK'])

        rbin1,rbin2 = self['RBINCEN']-self['RBINWID'],self['RBINCEN']+self['RBINWID']
        priorpars += '\n$f_{pl,specific} = %.2f, \in [%.2f,%.2f] R_\oplus$' % (self.priorfactors['fp_specific'],rbin1,rbin2)
        priorpars += '\n$r_{confusion} = %.1f$"' % sqrt(self.priorfactors['area']/pi)
        if self.priorfactors['multboost'] != 1:
            priorpars += '\nmultiplicity boost = %ix' % self.priorfactors['multboost']
        if priorinfo:
            plt.annotate(priorpars,xy=(0.03,0.4),xycoords='figure fraction',va='top')


        sigpars = ''
        sigpars += '\n$P = %s$ d' % self['P']
        depth,ddepth = self.trsig.depthfit
        sigpars += '\n$\delta = %i^{+%i}_{-%i}$ ppm' % (depth*1e6,ddepth[1]*1e6,ddepth[0]*1e6)
        dur,ddur = self.trsig.durfit
        sigpars += '\n$T = %.2f^{+%.2f}_{-%.2f}$ h' % (dur*24.,ddur[1]*24,ddur[0]*24)
        slope,dslope = self.trsig.slopefit
        sigpars += '\n'+r'$T/\tau = %.1f^{+%.1f}_{-%.1f}$' % (slope,dslope[1],dslope[0])
        sigpars += '\n'+r'$(T/\tau)_{max} = %.1f$' % (self.trsig.maxslope)
        if siginfo:
            plt.annotate(sigpars,xy=(0.81,0.91),xycoords='figure fraction',va='top')

            #p.annotate('${}^a$Not used for FP population simulations',xy=(0.02,0.02),
            #           xycoords='figure fraction',fontsize=9)
        """

        constraints = 'Constraints:'
        for c in self.popset.constraints:
            try:
                constraints += '\n  %s' % self['heb'].constraints[c]
            except KeyError:
                try:
                    constraints += '\n  %s' % self['beb'].constraints[c]
                except KeyError:
                    constraints += '\n  %s' % self['heb_px2'].constraints[c]
        if constraintinfo:
            plt.annotate(constraints,xy=(0.03,0.22),xycoords='figure fraction',
                         va='top',color='red')

        odds = 1./self.FPP()

        if odds > 1e6:
            fppstr = 'FPP: < 1 in 1e6'
        elif np.isfinite(odds):
            fppstr = 'FPP: 1 in %i' % odds
        else:
            fppstr = 'FPP calculation failed.'

        plt.annotate('$f_{pl,V} = %.3f$\n%s' % (self.fpV(),fppstr),xy=(0.7,0.02),
                     xycoords='figure fraction',fontsize=16,va='bottom')

        plt.suptitle(self.trsig.name,fontsize=22)

        #if not simple:
        #    plt.annotate('n = %.0e' % self.n,xy=(0.5,0.85),xycoords='figure fraction',
        #                 fontsize=14,ha='center')

        if saveplot:
            if tag is not None:
                plt.savefig('%s/FPPsummary_%s.%s' % (folder,tag,figformat))
            else:
                plt.savefig('%s/FPPsummary.%s' % (folder,figformat))
            plt.close()


    def lhoodplots(self,folder='.',tag=None,figformat='png',
                   recalc_lhood=False, **kwargs):
        """
        Make a plot of the likelihood for each model in PopulationSet

        """
        Ltot = 0

        for model in self.popset.modelnames:
            Ltot += self.prior(model)*self.lhood(model, recalc=recalc_lhood)

        for model in self.popset.shortmodelnames:
            if isinstance(self[model], ArtificialPopulation):
                continue
            self.lhoodplot(model,Ltot=Ltot,**kwargs)
            if tag is None:
                plt.savefig('%s/%s.%s' % (folder,model,figformat))
            else:
                plt.savefig('%s/%s_%s.%s' % (folder,model,figformat))
            plt.close()

    def lhoodplot(self,model,suptitle='',**kwargs):
        """
        Make a plot of the likelihood for a given model.
        """
        if suptitle=='':
            suptitle = self[model].model
        self[model].lhoodplot(self.trsig,colordict=self.popset.colordict,
                              suptitle=suptitle,**kwargs)

    def calc_lhoods(self,**kwargs):
        logging.debug('Calculating likelihoods...')
        for pop in self.popset.poplist:
            L = pop.lhood(self.trsig,**kwargs)
            logging.debug('%s: %.2e' % (pop.model,L))

    def __hash__(self):
        return hashcombine(self.popset, self.trsig)

    def __getitem__(self,model):
        return self.popset[model]

    def prior(self,model):
        """
        Return the prior for a given model.
        """
        return self[model].prior

    def lhood(self,model,**kwargs):
        """
        Return the likelihood for a given model.
        """
        return self[model].lhood(self.trsig,
                                 **kwargs)

    def Pval(self,skipmodels=None):
        Lfpp = 0
        if skipmodels is None:
            skipmodels = []
        logging.debug('evaluating likelihoods for %s' % self.trsig.name)

        for model in self.popset.modelnames:
            if model=='Planets':
                continue
            if model not in skipmodels:
                prior = self.prior(model)
                lhood = self.lhood(model)
                Lfpp += prior*lhood
                logging.debug('%s: %.2e = %.2e (prior) x %.2e (lhood)' % (model,prior*lhood,prior,lhood))
        prior = self.prior('pl')
        lhood = self.lhood('pl')
        Lpl = prior*lhood
        logging.debug('planet: %.2e = %.2e (prior) x %.2e (lhood)' % (prior*lhood,prior,lhood))
        return Lpl/Lfpp/self['pl'].priorfactors['fp_specific']

    def fpV(self,FPPV=0.005,skipmodels=None):
        P = self.Pval(skipmodels=skipmodels)
        return (1-FPPV)/(P*FPPV)

    def FPP(self,skipmodels=None):
        """
        Return the false positive probability (FPP)
        """
        Lfpp = 0
        if skipmodels is None:
            skipmodels = []
        logging.debug('evaluating likelihoods for %s' % self.trsig.name)
        for shortmodel, model in zip(self.popset.shortmodelnames, self.popset.modelnames):
            if model=='Planets':
                continue
            if (model not in skipmodels) and (shortmodel not in skipmodels):
                prior = self.prior(model)
                lhood = self.lhood(model)
                Lfpp += prior*lhood
                logging.debug('%s: %.2e = %.2e (prior) x %.2e (lhood)' % (model,prior*lhood,prior,lhood))
        prior = self.prior('pl')
        lhood = self.lhood('pl')
        Lpl = prior*lhood
        logging.debug('planet: %.2e = %.2e (prior) x %.2e (lhood)' % (prior*lhood,prior,lhood))
        return 1 - Lpl/(Lpl + Lfpp)

    def bootstrap_FPP(self, N=10, filename='results_bootstrap.txt'):
        lines = []
        for i in range(N):
            new = FPPCalculation(self.trsig, self.popset.resample(), folder=self.folder)
            h,l = new.write_results(to_file=False)
            lines.append(l)
        with open(os.path.join(self.folder, filename),'w') as fout:
            fout.write(h + '\n')
            for l in lines:
                fout.write(l + '\n')
        return h, lines
