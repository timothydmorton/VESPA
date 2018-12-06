from __future__ import print_function, division

import logging
import re, os, os.path

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats
    import numpy.random as rand

    from astropy import units as u
    from astropy.units import Quantity
    from astropy.coordinates import SkyCoord

else:
    #hacking...
    class np(object):
        inf = 1
    plt = None
    pd = None
    stats = None
    rand = None
    u = None
    Quantity = None
    SkyCoord = None

from ..orbits import OrbitPopulation
from ..orbits import TripleOrbitPopulation
from ..plotutils import setfig,plot2dhist

try:
    from isochrones.starmodel import StarModel, BinaryStarModel
    from isochrones.starmodel import TripleStarModel
    from isochrones import get_ichrone
except ImportError:
    StarModel, BinaryStarModel, TripleStarModel = (None, None, None)

try:
    from simpledist import distributions as dists
except ImportError:
    dists = None

from ..hashutils import hashcombine, hashdict, hashdf

from .constraints import Constraint,UpperLimit,LowerLimit,JointConstraintOr
from .constraints import ConstraintDict,MeasurementConstraint,RangeConstraint
from .contrastcurve import ContrastCurveConstraint,VelocityContrastCurveConstraint
from .contrastcurve import ContrastCurveFromFile

from .utils import randpos_in_circle, draw_raghavan_periods
from .utils import draw_msc_periods, draw_eccs
from .utils import flat_massratio, mult_masses
from .utils import distancemodulus, addmags, dfromdm

from .trilegal import get_trilegal

try:
    from isochrones import get_ichrone
except ImportError:
    logging.warning('isochrones package not installed; population simulations will not be fully functional')
    DARTMOUTH = None

BANDS = ['g','r','i','z','J','H','K','Kepler','TESS']

class StarPopulation(object):
    """A population of stars.

    This object contains information of a simulated population
    of stars.  It has a flexible purpose-- it could represent
    many random realizations of a single system, or it could
    also represent many different random systems.  This is the general
    base class; subclasses include, e.g., :class:`MultipleStarPopulation`
    and :class:`BGStarPopulation_TRILEGAL`.

    The :attr:`StarPopulation.stars` attribute is a
    :class:`pandas.DataFrame` containing
    all the information about all the random realizations, such
    as the physical star properties (mass, radius, etc.) and
    observational characteristics (magnitudes in different bands).

    The :attr:`StarPopulation.orbpop` attribute stores information
    about the orbits of the random stars, if such a thing is
    relevant for the population in question (such as, e.g., a
    :class:`MultipleStarPopulation`).  If orbits are relevant,
    then attributes such as :attr:`StarPopulation.Rsky`,
    :attr:`StarPopulation.RV`, and :func:`StarPopulation.dmag`
    are defined as well.

    Importantly, you can apply constraints to a :class:`StarPopulation`,
    implemented via the :class:`Constraint` class.  You can
    constrain properties of the stars to be within a given range,
    you can apply a :class:`ContrastCurveConstraint`, simulating
    the exclusion curve of an imaging observation, and many others.

    You can save and re-load :class:`StarPopulation` objects
    using :func:`StarPopulation.save_hdf` and
    :func:`StarPopulation.load_hdf`.

    .. warning::

        Support for saving constraints is planned and
        partially implemented but untested.

    Any subclass must be able to be initialized with no arguments,
    with no calculations being done; this enables the way that
    :func:`StarPopulation.load_hdf` is implemented to work properly.

    :param stars: (:class:`pandas.DataFrame`, optional)
        Table containing properties of stars.
        Magnitude properties end with "_mag".  Default
        is that these magnitudes are absolute, and get
        converted to apparent magnitudes based on distance,
        which is either provided or randomly assigned.

    :param distance:
        If ``None``, then distances of stars are assigned
        randomly out to max_distance, or by comparing to mags.
        If float, then assumed to be in parsec.  Or, if stars already
        has a distance column, this is ignored.
    :type distance:
        :class:`astropy.units.Quantity`, float, or array-like, optional

    :param max_distance: ``Quantity`` or float, optional
        Max distance out to which distances will be simulated,
        according to random placements in volume ($p(d)\simd^2$).
        Ignored if stars already has a distance column.
    :type max_distance:
        :class:`astropy.units.Quantity` or float, optional

    :param convert_absmags: (``bool``, optional)
        If ``True``, then magnitudes in ``stars`` will be converted
        to apparent magnitudes based on distance.  If ``False,``
        then magnitudes will be kept as-is.  Ignored if stars already
        has a distance column.

    :param orbpop:
        Describes the orbits of the stars.
    :type orbpop:
        :class:`orbits.OrbitPopulation`


    """
    def __init__(self,stars=None,distance=None,
                 max_distance=1000,convert_absmags=True,
                 name='', orbpop=None, mags=None):

        self.orbpop = orbpop
        self.name = name

        if stars is None:
            self.stars = None
        else:
            self.stars = stars.copy()
            N = len(self.stars)

            #if stars does not have a 'distance' column already, then
            # we define distances based on the provided arguments,
            # and covert absolute magnitudes into apparent (unless explicitly
            # forbidden from doing so by 'convert_absmags' being set
            # to False).

            if 'distance' not in self.stars:
                if type(max_distance) != Quantity:
                    max_distance = max_distance * u.pc


                if distance is None:
                    #generate random distances
                    dmax = max_distance.to('pc').value
                    #distance_distribution = dists.PowerLaw_Distribution(2.,1,dmax) # p(d)~d^2
                    #distance = distance_distribution.rvs(N)

                    # p(d) ~ d^2
                    distance = stats.powerlaw(3).rvs(N) * dmax

                if type(distance) != Quantity:
                    distance = distance * u.pc

                distmods = distancemodulus(distance)
                if convert_absmags:
                    for col in self.stars.columns:
                        if re.search('_mag',col):
                            self.stars[col] += distmods

                self.stars['distance'] = distance
                self.stars['distmod'] = distmods


            if 'distmod' not in self.stars:
                self.stars['distmod'] = distancemodulus(self.stars['distance'])


    @property
    def starmodel(self):
        if hasattr(self, '_starmodel'):
            return self._starmodel
        else:
            return AttributeError('No starmodel for this object.')

    @property
    def Rsky(self):
        """
        Projected angular distance between "primary" and "secondary" (exact meaning varies)

        """
        r = (self.orbpop.Rsky/self.distance)
        return r.to('arcsec',equivalencies=u.dimensionless_angles())

    @property
    def RV(self):
        """
        Radial velocity difference between "primary" and "secondary" (exact meaning varies)
        """
        return self.orbpop.RV

    def dRV(self,dt):
        """
        Change in RV between two epochs separated by dt

        :param dt:
            Time difference between two epochs, either :class:`astropy.units.Quantity`
            or days.

        :return:
            Change in RV.
        """
        return self.orbpop.dRV(dt)

    def dmag(self, band):
        """
        Magnitude difference between "primary" and "secondary" in given band

        Exact definition will depend on context.  Only legit if ``self.mags``
        is defined (i.e., not ``None``).

        :param band: (``string``)
            Desired photometric bandpass.
        """
        if self.mags is None:
            raise ValueError('This population does not have a "mags" attribute ' +
                             'defined; dmags is meaningless.')
        return self.stars['{}_mag'.format(band)] - self.mags[band]


    def append(self, other):
        """Appends stars from another StarPopulations, in place.

        :param other:
            Another :class:`StarPopulation`; must have same columns as ``self``.

        """
        if not isinstance(other,StarPopulation):
            raise TypeError('Only StarPopulation objects can be appended to a StarPopulation.')
        if not np.all(self.stars.columns == other.stars.columns):
            raise ValueError('Two populations must have same columns to combine them.')

        if len(self.constraints) > 0:
            logging.warning('All constraints are cleared when appending another population.')

        self.stars = pd.concat((self.stars, other.stars))

        if self.orbpop is not None and other.orbpop is not None:
            self.orbpop = self.orbpop + other.orbpop


    def __getitem__(self,prop):
        return self.selected[prop]

    def __hash__(self):
        return hashcombine(self.constraints,
                           hashdf(self.stars), self.orbpop)

    def __eq__(self, other):
        return hash(self)==hash(other)

    def generate(self, *args, **kwargs):
        """
        Function that generates population.
        """
        raise NotImplementedError

    @property
    def is_ruled_out(self):
        """
        Will be ``True`` if contraints rule out all (or all but one) instances
        """
        if hasattr(self,'is_empty'):
            return self.is_empty
        else:
            return self.distok.sum() < 2

    @property
    def bands(self):
        """
        Bandpasses for which StarPopulation has magnitude data
        """
        bands = []
        for c in self.stars.columns:
            if re.search('_mag',c):
                bands.append(c)
        return bands

    @property
    def distance(self):
        """Distance to stars.

        """
        return np.array(self.stars['distance'])*u.pc

    @distance.setter
    def distance(self,value):
        """New distance value must be a ``Quantity`` object
        """
        self.stars['distance'] = value.to('pc').value

        old_distmod = self.stars['distmod'].copy()
        new_distmod = distancemodulus(self.stars['distance'])

        for m in self.bands:
            self.stars[m] += new_distmod - old_distmod

        self.stars['distmod'] = new_distmod

        logging.warning('Setting the distance manually may have screwed up your constraints.  Re-apply constraints as necessary.')



    @property
    def distok(self):
        """
        Boolean array showing which stars pass all distribution constraints.

        A "distribution constraint" is a constraint that affects the
        distribution of stars, rather than just the number.
        """
        ok = np.ones(len(self.stars)).astype(bool)
        for name in self.constraints:
            c = self.constraints[name]
            if c.name not in self.distribution_skip:
                ok &= c.ok
        return ok

    @property
    def countok(self):
        """
        Boolean array showing which stars pass all count constraints.

        A "count constraint" is a constraint that affects the number of stars.
        """
        ok = np.ones(len(self.stars)).astype(bool)
        for name in self.constraints:
            c = self.constraints[name]
            if c.name not in self.selectfrac_skip:
                ok &= c.ok
        return ok

    @property
    def selected(self):
        """
        All stars that pass all distribution constraints.

        """
        return self.stars[self.distok]

    @property
    def selectfrac(self):
        """
        Fraction of stars that pass count constraints.
        """
        return self.countok.sum()/len(self.stars)

    def prophist2d(self,propx,propy, mask=None,
                   logx=False,logy=False,
                   fig=None,selected=False,**kwargs):
        """Makes a 2d density histogram of two given properties

        :param propx,propy:
            Names of properties to histogram.  Must be names of columns
            in ``self.stars`` table.

        :param mask: (optional)
            Boolean mask (``True`` is good) to say which indices to plot.
            Must be same length as ``self.stars``.

        :param logx,logy: (optional)
            Whether to plot the log10 of x and/or y properties.

        :param fig: (optional)
            Argument passed to :func:`plotutils.setfig`.

        :param selected: (optional)
            If ``True``, then only the "selected" stars (that is, stars
            obeying all distribution constraints attached to this object)
            will be plotted.  In this case, ``mask`` will be ignored.

        :param kwargs:
            Additional keyword arguments passed to :func:`plotutils.plot2dhist`.

        """

        if mask is not None:
            inds = np.where(mask)[0]
        else:
            if selected:
                inds = self.selected.index
            else:
                inds = self.stars.index

        if selected:
            xvals = self.selected[propx].iloc[inds].values
            yvals = self.selected[propy].iloc[inds].values
        else:
            if mask is None:
                mask = np.ones_like(self.stars.index)
            xvals = self.stars[mask][propx].values
            yvals = self.stars[mask][propy].values

        #forward-hack for EclipsePopulations...
        #TODO: reorganize.
        if propx=='depth' and hasattr(self,'depth'):
            xvals = self.depth.iloc[inds].values
        if propy=='depth' and hasattr(self,'depth'):
            yvals = self.depth.iloc[inds].values

        if logx:
            xvals = np.log10(xvals)
        if logy:
            yvals = np.log10(yvals)

        plot2dhist(xvals,yvals,fig=fig,**kwargs)
        plt.xlabel(propx)
        plt.ylabel(propy)


    def prophist(self,prop,fig=None,log=False, mask=None,
                 selected=False,**kwargs):
        """Plots a 1-d histogram of desired property.

        :param prop:
            Name of property to plot.  Must be column of ``self.stars``.

        :param fig: (optional)
            Argument for :func:`plotutils.setfig`

        :param log: (optional)
            Whether to plot the histogram of log10 of the property.

        :param mask: (optional)
            Boolean array (length of ``self.stars``) to say
            which indices to plot (``True`` is good).

        :param selected: (optional)
            If ``True``, then only the "selected" stars (that is, stars
            obeying all distribution constraints attached to this object)
            will be plotted.  In this case, ``mask`` will be ignored.

        :param **kwargs:
            Additional keyword arguments passed to :func:`plt.hist`.

        """

        setfig(fig)

        inds = None
        if mask is not None:
            inds = np.where(mask)[0]
        elif inds is None:
            if selected:
                #inds = np.arange(len(self.selected))
                inds = self.selected.index
            else:
                #inds = np.arange(len(self.stars))
                inds = self.stars.index

        if selected:
            vals = self.selected[prop].values#.iloc[inds] #invalidates mask?
        else:
            vals = self.stars[prop].iloc[inds].values

        if prop=='depth' and hasattr(self,'depth'):
            vals *= self.dilution_factor[inds]

        if log:
            h = plt.hist(np.log10(vals),**kwargs)
        else:
            h = plt.hist(vals,**kwargs)

        plt.xlabel(prop)

    def constraint_stats(self,primarylist=None):
        """Returns information about effect of constraints on population.

        :param primarylist:
           List of constraint names that you want specific information on
           (i.e., not blended within "multiple constraints".)

        :return:
           ``dict`` of what percentage of population is ruled out by
           each  constraint, including a "multiple constraints" entry.
        """
        if primarylist is None:
            primarylist = []
        n = len(self.stars)
        primaryOK = np.ones(n).astype(bool)
        tot_reject = np.zeros(n)

        for name in self.constraints:
            if name in self.selectfrac_skip:
                continue
            c = self.constraints[name]
            if name in primarylist:
                primaryOK &= c.ok
            tot_reject += ~c.ok
        primary_rejected = ~primaryOK
        secondary_rejected = tot_reject - primary_rejected
        lone_reject = {}
        for name in self.constraints:
            if name in primarylist or name in self.selectfrac_skip:
                continue
            c = self.constraints[name]
            lone_reject[name] = ((secondary_rejected==1) & (~primary_rejected) & (~c.ok)).sum()/float(n)
        mult_rejected = (secondary_rejected > 1) & (~primary_rejected)
        not_rejected = ~(tot_reject.astype(bool))
        primary_reject_pct = primary_rejected.sum()/float(n)
        mult_reject_pct = mult_rejected.sum()/float(n)
        not_reject_pct = not_rejected.sum()/float(n)
        tot = 0

        results = {}
        results['pri'] = primary_reject_pct
        tot += primary_reject_pct
        for name in lone_reject:
            results[name] = lone_reject[name]
            tot += lone_reject[name]
        results['multiple constraints'] = mult_reject_pct
        tot += mult_reject_pct
        results['remaining'] = not_reject_pct
        tot += not_reject_pct

        if tot != 1:
            logging.warning('total adds up to: %.2f (%s)' % (tot,self.model))

        return results


    def constraint_piechart(self,primarylist=None,
                            fig=None,title='',colordict=None,
                            legend=True,nolabels=False):
        """Makes piechart illustrating constraints on population

        :param primarylist: (optional)
            List of most import constraints to show (see
            :func:`StarPopulation.constraint_stats`)

        :param fig: (optional)
            Passed to :func:`plotutils.setfig`.

        :param title: (optional)
            Title for pie chart

        :param colordict: (optional)
            Dictionary describing colors (keys are constraint names).

        :param legend: (optional)
            ``bool`` indicating whether to display a legend.

        :param nolabels: (optional)
            If ``True``, then leave out legend labels.

        """

        setfig(fig,figsize=(6,6))
        stats = self.constraint_stats(primarylist=primarylist)
        if primarylist is None:
            primarylist = []
        if len(primarylist)==1:
            primaryname = primarylist[0]
        else:
            primaryname = ''
            for name in primarylist:
                primaryname += '%s,' % name
            primaryname = primaryname[:-1]
        fracs = []
        labels = []
        explode = []
        colors = []
        fracs.append(stats['remaining']*100)
        labels.append('remaining')
        explode.append(0.05)
        colors.append('b')
        if 'pri' in stats and stats['pri']>=0.005:
            fracs.append(stats['pri']*100)
            labels.append(primaryname)
            explode.append(0)
            if colordict is not None:
                colors.append(colordict[primaryname])
        for name in stats:
            if name == 'pri' or \
                    name == 'multiple constraints' or \
                    name == 'remaining':
                continue

            fracs.append(stats[name]*100)
            labels.append(name)
            explode.append(0)
            if colordict is not None:
                colors.append(colordict[name])

        if stats['multiple constraints'] >= 0.005:
            fracs.append(stats['multiple constraints']*100)
            labels.append('multiple constraints')
            explode.append(0)
            colors.append('w')

        autopct = '%1.1f%%'

        if nolabels:
            labels = None
        if legend:
            legendlabels = []
            for i,l in enumerate(labels):
                legendlabels.append('%s (%.1f%%)' % (l,fracs[i]))
            labels = None
            autopct = ''
        if colordict is None:
            plt.pie(fracs,labels=labels,autopct=autopct,explode=explode)
        else:
            plt.pie(fracs,labels=labels,autopct=autopct,explode=explode,
                    colors=colors)
        if legend:
            plt.legend(legendlabels,bbox_to_anchor=(-0.05,0),
                       loc='lower left',prop={'size':10})
        plt.title(title)

    @property
    def selectfrac_skip(self):
        """
        Names of constraints that should *not* be considered for counting purposes
        """
        try:
            return self._selectfrac_skip
        except AttributeError:
            self._selectfrac_skip = []
            return self._selectfrac_skip

    @selectfrac_skip.setter
    def selectfrac_skip(self, value):
        self._selectfrac_skip = value

    @property
    def distribution_skip(self):
        """
        Names of constraints that should *not* be considered for distribution purposes
        """
        try:
            return self._distribution_skip
        except AttributeError:
            self._distribution_skip = []
            return self._distribution_skip

    @distribution_skip.setter
    def distribution_skip(self, value):
        self._distribution_skip = value

    @property
    def constraints(self):
        """
        Constraints applied to the population.
        """
        try:
            return self._constraints
        except AttributeError:
            self._constraints = ConstraintDict()
            return self._constraints

    @constraints.setter
    def constraints(self, value):
        self._constraints = value

    @property
    def hidden_constraints(self):
        """
        Constraints applied to the population, but temporarily removed.
        """
        try:
            return self._hidden_constraints
        except AttributeError:
            self._hidden_constraints = ConstraintDict()
            return self._hidden_constraints

    @hidden_constraints.setter
    def hidden_constraints(self, value):
        self._hidden_constraints = value

    def apply_constraint(self,constraint,selectfrac_skip=False,
                         distribution_skip=False,overwrite=False):
        """Apply a constraint to the population

        :param constraint:
            Constraint to apply.
        :type constraint:
            :class:`Constraint`

        :param selectfrac_skip: (optional)
            If ``True``, then this constraint will not be considered
            towards diminishing the
        """
        #grab properties
        constraints = self.constraints
        my_selectfrac_skip = self.selectfrac_skip
        my_distribution_skip = self.distribution_skip

        if constraint.name in constraints and not overwrite:
            logging.warning('constraint already applied: {}'.format(constraint.name))
            return
        constraints[constraint.name] = constraint
        if selectfrac_skip:
            my_selectfrac_skip.append(constraint.name)
        if distribution_skip:
            my_distribution_skip.append(constraint.name)

        #forward-looking for EclipsePopulation
        if hasattr(self, '_make_kde'):
            self._make_kde()

        self.constraints = constraints
        self.selectfrac_skip = my_selectfrac_skip
        self.distribution_skip = my_distribution_skip

        #self._apply_all_constraints()

    def replace_constraint(self,name,selectfrac_skip=False,distribution_skip=False):
        """
        Re-apply constraint that had been removed

        :param name:
            Name of constraint to replace

        :param selectfrac_skip,distribution_skip: (optional)
            Same as :func:`StarPopulation.apply_constraint`

        """
        hidden_constraints = self.hidden_constraints
        if name in hidden_constraints:
            c = hidden_constraints[name]
            self.apply_constraint(c,selectfrac_skip=selectfrac_skip,
                                  distribution_skip=distribution_skip)
            del hidden_constraints[name]
        else:
            logging.warning('Constraint {} not available for replacement.'.format(name))

        self.hidden_constraints = hidden_constraints

    def remove_constraint(self,name):
        """
        Remove a constraint (make it "hidden")

        :param name:
            Name of constraint.
        """
        constraints = self.constraints
        hidden_constraints = self.hidden_constraints
        my_distribution_skip = self.distribution_skip
        my_selectfrac_skip = self.selectfrac_skip

        if name in constraints:
            hidden_constraints[name] = constraints[name]
            del constraints[name]
            if name in self.distribution_skip:
                my_distribution_skip.remove(name)
            if name in self.selectfrac_skip:
                my_selectfrac_skip.remove(name)
            #self._apply_all_constraints()
        else:
            logging.warning('Constraint {} does not exist.'.format(name))

        self.constraints = constraints
        self.hidden_constraints = hidden_constraints
        self.selectfrac_skip = my_selectfrac_skip
        self.distribution_skip = my_distribution_skip

    def constrain_property(self,prop,lo=-np.inf,hi=np.inf,
                           measurement=None,thresh=3,
                           selectfrac_skip=False,distribution_skip=False):
        """Apply constraint that constrains property.

        :param prop:
            Name of property.  Must be column in ``self.stars``.
        :type prop:
            ``str``

        :param lo,hi: (optional)
            Low and high allowed values for ``prop``.  Defaults
            to ``-np.inf`` and ``np.inf`` to allow for defining
            only lower or upper limits if desired.

        :param measurement: (optional)
            Value and error of measurement in form ``(value, error)``.

        :param thresh: (optional)
            Number of "sigma" to allow for measurement constraint.

        :param selectfrac_skip,distribution_skip:
            Passed to :func:`StarPopulation.apply_constraint`.

        """
        if prop in self.constraints:
            logging.info('re-doing {} constraint'.format(prop))
            self.remove_constraint(prop)
        if measurement is not None:
            val,dval = measurement
            self.apply_constraint(MeasurementConstraint(getattr(self.stars,prop),
                                                        val,dval,name=prop,
                                                        thresh=thresh),
                                  selectfrac_skip=selectfrac_skip,
                                  distribution_skip=distribution_skip)
        else:
            self.apply_constraint(RangeConstraint(getattr(self.stars,prop),
                                                  lo=lo,hi=hi,name=prop),
                                  selectfrac_skip=selectfrac_skip,
                                  distribution_skip=distribution_skip)

    def apply_trend_constraint(self, limit, dt, distribution_skip=False,
                               **kwargs):
        """
        Constrains change in RV to be less than limit over time dt.

        Only works if ``dRV`` and ``Plong`` attributes are defined
        for population.

        :param limit:
            Radial velocity limit on trend.  Must be
            :class:`astropy.units.Quantity` object, or
            else interpreted as m/s.

        :param dt:
            Time baseline of RV observations.  Must be
            :class:`astropy.units.Quantity` object; else
            interpreted as days.

        :param distribution_skip:
            This is by default ``True``.  *To be honest, I'm not
            exactly sure why.  Might be important, might not
            (don't remember).*

        :param **kwargs:
            Additional keyword arguments passed to
            :func:`StarPopulation.apply_constraint`.

        """
        if type(limit) != Quantity:
            limit = limit * u.m/u.s
        if type(dt) != Quantity:
            dt = dt * u.day

        dRVs = np.absolute(self.dRV(dt))
        c1 = UpperLimit(dRVs, limit)
        c2 = LowerLimit(self.Plong, dt*4)

        self.apply_constraint(JointConstraintOr(c1,c2,name='RV monitoring',
                                                Ps=self.Plong,dRVs=dRVs),
                              distribution_skip=distribution_skip, **kwargs)

    def apply_cc(self, cc, distribution_skip=False,
                 **kwargs):
        """
        Apply contrast-curve constraint to population.

        Only works if object has ``Rsky``, ``dmag`` attributes

        :param cc:
            Contrast curve.
        :type cc:
            :class:`ContrastCurveConstraint`

        :param distribution_skip:
            This is by default ``True``.  *To be honest, I'm not
            exactly sure why.  Might be important, might not
            (don't remember).*

        :param **kwargs:
            Additional keyword arguments passed to
            :func:`StarPopulation.apply_constraint`.

        """
        rs = self.Rsky.to('arcsec').value
        dmags = self.dmag(cc.band)
        self.apply_constraint(ContrastCurveConstraint(rs,dmags,cc,name=cc.name),
                              distribution_skip=distribution_skip, **kwargs)

    def apply_vcc(self, vcc, distribution_skip=False,
                  **kwargs):
        """
        Applies "velocity contrast curve" to population.

        That is, the constraint that comes from not seeing two sets
        of spectral lines in a high resolution spectrum.

        Only works if population has ``dmag`` and ``RV`` attributes.

        :param vcc:
            Velocity contrast curve; dmag vs. delta-RV.
        :type cc:
            :class:`VelocityContrastCurveConstraint`

        :param distribution_skip:
            This is by default ``True``.  *To be honest, I'm not
            exactly sure why.  Might be important, might not
            (don't remember).*

        :param **kwargs:
            Additional keyword arguments passed to
            :func:`StarPopulation.apply_constraint`.

        """
        rvs = self.RV.value
        dmags = self.dmag(vcc.band)
        self.apply_constraint(VelocityContrastCurveConstraint(rvs,dmags,vcc,
                                                              name='secondary spectrum'),
                              distribution_skip=distribution_skip, **kwargs)

    def set_maxrad(self,maxrad, distribution_skip=True):
        """
        Adds a constraint that rejects everything with Rsky > maxrad

        Requires ``Rsky`` attribute, which should always have units.

        :param maxrad:
            The maximum angular value of Rsky.
        :type maxrad:
            :class:`astropy.units.Quantity`

        :param distribution_skip:
            This is by default ``True``.  *To be honest, I'm not
            exactly sure why.  Might be important, might not
            (don't remember).*

        """
        self.maxrad = maxrad
        self.apply_constraint(UpperLimit(self.Rsky,maxrad,
                                         name='Max Rsky'),
                              overwrite=True,
                              distribution_skip=distribution_skip)
        #self._apply_all_constraints()


    @property
    def constraint_df(self):
        """
        A DataFrame representing all constraints, hidden or not
        """
        df = pd.DataFrame()
        for name,c in self.constraints.items():
            df[name] = c.ok
        for name,c in self.hidden_constraints.items():
            df[name] = c.ok
        return df

    @property
    def _properties(self):
        return ['name']

    def save_hdf(self,filename,path='',properties=None,
                 overwrite=False, append=False):
        """Saves to HDF5 file.

        Subclasses should be sure to define
        ``_properties`` attribute to ensure that all
        correct attributes get saved.  Load a saved population
        with :func:`StarPopulation.load_hdf`.

        Example usage::

            >>> from vespa.stars import Raghavan_BinaryPopulation, StarPopulation
            >>> pop = Raghavan_BinaryPopulation(1., n=1000)
            >>> pop.save_hdf('test.h5')
            >>> pop2 = StarPopulation.load_hdf('test.h5')
            >>> pop == pop2
                True
            >>> pop3 = Ragahavan_BinaryPopulation.load_hdf('test.h5')
            >>> pop3 == pop2
                True


        :param filename:
            Name of HDF file.

        :param path: (optional)
            Path within HDF file to save object.

        :param properties: (optional)
            Names of any properties (in addition to
            those defined in ``_properties`` attribute)
            that you wish to save.  (This is an old
            keyword, and should probably be removed.
            Feel free to ignore it.)

        :param overwrite: (optional)
            Whether to overwrite file if it already
            exists.  If ``True``, then any existing file
            will be deleted before object is saved.  Use
            ``append`` if you don't wish this to happen.

        :param append: (optional)
            If ``True``, then if the file exists,
            then only the particular path in the file
            will get written/overwritten.  If ``False`` and both
            file and path exist, then an ``IOError`` will
            be raised.  If ``False`` and file exists but not
            path, then no error will be raised.

        """
        if os.path.exists(filename):
            with pd.HDFStore(filename) as store:
                if path in store:
                    if overwrite:
                        os.remove(filename)
                    elif not append:
                        raise IOError('{} in {} exists.  '.format(path,filename) +
                                     'Set either overwrite or append option.')

        if properties is None:
            properties = {}

        for prop in self._properties:
            properties[prop] = getattr(self, prop)

        self.stars.to_hdf(filename,'{}/stars'.format(path))
        self.constraint_df.to_hdf(filename,'{}/constraints'.format(path))

        if self.orbpop is not None:
            self.orbpop.save_hdf(filename, path=path+'/orbpop')

        with pd.HDFStore(filename) as store:
            attrs = store.get_storer('{}/stars'.format(path)).attrs
            attrs.selectfrac_skip = self.selectfrac_skip
            attrs.distribution_skip = self.distribution_skip
            attrs.name = self.name
            attrs.poptype = type(self)
            attrs.properties = properties

    @classmethod
    def load_hdf(cls, filename, path=''):
        """Loads StarPopulation from .h5 file

        Correct properties should be restored to object, and object
        will be original type that was saved.  Complement to
        :func:`StarPopulation.save_hdf`.

        Example usage::

            >>> from vespa.stars import Raghavan_BinaryPopulation, StarPopulation
            >>> pop = Raghavan_BinaryPopulation(1., n=1000)
            >>> pop.save_hdf('test.h5')
            >>> pop2 = StarPopulation.load_hdf('test.h5')
            >>> pop == pop2
                True
            >>> pop3 = Ragahavan_BinaryPopulation.load_hdf('test.h5')
            >>> pop3 == pop2
                True


        :param filename:
            HDF file with saved :class:`StarPopulation`.

        :param path:
            Path within HDF file.

        :return:
            :class:`StarPopulation` or appropriate subclass; whatever
            was saved with :func:`StarPopulation.save_hdf`.

        """
        stars = pd.read_hdf(filename,path+'/stars')
        constraint_df = pd.read_hdf(filename,path+'/constraints')

        with pd.HDFStore(filename) as store:
            has_orbpop = '{}/orbpop/df'.format(path) in store
            has_triple_orbpop = '{}/orbpop/long/df'.format(path) in store
            attrs = store.get_storer('{}/stars'.format(path)).attrs

            poptype = attrs.poptype
            new = poptype()

            #if poptype != type(self):
            #    raise TypeError('Saved population is {}.  Please instantiate proper class before loading.'.format(poptype))


            distribution_skip = attrs.distribution_skip
            selectfrac_skip = attrs.selectfrac_skip
            name = attrs.name

            for kw,val in attrs.properties.items():
                setattr(new, kw, val)

        #load orbpop if there
        orbpop = None
        if has_orbpop:
            orbpop = OrbitPopulation.load_hdf(filename, path=path+'/orbpop')
        elif has_triple_orbpop:
            orbpop = TripleOrbitPopulation.load_hdf(filename, path=path+'/orbpop')

        new.stars = stars
        new.orbpop = orbpop


        for n in constraint_df.columns:
            mask = np.array(constraint_df[n])
            c = Constraint(mask,name=n)
            sel_skip = n in selectfrac_skip
            dist_skip = n in distribution_skip
            new.apply_constraint(c,selectfrac_skip=sel_skip,
                                  distribution_skip=dist_skip)

        return new

class BinaryPopulation(StarPopulation):
    """A population of binary stars.

    If :class:`vespa.orbits.OrbitPopulation` provided via ``orbpop`` keyword,
    that will describe the orbits;
    if not, then orbit population will be generated.  Single stars may
    be indicated if desired by having their mass set to zero and all
    magnitudes set to ``inf``.

    This will usually be used via, e.g., the
    :class:`Raghavan_BinaryPopulation` subclass, rather than
    instantiated directly.

    :param primary,secondary: (:class:`pandas.DataFrame`)
        Properties of primary and secondary stars, respectively.
        These get merged into new ``stars`` attribute, with "_A"
        and "_B" tags.

    :param orbpop: (:class:`vespa.orbits.OrbitPopulation`, optional)
        Object describing orbits of stars.  If not provided, then ``period``
        and ``ecc`` keywords must be provided, or else they will be
        randomly generated (see below).

    :param period,ecc:
        Periods and eccentricities of orbits.  If ``orbpop``
        not passed, and these are not provided, then periods and eccs
        will be randomly generated according
        to the empirical distributions of the Raghavan (2010) and
        Multiple Star Catalog distributions using
        :func:`utils.draw_raghavan_periods` and
        :func:`utils.draw_eccs`.

    """
    def __init__(self, stars=None,
                 primary=None,secondary=None,
                 orbpop=None, period=None,
                 ecc=None,
                 is_single=None,
                 **kwargs):


        if stars is None and primary is not None:
            assert len(primary)==len(secondary)

            stars = pd.DataFrame()

            for c in primary.columns:
                if re.search('_mag',c):
                    stars[c] = addmags(primary[c],secondary[c])
                stars['{}_A'.format(c)] = primary[c]
            for c in secondary.columns:
                stars['{}_B'.format(c)] = secondary[c]

            stars['q'] = stars['mass_B']/stars['mass_A']


            if orbpop is None:
                if period is None:
                    period = draw_raghavan_periods(len(secondary))
                if ecc is None:
                    ecc = draw_eccs(len(secondary),period)
                orbpop = OrbitPopulation(primary['mass'],
                                         secondary['mass'],
                                         period,ecc)

        StarPopulation.__init__(self,stars=stars,orbpop=orbpop,**kwargs)


    @property
    def singles(self):
        """
        Subset of stars that are single.
        """
        return self.stars.query('mass_B == 0')

    @property
    def binaries(self):
        """
        Subset of stars that are binaries.
        """
        return self.stars.query('mass_B > 0')

    def binary_fraction(self,query='mass_A >= 0'):
        """
        Binary fraction of stars passing given query

        :param query:
            Query to pass to stars ``DataFrame``.

        """
        subdf = self.stars.query(query)
        nbinaries = (subdf['mass_B'] > 0).sum()
        frac = nbinaries/len(subdf)
        return frac, frac/np.sqrt(nbinaries)

    @property
    def Plong(self):
        """ Orbital period.

        Called "Plong" to be consistent with hierarchical
        populations that have this attribute mean the
        longer of two periods.

        """
        return self.orbpop.P

    def dmag(self,band):
        """
        Difference in magnitude between primary and secondary stars

        :param band:
            Photometric bandpass.

        """
        mag2 = self.stars['{}_mag_B'.format(band)]
        mag1 = self.stars['{}_mag_A'.format(band)]
        return mag2-mag1

    def rsky_distribution(self,rmax=None,smooth=0.1,nbins=100):
        """
        Distribution of projected separations

        Returns a :class:`simpledists.Hist_Distribution` object.

        :param rmax: (optional)
            Maximum radius to calculate distribution.

        :param dr: (optional)
            Bin width for histogram

        :param smooth: (optional)
            Smoothing parameter for :class:`simpledists.Hist_Distribution`

        :param nbins: (optional)
            Number of bins for histogram

        :return:
            :class:`simpledists.Hist_Distribution` describing Rsky distribution

        """
        if rmax is None:
            if hasattr(self,'maxrad'):
                rmax = self.maxrad
            else:
                rmax = np.percentile(self.Rsky,99)
        dist = dists.Hist_Distribution(self.Rsky.value,bins=nbins,maxval=rmax,smooth=smooth)
        return dist

    def rsky_lhood(self,rsky,**kwargs):
        """
        Evaluates Rsky likelihood at provided position(s)

        :param rsky:
            position

        :param **kwargs:
            Keyword arguments passed to :func:`BinaryPopulation.rsky_distribution`

        """
        dist = self.rsky_distribution(**kwargs)
        return dist(rsky)


class Simulated_BinaryPopulation(BinaryPopulation):
    """Simulates BinaryPopulation according to provide primary mass(es), generating functions, and stellar isochrone models.


    :param M:
        Primary mass(es).
    :type M:
        ``float`` or array-like

    :param q_fn: (optional)
        Mass ratio generating function. Must return 'n' mass ratios, and be
        called as follows::

            qs = q_fn(n)

    :type q_fn:
        Callable function.

    :param P_fn: (optional)
        Orbital period generating function.  Must return ``n`` orbital periods,
        and be called as follows::

            Ps = P_fn(n)

    :type P_fn:
        Callable function.

    :param ecc_fn: (optional)
        Orbital eccentricity generating function.  Must return ``n`` orbital
        eccentricities generated according to provided period(s)::

            eccs = ecc_fn(n,Ps)

    :type ecc_fn:
        Callable function.

    :param n: (optional)
        Number of instances to simulate.

    :param ichrone: (optional)
        Stellar model object from which to simulate stellar properties.
        Default is the default Dartmouth isochrone.
    :type ichrone:
        :class:`isochrones.Isochrone`

    :param bands: (optional)
        Photometric bands to simulate via ``ichrone``.

    :param age,feh: (optional)
        log(age) and metallicity at which to simulate population.
        Can be ``float`` or array-like

    :param minmass: (optional)
        Minimum mass to simulate.  Default = 0.12.

    """
    def __init__(self,M=None,q_fn=None,P_fn=None,ecc_fn=None,
                 n=1e4,ichrone='mist', qmin=0.1, bands=BANDS,
                 age=9.6,feh=0.0, minmass=0.12, **kwargs):

        if q_fn is None:
            q_fn = flat_massratio
        self.q_fn = q_fn
        self.qmin = qmin
        self.P_fn = P_fn
        self.ecc_fn = ecc_fn
        self.minmass = minmass

        if M is None:
            BinaryPopulation.__init__(self) #empty
        else:
            self.generate(M, age=age, feh=feh, ichrone=ichrone,
                          n=n, bands=bands, **kwargs)

    def generate(self, M, age=9.6, feh=0.0,
                 ichrone='mist', n=1e4, bands=None, **kwargs):
        """
        Function that generates population.

        Called by ``__init__`` if ``M`` is passed.
        """
        ichrone = get_ichrone(ichrone, bands=bands)
        if np.size(M) > 1:
            n = np.size(M)
        else:
            n = int(n)
        M2 = M * self.q_fn(n, qmin=np.maximum(self.qmin,self.minmass/M))

        P = self.P_fn(n)
        ecc = self.ecc_fn(n,P)

        mass = np.ascontiguousarray(np.ones(n)*M)
        mass2 = np.ascontiguousarray(M2)
        age = np.ascontiguousarray(age)
        feh = np.ascontiguousarray(feh)
        pri = ichrone(mass, age, feh, return_df=True, bands=bands)
        sec = ichrone(mass2, age, feh, return_df=True, bands=bands)

        BinaryPopulation.__init__(self, primary=pri, secondary=sec,
                                  period=P, ecc=ecc, **kwargs)
        return self

    @property
    def _properties(self):
        return ['q_fn', 'qmin', 'P_fn', 'ecc_fn', 'minmass'] +\
            super(Simulated_BinaryPopulation, self)._properties


class Raghavan_BinaryPopulation(Simulated_BinaryPopulation):
    """A Simulated_BinaryPopulation with empirical default distributions.

    Default mass ratio distribution is flat down to chosen minimum mass,
    default period distribution is from Raghavan (2010), default
    eccentricity/period relation comes from data from the Multiple Star
    Catalog (Tokovinin, xxxx).

    :param M:
        Primary mass(es) in solar masses.

    :param e_M: (optional)
        1-sigma uncertainty in primary mass.

    :param n: (optional)
        Number of simulated instances to create.

    :param ichrone: (optional)
        Stellar models from which to generate binary companions.
    :type ichrone:
        :class:`isochrones.Isochrone`

    :param age,feh: (optional)
        Age and metallicity of system.

    :param name: (optional)
        Name of population.

    :param q_fn: (optional)
        A function that returns random mass ratios.  Defaults to flat
        down to provided minimum mass.  Must be able to be called as
        follows::

            qs = q_fn(n, qmin, qmax)

        to provide ``n`` random mass ratios.



    """
    def __init__(self,M=None,e_M=0,n=1e4,ichrone='mist',
                 age=9.5, feh=0.0, q_fn=None, qmin=0.1,
                 minmass=0.12, **kwargs):

        if M is not None:
            if q_fn is None:
                q_fn = flat_massratio

            if e_M != 0:
                M = stats.norm(M,e_M).rvs(n)

        Simulated_BinaryPopulation.__init__(self, M=M, q_fn=q_fn,
                                            P_fn=draw_raghavan_periods,
                                            ecc_fn=draw_eccs, n=n,
                                            qmin=qmin,
                                            ichrone=ichrone,
                                            age=age, feh=feh,
                                            minmass=minmass, **kwargs)

class TriplePopulation(StarPopulation):
    """A population of triple stars.

    (Primary) orbits (secondary + tertiary) in a long orbit;
    secondary and tertiary orbit each other with a shorter orbit.
    Single or double stars may be indicated if desired by having
    the masses of secondary or tertiary set to zero, and all magnitudes
    to ``inf``.

    :param stars: (optional)
        Full stars ``DataFrame``.  If not passed, then primary, secondary,
        and tertiary must be.

    :param primary,secondary,tertiary: (optional)
        Properties of primary, secondary, and tertiary stars,
        in :class:`pandas.DataFrame` form.
        These will get merged into a new ``stars`` attribute,
        with "_A", "_B", and "_C" tags.

    :param orbpop: (optional)
        Object describing orbits of stars.  If not provided, then the period
        and eccentricity keywords must be provided, or else they will be
        randomly generated (see below).
    :type orbpop:
        :class:`TripleOrbitPopulation`

    :param period_short,period_long,ecc_short,ecc_long: (array-like, optional)
        Orbital periods and eccentricities of short and long-period orbits.
        "Short" describes the close pair of the hierarchical system; "long"
        describes the separation between the two major components.  Randomly
        generated if not provided.


    """
    def __init__(self, stars=None,
                 primary=None, secondary=None,
                 tertiary=None,
                 orbpop=None,
                 period_short=None, period_long=None,
                 ecc_short=0, ecc_long=0,
                 **kwargs):

        if stars is None and primary is not None:
            assert len(primary)==len(secondary) and len(primary)==len(tertiary)
            N = len(primary)

            stars = pd.DataFrame()

            for c in primary.columns:
                if re.search('_mag',c):
                     stars[c] = addmags(primary[c],secondary[c],tertiary[c])
                stars['{}_A'.format(c)] = primary[c]
            for c in secondary.columns:
                stars['{}_B'.format(c)] = secondary[c]
            for c in tertiary.columns:
                stars['{}_C'.format(c)] = tertiary[c]


            if orbpop is None:
                if period_long is None or period_short is None:
                    period_1 = draw_raghavan_periods(N)
                    period_2 = draw_msc_periods(N)
                    period_short = np.minimum(period_1, period_2)
                    period_long = np.maximum(period_1, period_2)

                if ecc_short is None:
                    ecc_short = draw_eccs(N,period_short)
                if ecc_long is None:
                    ecc_long = draw_eccs(N,period_long)

                M1 = stars['mass_A']
                M2 = stars['mass_B']
                M3 = stars['mass_C']

                orbpop = TripleOrbitPopulation(M1,M2,M3,period_long,period_short,
                                               ecclong=ecc_long, eccshort=ecc_short)

        StarPopulation.__init__(self, stars=stars, orbpop=orbpop, **kwargs)

    def dmag(self, band):
        """
        Difference in magnitudes between fainter and brighter components in band.

        :param band:
            Photometric bandpass.

        """
        m1 = self.stars['{}_mag_A'.format(band)]
        m2 = addmags(self.stars['{}_mag_B'.format(band)],
                     self.stars['{}_mag_C'.format(band)])
        return np.abs(m2-m1)

    def A_brighter(self, band='g'):
        """
        Instances where star A is brighter than (B+C)
        """
        mA = self.stars['{}_mag_A'.format(band)]
        mBC = addmags(self.stars['{}_mag_B'.format(band)],
                     self.stars['{}_mag_C'.format(band)])
        return mA < mBC

    def BC_brighter(self, band='g'):
        """
        Instances where stars (B+C) are brighter than star A
        """
        return ~self.A_brighter(band=band)

    def dRV(self, dt, band='g'):
        """Returns dRV of star A, if A is brighter than B+C, or of star B if B+C is brighter
        """
        return (self.orbpop.dRV_1(dt)*self.A_brighter(band) +
                self.orbpop.dRV_2(dt)*self.BC_brighter(band))

    @property
    def Plong(self):
        """
        Longer of two orbital periods in Triple system
        """
        return self.orbpop.orbpop_long.P

    @property
    def singles(self):
        return self.stars.query('mass_B==0 and mass_C==0')

    @property
    def binaries(self):
        return self.stars.query('mass_B > 0 and mass_C==0')

    @property
    def triples(self):
        return self.stars.query('mass_B > 0 and mass_C > 0')

    def binary_fraction(self,query='mass_A > 0', unc=False):
        """
        Binary fraction of stars following given query
        """
        subdf = self.stars.query(query)
        nbinaries = ((subdf['mass_B'] > 0) & (subdf['mass_C']==0)).sum()
        frac = nbinaries/len(subdf)
        if unc:
            return frac, frac/np.sqrt(nbinaries)
        else:
            return frac

    def triple_fraction(self,query='mass_A > 0', unc=False):
        """
        Triple fraction of stars following given query
        """
        subdf = self.stars.query(query)
        ntriples = ((subdf['mass_B'] > 0) & (subdf['mass_C'] > 0)).sum()
        frac = ntriples/len(subdf)
        if unc:
            return frac, frac/np.sqrt(ntriples)
        else:
            return frac

class Observed_BinaryPopulation(BinaryPopulation):
    """
    A population of binary stars matching observed constraints.

    :param mags:
        Observed apparent magnitudes
    :type mags:
        ``dict``

    :param Teff,logg,feh:
        Observed spectroscopic properties of primary star, if available.
        Format: ``(value, err)``.

    :param starmodel:
        :class:`isochrones.BinaryStarModel`. If not
        passed, it will be generated.


    """
    def __init__(self, mags=None, mag_errs=None,
                 Teff=None,
                 logg=None, feh=None,
                 starmodel=None, n=2e4,
                 ichrone='mist', bands=BANDS,
                 period=None, ecc=None,
                 orbpop=None, stars=None,
                 **kwargs):

        self.mags = mags
        self.mag_errs = mag_errs
        self.Teff = Teff
        self.logg = logg
        self.feh = feh
        self._starmodel = starmodel

        if stars is None and mags is not None \
           or starmodel is not None:

            self.generate(mags=mags, mag_errs=mag_errs,
                          n=n, ichrone=ichrone,
                          starmodel=starmodel,
                          Teff=Teff, logg=logg, feh=feh,
                          bands=bands, orbpop=orbpop,
                          period=period, ecc=ecc, **kwargs)

        else:
            self.stars = stars
            self.orbpop = orbpop

    @property
    def starmodel_props(self):
        """Default mag_err is 0.05, arbitrarily
        """
        props = {}
        mags = self.mags
        mag_errs = self.mag_errs
        for b in mags.keys():
            if np.size(mags[b])==2:
                props[b] = mags[b]
            elif np.size(mags[b])==1:
                mag = mags[b]
                try:
                    e_mag = mag_errs[b]
                except:
                    e_mag = 0.05
                props[b] = (mag, e_mag)

        if self.Teff is not None:
            props['Teff'] = self.Teff
        if self.logg is not None:
            props['logg'] = self.logg
        if self.feh is not None:
            props['feh'] = self.feh

        return props


    def generate(self, mags=None, mag_errs=None,
                 n=1e4, ichrone='mist',
                 starmodel=None, Teff=None, logg=None, feh=None,
                 bands=BANDS, orbpop=None, period=None,
                 ecc=None, **kwargs):

        ichrone = get_ichrone(ichrone, bands=bands)
        if starmodel is None:
            params = self.starmodel_props
            logging.info('Fitting BinaryStarModel to {}...'.format(params))
            starmodel = BinaryStarModel(ichrone, **params)
            starmodel.fit()
            logging.info('BinaryStarModel fit Done.')

        # if type(starmodel) != BinaryStarModel:
        #     raise TypeError('starmodel must be BinaryStarModel.')

        self._starmodel = starmodel

        samples = starmodel.random_samples(n)
        age, feh = (np.ascontiguousarray(samples['age_0']),
                    np.ascontiguousarray(samples['feh_0']))
        dist, AV = (samples['distance_0'], samples['AV_0'])
        mass_A, mass_B = (np.ascontiguousarray(samples['mass_0_0']),
                          np.ascontiguousarray(samples['mass_0_1']))
        primary = ichrone(mass_A, age, feh,
                          distance=dist, AV=AV, bands=BANDS)
        secondary = ichrone(mass_B, age, feh,
                            distance=dist, AV=AV, bands=BANDS)

        BinaryPopulation.__init__(self, primary=primary,
                                  secondary=secondary,
                                  orbpop=orbpop, period=period,
                                  ecc=ecc, **kwargs)


    def save_hdf(self, filename, path='', **kwargs):
        super(Observed_BinaryPopulation,self).save_hdf(filename, path=path, **kwargs)
        self.starmodel.save_hdf(filename, path='{}/starmodel'.format(path), append=True)

    @classmethod
    def load_hdf(cls, filename, path=''):
        pop = super(Observed_BinaryPopulation, cls).load_hdf(filename, path=path)
        pop._starmodel = BinaryStarModel.load_hdf(filename,
                                                  path='{}/starmodel'.format(path))
        return pop

    # def __getattr__(self, attr):
    #     # Don't remember why I've done this.  Must be a reason.
    #     if attr not in ['starmodel','_starmodel']:
    #         return getattr(self.starmodel, attr)

class Observed_TriplePopulation(TriplePopulation):
    """
    A population of triple stars matching observed constraints.

    :param mags:
        Observed apparent magnitudes
    :type mags:
        ``dict``

    :param Teff,logg,feh:
        Observed spectroscopic properties of primary star, if available.
        Format: ``(value, err)``.

    :param starmodel:
        :class:`isochrones.TripleStarModel`. If not
        passed, it will be generated.


    """
    def __init__(self, mags=None, mag_errs=None,
                 Teff=None,
                 logg=None, feh=None,
                 starmodel=None, n=2e4,
                 ichrone='mist', bands=BANDS,
                 period=None, ecc=None,
                 orbpop=None, stars=None,
                 **kwargs):

        self.mags = mags
        self.mag_errs = mag_errs
        self.Teff = Teff
        self.logg = logg
        self.feh = feh
        self._starmodel = starmodel

        if stars is None and mags is not None \
           or starmodel is not None:

            self.generate(mags=mags, mag_errs=mag_errs,
                          n=n, ichrone=ichrone,
                          starmodel=starmodel,
                          Teff=Teff, logg=logg, feh=feh,
                          bands=bands, orbpop=orbpop,
                          period=period, ecc=ecc, **kwargs)
        else:
            self.stars = stars
            self.orbpop = orbpop

    @property
    def starmodel_props(self):
        """Default mag_err is 0.05, arbitrarily
        """
        props = {}
        mags = self.mags
        mag_errs = self.mag_errs
        for b in mags.keys():
            if np.size(mags[b])==2:
                props[b] = mags[b]
            elif np.size(mags[b])==1:
                mag = mags[b]
                try:
                    e_mag = mag_errs[b]
                except:
                    e_mag = 0.05
                props[b] = (mag, e_mag)

        if self.Teff is not None:
            props['Teff'] = self.Teff
        if self.logg is not None:
            props['logg'] = self.logg
        if self.feh is not None:
            props['feh'] = self.feh

        return props


    def generate(self, mags=None, mag_errs=None,
                 n=1e4, ichrone='mist',
                 starmodel=None, Teff=None, logg=None, feh=None,
                 bands=BANDS, orbpop=None, period=None,
                 ecc=None, **kwargs):

        ichrone = get_ichrone(ichrone, bands=bands)

        if starmodel is None:
            params = self.starmodel_props
            logging.info('Fitting TripleStarModel to {}...'.format(params))
            starmodel = TripleStarModel(ichrone, **params)
            starmodel.fit()
            logging.info('TripleStarModel fit Done.')

        # if type(starmodel) != TripleStarModel:
        #     raise TypeError('starmodel must be TripleStarModel.')

        self._starmodel = starmodel

        samples = starmodel.random_samples(n)
        age, feh = (np.ascontiguousarray(samples['age_0']),
                    np.ascontiguousarray(samples['feh_0']))
        dist, AV = (samples['distance_0'], samples['AV_0'])
        mass_A, mass_B, mass_C = (np.ascontiguousarray(samples['mass_0_0']),
                                  np.ascontiguousarray(samples['mass_0_1']),
                                  np.ascontiguousarray(samples['mass_0_2']))
        primary = ichrone(mass_A, age, feh,
                          distance=dist, AV=AV, bands=BANDS)
        secondary = ichrone(mass_B, age, feh,
                            distance=dist, AV=AV, bands=BANDS)
        tertiary = ichrone(mass_C, age, feh,
                            distance=dist, AV=AV, bands=BANDS)

        TriplePopulation.__init__(self, primary=primary,
                                  secondary=secondary,
                                  tertiary=tertiary,
                                  orbpop=orbpop, period_short=period,
                                  ecc_short=ecc, **kwargs)

    def save_hdf(self, filename, path='', **kwargs):
        super(Observed_TriplePopulation,self).save_hdf(filename, path=path, **kwargs)
        self.starmodel.save_hdf(filename, path='{}/starmodel'.format(path), append=True)

    @classmethod
    def load_hdf(cls, filename, path=''):
        pop = super(Observed_TriplePopulation, cls).load_hdf(filename, path=path)
        pop._starmodel = TripleStarModel.load_hdf(filename,
                                                  path='{}/starmodel'.format(path))
        return pop

    # def __getattr__(self, attr):
    #     # Why did I do this again?  Probably a reason...
    #     if attr not in ['starmodel', '_starmodel']:
    #         return getattr(self.starmodel, attr)


class MultipleStarPopulation(TriplePopulation):
    """A population of single, double, and triple stars, generated according to prescription.

    :param mA: (optional)
        Mass of primary star(s).  Default=1.
        If array, then the simulation will be
        lots of individual systems; if float,
        then the simulation will be lots of
        realizations of one system.

    :param age,feh: (optional)
        Age, feh of system(s).

    :param f_binary,f_triple: (optional)
        Fraction of systems that should be binaries or triples.
        Should have ``f_binary + f_triple < 1``, though if
        ``f_binary + f_triple >= 1``, then ``f_binary`` will
        implicitly be treated as ``1 - f_triple``.

    :param qmin: (optional)
        Minimum mass ratio.

    :param minmass: (optional)
        Minimum stellar mass to simulate.

    :param n: (optional)
        Size of simulation (if ``mA`` is a scalar).  If
        ``mA`` is array-like, then ``n = len(mA)``.

    :param ichrone: (:class:`isochrones.Isochrone`, optional)
        Stellar model isochrone to generate simulations.  Defaults
        to Dartmouth model grid.

    :param bands: (optional)
        Photometry bandpasses to simulate using ``ichrone``.

    :param multmass_fn,period_long_fn,period_short_fn,ecc_fn: (optional)
        Functions to generate masses, orbital periods, and eccentricities.
        Defaults built in.  See :class`TriplePopulation`.

    :param orbpop: (optional)
        Object describing orbits of stars.  If not provided, orbits will
        be randomly generated according to generating functions.
    :type orbpop:
        :class:`orbits.TripleOrbitPopulation`

    Additional keyword arguments passed to :class:`TriplePopulation`.


    """
    def __init__(self, mA=None, age=9.6, feh=0.0,
                 f_binary=0.4, f_triple=0.12,
                 qmin=0.1, minmass=0.11,
                 n=1e4, ichrone='mist',
                 multmass_fn=mult_masses,
                 period=None,
                 period_long_fn=draw_raghavan_periods,
                 period_short_fn=draw_msc_periods,
                 period_short=None, period_long=None,
                 ecc_fn=draw_eccs,
                 ecc_kws=None,
                 bands=BANDS,
                 orbpop=None, stars=None,
                 **kwargs):

        #These get set even if stars is passed
        self.f_binary = f_binary
        self.f_triple = f_triple
        self.qmin = qmin
        self.minmass = minmass
        self.multmass_fn = multmass_fn
        self.period_long_fn = period_long_fn
        self.period_short_fn = period_short_fn
        if period_long is not None:
            self.period_long_fn = None
        if period_short is not None:
            self.period_short_fn = None
        self.ecc_fn = ecc_fn

        if stars is None and mA is not None:
            self.generate(mA=mA, age=age, feh=feh, n=n, ichrone=ichrone,
                          orbpop=orbpop, bands=bands, period_long=period_long,
                          period_short=period_short, **kwargs)
        else:
            TriplePopulation.__init__(self, stars=stars, orbpop=orbpop, **kwargs)


    def generate(self, mA=1, age=9.6, feh=0.0, n=1e5, ichrone='mist',
                 orbpop=None, bands=None, **kwargs):
        """
        Generates population.

        Called if :class:`MultipleStarPopulation` is initialized without
        providing ``stars``, and if ``mA`` is provided.

        """
        ichrone = get_ichrone(ichrone, bands=bands)

        n = int(n)
        #star with m1 orbits (m2+m3).  So mA (most massive)
        # will correspond to either m1 or m2.
        m1, m2, m3 = self.multmass_fn(mA, f_binary=self.f_binary,
                                      f_triple=self.f_triple,
                                      qmin=self.qmin, minmass=self.minmass,
                                      n=n)
        #reset n if need be
        n = len(m1)

        feh = np.ascontiguousarray(np.atleast_1d(feh))
        age = np.ascontiguousarray(age)

        #generate stellar properties
        primary = ichrone(np.ascontiguousarray(m1), age, feh,
                          bands=bands)
        secondary = ichrone(np.ascontiguousarray(m2),age,feh,
                            bands=bands)
        tertiary = ichrone(np.ascontiguousarray(m3),age,feh,
                           bands=bands)

        #clean up columns that become nan when called with mass=0
        # Remember, we want mass=0 and mags=inf when something doesn't exist
        no_secondary = (m2==0)
        no_tertiary = (m3==0)
        for c in secondary.columns: #
            if re.search('_mag',c):
                secondary[c][no_secondary] = np.inf
                tertiary[c][no_tertiary] = np.inf
        secondary['mass'][no_secondary] = 0
        tertiary['mass'][no_tertiary] = 0

        if kwargs['period_short'] is None:
            if kwargs['period_long'] is None:
                period_1 = self.period_long_fn(n)
                period_2 = self.period_short_fn(n)
                kwargs['period_short'] = np.minimum(period_1, period_2)
                kwargs['period_long'] = np.maximum(period_1, period_2)
            else:
                kwargs['period_short'] = self.period_short_fn(n)

                #correct any short periods that are longer than period_long
                bad = kwargs['period_short'] > kwargs['period_long']
                n_bad = bad.sum()
                good_inds = np.where(~bad)[0]
                inds = np.random.randint(len(good_inds),size=n_bad)
                kwargs['period_short'][bad] = \
                    kwargs['period_short'][good_inds[inds]]
        else:
            if kwargs['period_long'] is None:
                kwargs['period_long'] = self.period_long_fn(n)

                #correct any long periods that are shorter than period_short
                bad = kwargs['period_long'] < kwargs['period_short']
                n_bad = bad.sum()
                good_inds = np.where(~bad)[0]
                inds = np.random.randint(len(good_inds),size=n_bad)
                kwargs['period_long'][bad] = \
                    kwargs['period_long'][good_inds[inds]]

        if 'ecc_short' not in kwargs:
            kwargs['ecc_short'] = self.ecc_fn(n, kwargs['period_short'])
        if 'ecc_long' not in kwargs:
            kwargs['ecc_long'] = self.ecc_fn(n, kwargs['period_long'])

        TriplePopulation.__init__(self, primary=primary,
                                  secondary=secondary, tertiary=tertiary,
                                  orbpop=orbpop, **kwargs)

        return self

    @property
    def _properties(self):
        return ['f_binary', 'f_triple',
                'qmin', 'minmass',
                'period_long_fn', 'period_short_fn',
                'ecc_fn'] + super(MultipleStarPopulation, self)._properties


class BGStarPopulation(StarPopulation):
    """Background star population

    This should usually be accessed via the
    :class:`BGStarPopulation_TRILEGAL` subclass.

    :param stars: (:class:`pandas.DataFrame`, optional)
        Properties of stars.  Must have 'distance' column defined.

    :param mags: (optional)
        Magnitudes of primary (foreground) stars.

    :param maxrad: (optional)
        Maximum distance (arcseconds) of BG stars from
        foreground primary star.

    :param density: (optional)
        Density in arcsec^{-2} for BG star population.

    :param **kwargs:
        Additional keyword arguments passed to :class:`StarPopulation`.

    """
    def __init__(self,stars=None,mags=None,maxrad=1800,density=None,
                 **kwargs):

        self.mags = mags

        if stars is not None:
            if 'distance' not in stars:
                raise ValueError('Stars must have distance column defined')

            if density is None:
                self.density = len(stars)/((3600.*u.arcsec)**2) #default is for TRILEGAL sims to be 1deg^2
            else:
                if type(density)!=Quantity:
                    raise ValueError('Provided stellar density must have units.')
                self.density = density

            if type(maxrad) != Quantity:
                self._maxrad = maxrad*u.arcsec #arcsec
            else:
                self._maxrad = maxrad

        StarPopulation.__init__(self,stars=stars, **kwargs)

        if stars is not None:
            self.stars['Rsky'] = randpos_in_circle(len(stars),maxrad,return_rad=True)

    @property
    def Rsky(self):
        """
        Project on-sky separation between primary star and BG stars

        """
        return np.array(self.stars['Rsky'])*u.arcsec

    @property
    def maxrad(self):
        return self._maxrad

    @maxrad.setter
    def maxrad(self,value):
        if type(value) != Quantity:
            value = value*u.arcsec
        self.stars['Rsky'] *= (value/self._maxrad).decompose()
        self._maxrad = value

        #look for contrast curve constraints & re-apply them
        cc_names = []
        cc_list = []
        for k,c in self.constraints.items():
            if isinstance(c, ContrastCurveConstraint):
                cc_names.append(k)
                cc_list.append(c.cc)

        for name,cc in zip(cc_names, cc_list):
            self.remove_constraint(name)
            self.apply_cc(cc)
            logging.warning('maxrad changed for {} population; {} contrast curve re-applied'.format(self.name, cc.name))

    def dmag(self,band):
        """
        Magnitude difference between primary star and BG stars

        """
        if self.mags is None:
            raise ValueError('dmag is not defined because primary mags are not defined for this population.')
        return self.stars['{}_mag'.format(band)] - self.mags[band]

    @property
    def _properties(self):
        return ['mags', '_maxrad', 'density'] + \
            super(BGStarPopulation, self)._properties

class BGStarPopulation_TRILEGAL(BGStarPopulation):
    """Creates TRILEGAL simulation for ra,dec; loads as BGStarPopulation

    :param filename:
        Desired name of the TRILEGAL simulation.  Can either have '.h5' extension
        or not.  If filename (or 'filename.h5') exists locally, it will be
        loaded; otherwise, TRILEGAL will be called via the ``get_trilegal`` perl
        script, and the file will be generated.

    :param ra,dec: (optional)
        Sky coordinates of TRILEGAL simulation.  Must be passed if generating
        TRILEGAL simulation and not just reading from existing file.

    :param mags: (optional)
        Dictionary of primary star magnitudes (if this is being used to generate
        a background population behind a particular foreground star).  This
        must be set in order to use the ``dmag`` attribute.

    :type mags: (optional)
        ``dict``

    :param maxrad: (optional)
        Maximum distance (arcsec) out to which to place simulated stars.

    :param **kwargs:
        Additional keyword arguments passed to
        :func:`stars.trilegal.get_trilegal`

    """
    def __init__(self,filename=None,ra=None,dec=None,mags=None,maxrad=1800,
                 **kwargs):

        self.trilegal_args = {}

        if filename is None:
            BGStarPopulation.__init__(self)
        else:
            m = re.search('(.*)\.h5$',filename)
            if not m:
                h5filename = '{}.h5'.format(filename)
                basefilename = filename
            else:
                h5filename = filename
                basefilename = m.group(1)

            if os.path.exists(h5filename):
                logging.info('Loading TRILEGAL simulation from {}'.format(h5filename))
                stars = pd.read_hdf(h5filename,'df')
            else:
                if ra is None or dec is None:
                    raise ValueError('Must provide ra,dec if simulation file does not already exist.')
                logging.info('Getting TRILEGAL simulation at {}, {}...'.format(ra,dec))
                get_trilegal(basefilename,ra,dec,**kwargs)
                logging.info('Done.')
                stars = pd.read_hdf(h5filename,'df')

            with pd.HDFStore(h5filename) as store:
                self.trilegal_args = store.get_storer('df').attrs.trilegal_args

            c = SkyCoord(self.trilegal_args['l'],self.trilegal_args['b'],
                         unit='deg',frame='galactic')

            self.coords = c.icrs

            area = self.trilegal_args['area']*(u.deg)**2
            density = len(stars)/area

            stars['distmod'] = stars['m-M0']
            stars['distance'] = dfromdm(stars['distmod'])

            BGStarPopulation.__init__(self,stars,mags=mags,maxrad=maxrad,
                                      density=density,**kwargs)

    @property
    def _properties(self):
        return ['trilegal_args'] + \
            super(BGStarPopulation_TRILEGAL,self)._properties

############## Exceptions ################

class PoorColorsError(Exception):
    pass


#methods below should be applied to relevant subclasses
'''
    def set_dmaglim(self,dmaglim):
        if not (hasattr(self,'blendmag') and hasattr(self,'dmaglim')):
            return
        self.dmaglim = dmaglim
        self.apply_constraint(LowerLimit(self.dmags(),self.dmaglim,name='bright blend limit'),overwrite=True)
        self._apply_all_constraints()  #not necessary?
'''
