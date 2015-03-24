.. _transits:

Transit Utilities
=============

.. py:currentmodule:: vespa.transit_basic

In order to enable simulation of large numbers of eclipses, ``vespa``
includes a number of utilities relating to calculating and fitting transit shapes.
In particular, the :class:`MAInterpolationFunction`
object enables fast, vectorized evaluation of the Mandel-Agol
transit model, based on interpolating a pre-calculated interpolation grid.

Mandel-Agol Interpolation Object
-----------------------

.. autoclass:: vespa.transit_basic.MAInterpolationFunction


Fitting Trapezoid Models
--------------------

In order to generate population distribution of trapezoidal shape parameters,
all the eclipses in an :class:`vespa.EclipsePopulation` object must get their
theoretical Mandel-Agol eclipse shapes fit with the trapezoid model.  This is
the function that does 

.. autofunction:: vespa.fitebs.fitebs

	       
Other Utility Functions
------------------

There's a lot going on behind the scenes here. Highlights
include :func:`ldcoeffs`, :func:`eclipse`, the Eastman &
Agol :func:`occultquad` (from some generation of `exofast
<http://astroutils.astronomy.ohio-state.edu/exofast/>`_, maybe?),
and :func:`traptransit_MCMC`. Apologies for the occasionally sparse
documentation; source code should be visible.

.. automodule:: vespa.transit_basic
  :members: ldcoeffs, impact_parameter, transit_T14,
	    transit_T23, minimum_inclination, a_over_Rs,
	    eclipse_tz, eclipse_pars, eclipse, eclipse_tt,
	    occultquad, TraptransitModel, traptransit_MCMC

		  
