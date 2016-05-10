.. _transits:

Transit Utilities
=============

.. py:currentmodule:: vespa.transit_basic

In order to enable fast simulation of large numbers of eclipses, ``vespa``
makes use of the Mandel-Agol (2002) transit model
implemented by the `batman <https://github.com/lkreidberg/batman>`_ module.


.. automodule:: vespa.transit_basic
    :members: ldcoeffs, impact_parameter, transit_T14, transit_T23,
	    minimum_inclination, a_over_Rs, eclipse_tz, eclipse_pars,
	    eclipse, eclipse_tt, occultquad, TraptransitModel, traptransit_MCMC

		  
