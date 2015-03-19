vespa
=================================

``vespa`` is a Python package built to enable automated false positive 
analysis of transiting planet signals. It implements the latest version of the 
general procedure described in detail in `Morton (2012) <http://adsabs.harvard.edu/abs/2012ApJ...761....6M>`_.

Installation
------------

To install, you can get the most recently released version from PyPI::

    pip install vespa [--user]

Or you can clone from github::

    git clone https://github.com/timothydmorton/vespa.git
    cd vespa
    python setup.py install [--user]

The ``--user`` argument may be necessary if you don't have root privileges.

Overview
--------

A false positive probability calculation in ``vespa`` is built of two basic
components: a :class:`TransitSignal` and a :class:`PopulationSet`,
joined together in a :class:`FPPCalculation` object.  The 
:class:`TransitSignal` holds the data about the transit signal photometry,
and the :class:`PopulationSet` contains a set of simulated
populations, one :class:`EclipsePopulation` for each astrophysical
model that is considered as a possible origin for the observed
transit-like signal.  By default, the populations included will be
the true transiting planet (:class:`PlanetPopulation`) and
three astrophysical false positive scenarios:
undiluted eclipsing binary (:class:`EBPopulation`),  hierarchical
triple eclipsing binary ( :class:`HEBPopulation`), and background
eclipsing binary (:class:`BEBPopulation`).

The :class:`EclipsePopulation` object derives from the more general
:class:`StarPopulation`, which is useful beyond false positive
calculations, such as for generating a hypothetical population of
binary companions for a given star in order to help quantify
completeness to stellar companions of an imaging survey.  

.. _fpp-section:
False Positive Probability Calculation
----------------------

``vespa`` calculates the false positive probability for a transit
signal as follows:

  .. math::

     {\rm FPP} = 1 - P_{\rm pl},

where
     
  .. math::

     P_{\rm pl} = \frac{\mathcal L_{\rm pl} \pi_{\rm pl}}
                                 {\mathcal L_{\rm pl} \pi_{\rm pl} +
				 \mathcal L_{\rm FP} \pi_{\rm FP}}.

The :math:`\mathcal L_i` here represent the "model likelihood"
factors and the :math:`\pi_i` represent the "model priors," with the
:math:`{\rm FP}` subscript representing the sum of :math:`\mathcal L_i
\pi_i` for each of the false positive scenarios.

Likelihoods
^^^^^^^^^^^

Each :class:`EclipsePopulation` contains a large number of simulated
instances of the particular physical scenario, each of which has a
simulated eclipse shape and a corresponding trapezoidal fit.  This
enables each population to define a 3-dimensional probability
distribution function (PDF) for these trapezoid parameters,
:math:`p_{\rm mod} (\log_{10} (\delta), T, T/\tau)`.  As the
:class:`TransitSignal` object provides an MCMC sampling of the
trapezoid parameters for the observed transit signal, the likelihood
of the transit signal under a given model can thus be approximated as
a sum over the model PDF evaluated at the :math:`K` samples:

  .. math::
     
     \mathcal L = \displaystyle \sum_{k=1}^K  p_{\rm mod}
                                  \left(\log_{10} (\delta_k),
                                  T_k, (T/\tau)_k\right)

This is implemented in :func:`EclipsePopulation.lhood`.

Priors
^^^^^^

Each :class:`EclipsePopulation` also has a
:attr:`EclipsePopulation.prior` attribute, the value of which 
represents the probability of that particular astrophysical scenario
existing.  For a :class:`BEBPopulation`, for example, the prior is
``(star density) * (sky area) * (binary fraction) * (eclipse
probability)``.  If observational constraints are applied to a
population, then an additional ``selectfrac`` factor will be
multiplied into the prior, representing the fraction of scenarios that
are still allowed to exist, given the constraints.



API Documentation
-----------------

.. toctree::
   :maxdepth: 2


   api
