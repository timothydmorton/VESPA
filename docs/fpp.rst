.. _fpp:

False Positive Probability Calculation
================

.. py:currentmodule:: vespa

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
--------------

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
------

Each :class:`EclipsePopulation` also has a
:attr:`EclipsePopulation.prior` attribute, the value of which 
represents the probability of that particular astrophysical scenario
existing.  For a :class:`BEBPopulation`, for example, the prior is
``(star density) * (sky area) * (binary fraction) * (eclipse
probability)``.  If observational constraints are applied to a
population, then an additional ``selectfrac`` factor will be
multiplied into the prior, representing the fraction of scenarios that
are still allowed to exist, given the constraints.
