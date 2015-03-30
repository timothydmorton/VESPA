.. _stars:

Star Populations
=============

.. module:: vespa.stars

The fundamental population unit within ``vespa`` is a
:class:`StarPopulation`, from which :class:`EclipsePopulation` inherits.
This is the basic object which keeps track of the properties of a population
of stars and enables application of various observational constraints to rule
out portions of the population.

For the built-in false positive populations, :class:`EBPopulation` inherits from
:class:`Observed_BinaryPopulation`, and :class:`HEBPopulation` inherits
from :class:`Observed_TriplePopulation`.
:class:`BEBPopulation` inherits from :class:`BGStarPopulation` through
:class:`BGStarPopulation_TRILEGAL`. 

.. autoclass:: vespa.stars.StarPopulation
   :members:

Observationally Constrained Star Populations
------------------

:class:`EBPopulation` and :class:`HEBPopulation` inherit from
very similar star population classes:
:class:`Observed_BinaryPopulation` and
:class:`Observed_TriplePopulation`.  Both of these take either
photometric or spectroscopic observed properties of a
star and generate binary or triple populations consistent with those
observations.

.. autoclass:: vespa.stars.Observed_BinaryPopulation
  :members:

.. autoclass:: vespa.stars.Observed_TriplePopulation
  :members:

Background Star Population
----------------------

:class:`BEBPopulation` inherits from :class:`BGStarPopulation` through
:class:`BGStarPopulation_TRILEGAL`.  

.. autoclass:: vespa.stars.BGStarPopulation_TRILEGAL
  :members:
	       
.. autoclass:: vespa.stars.BGStarPopulation
  :members:

Other Star Populations
---------

These are the other :class:`StarPopulation` classes defined in ``vespa``.
:class:`Raghavan_BinaryPopulation` is particularly useful, which
produces a population according to the binary distribution
described by the `Raghavan (2010) <http://arxiv.org/abs/1007.0414>`_
survey.

.. autoclass:: vespa.stars.BinaryPopulation
  :members:

.. autoclass:: vespa.stars.Simulated_BinaryPopulation
  :members:

.. autoclass:: vespa.stars.Raghavan_BinaryPopulation
  :members:
     
.. autoclass:: vespa.stars.TriplePopulation
  :members:

     
