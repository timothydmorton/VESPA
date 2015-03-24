.. _stars:

Star Populations
=============

.. module:: vespa.stars

The fundamental population unit within ``vespa`` is a
:class:`StarPopulation`, from which :class:`vespa.EclipsePopulation` inherits.
This is the basic object which keeps track of the properties of a population
of stars and enables application of various observational constraints to rule
out portions of the population.

Both :class:`vespa.EBPopulation` and :class:`vespa.HEBPopulation` inherit from
:class:`MultipleStarPopulation`.  :class:`vespa.BEBPopulation` inherits from :class:`BGStarPopulation` through
:class:`BGStarPopulation_TRILEGAL`. 

.. autoclass:: vespa.stars.populations.StarPopulation
   :members:

Multiple Star Population
------------------

Both :class:`vespa.EBPopulation` and :class:`vespa.HEBPopulation` inherit from
:class:`MultipleStarPopulation`.  Depending on what observational
constraints there may be on the properties of the target star, this may be
through :class:`ColormatchMultipleStarPopulation` or
:class:`Spectroscopic_MultipleStarPopulation`.

.. autoclass:: vespa.stars.populations.MultipleStarPopulation
  :members:

.. autoclass:: vespa.stars.populations.ColormatchMultipleStarPopulation
  :members:

.. autoclass:: vespa.stars.populations.Spectroscopic_MultipleStarPopulation
  :members:

Background Star Population
----------------------

:class:`vespa.BEBPopulation` inherits from :class:`BGStarPopulation` through
:class:`BGStarPopulation_TRILEGAL`.  

.. autoclass:: vespa.stars.populations.BGStarPopulation_TRILEGAL
  :members:
	       
.. autoclass:: vespa.stars.populations.BGStarPopulation
  :members:

Other Star Populations
---------

These are the other :class:`StarPopulation` classes defined in ``vespa``.
:class:`Raghavan_BinaryPopulation` is particularly useful, which
produces a population according to the binary distribution
described by the `Raghavan (2010) <http://arxiv.org/abs/1007.0414>`_
survey.

.. autoclass:: vespa.stars.populations.BinaryPopulation
  :members:

.. autoclass:: vespa.stars.populations.Simulated_BinaryPopulation
  :members:

.. autoclass:: vespa.stars.populations.Raghavan_BinaryPopulation
  :members:
     
.. autoclass:: vespa.stars.populations.TriplePopulation
  :members:

     
