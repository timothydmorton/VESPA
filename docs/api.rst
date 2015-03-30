.. _api:

High-level API
===========

.. py:currentmodule:: vespa

This page details the top-level classes that provide access to the
``vespa`` module.  The simplest entry point into these calculations is
the ``calcfpp`` command-line script, which creates a
:class:`FPPCalculation` object using :func:`FPPCalculation.from_ini`,
and creates a bunch of data files/diagnostic plots.  A
:class:`FPPCalculation` is made up of a :class:`PopulationSet` and a
:class:`TransitSignal`.

For more details on the guts of the objects that make up a
:class:`PopulationSet`, please see the documentation on :ref:`eclipse`,
:ref:`stars`, and :ref:`orbits`.

FPPCalculation
-----------------

.. autoclass:: vespa.FPPCalculation
   :members:

PopulationSet
-----------------

This object is essentially an organized list of
:class:`EclipsePopulation` objects.  

.. autoclass:: vespa.PopulationSet
  :members:

TransitSignal
----------------

.. autoclass:: vespa.TransitSignal
  :members:
     
