.. _eclipse:

Eclipse Populations
=============

.. py:currentmodule:: vespa.populations

All physical eclipse models proposed as potential explanations for
an obseved transit signal are defined as :class:`EclipsePopulation`
objects.  Currently implemented within ``vespa`` are :class:`EBPopulation`,
:class:`HEBPopulation`, :class:`BEBPopulation`, and :class:`PlanetPopulation`.

.. note::
    More subclasses are under development for other scenarios, in particular
    eclipses around *specific* observed stars.

Also see the documentation for :class:`vespa.stars.StarPopulation`, from which
:class:`EclipsePopulation` derives.
       
.. autoclass:: vespa.populations.EclipsePopulation
   :members:

Undiluted Eclipsing Binary
-------------------

.. autoclass:: vespa.populations.EBPopulation
  :members:

Hierarchical Elipsing Binary
----------------------

.. autoclass:: vespa.populations.HEBPopulation
  :members:

Background Eclipsing Binary
------------------

.. autoclass:: vespa.populations.BEBPopulation
  :members:

Transiting Planet
---------------

.. autoclass:: vespa.populations.PlanetPopulation
  :members:
