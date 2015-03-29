.. _constraints:

Observational Constraints
=============

.. module:: vespa.stars.constraints

The mechanism for incorporating observational constraints into the
``vespa`` calculations is via the :class:`Constraint` object. The way
this is currently implemented is that a :class:`Constraint` is
essentially a boolean array of the same length as
a :class:`EclipsePopulation` (or :class:`StarPopulation`, more
generally), where simulated instances that would not have been
detected by the observation in question remain ``True``, and any
instances that would have been observed become ``False``.

.. module:: vespa.stars.contrastcurve

Contrast Curve Constraint
---------------------------

One of the most common kinds of follow-up observation for false
positive identification/ analysis is a high-resolution imaging
observation. The output of such an observation is a "contrast curve":
the detectable brightness contrast as a function of angular separation
from the central source. As every false
positive :class:`EclipsePopulation` simulation includes simulated
magnitudes in many different bands as well as simulated sky-positions
relative to the central target star, it is very easy to implement a
contrast curve in this way: any instances that would have been
detected by the observation get ruled out, and thus the "prior" factor
diminishes for that scenario (this is kept track of by
the :attr:`EclipsePopulation.countok` attribute).  

.. autoclass:: vespa.stars.contrastcurve.ContrastCurve
  :members:

.. autoclass:: vespa.stars.contrastcurve.ContrastCurveFromFile

.. autoclass:: vespa.stars.contrastcurve.ContrastCurveConstraint
  :members:

     
