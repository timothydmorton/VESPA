from __future__ import print_function, division

import logging

import numpy as np
import matplotlib.pyplot as plt

from starutils.populations import StarPopulation

SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}
                        
INV_SHORT_MODELNAMES = {v:k for k,v in SHORT_MODELNAMES.iteritems()}


class EclipsePopulation(StarPopulation):
    def __init__(self, stars, trapfit_params,
                 **kwargs):
        """Base class for populations of eclipsing things.


        
        """
        
