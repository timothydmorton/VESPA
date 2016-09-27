from __future__ import print_function,division

import os,re
import logging
import subprocess as sp

try:
    from astropy.coordinates import SkyCoord
except:
    SkyCoord = None
   
from isochrones.exctinction import get_AV_infinity    
