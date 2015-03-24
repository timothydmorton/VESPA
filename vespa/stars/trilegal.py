from __future__ import print_function,division

import logging
import subprocess as sp
import os, re

try:
    import numpy as np
    import pandas as pd

    from astropy.units import UnitsError
    from astropy.coordinates import SkyCoord
except ImportError:
    np, pd = (None, None)
    UnitsError, SkyCoord = (None, None)
    
from .extinction import get_AV_infinity

NONMAG_COLS = ['Gc','logAge', '[M/H]', 'm_ini', 'logL', 'logTe', 'logg',
               'm-M0', 'Av', 'm2/m1', 'mbol', 'Mact'] #all the rest are mags

def get_trilegal(filename,ra,dec,folder='.',
                 filterset='kepler_2mass',area=1,maglim=27,binaries=False,
                 trilegal_version='1.6',sigma_AV=0.1,convert_h5=True):
    """Runs get_trilegal perl script; optionally saves output into .h5 file

    Depends on a perl script provided by L. Girardi; calls the
    web form simulation, downloads the file, and (optionally) converts
    to HDF format.

    Uses A_V at infinity from :func:`utils.get_AV_infinity`.

    .. note::

        Would be desirable to re-write the get_trilegal script
        all in python.
                 
    :param filename:
        Desired output filename.  If extension not provided, it will
        be added.

    :param ra,dec:
        Coordinates (ecliptic) for line-of-sight simulation.

    :param folder: (optional)
        Folder to which to save file.  *Acknowledged, file control
        in this function is a bit wonky.*

    :param filterset: (optional)
        Filter set for which to call TRILEGAL.

    :param area: (optional)
        Area of TRILEGAL simulation [sq. deg]

    :param maglim: (optional)
        Limiting magnitude in first mag (by default will be Kepler band)
        If want to limit in different band, then you have to
        got directly to the ``get_trilegal`` perl script.

    :param binaries: (optional)
        Whether to have TRILEGAL include binary stars.  Default ``False``.

    :param trilegal_version: (optional)
        Default ``'1.6'``.

    :param sigma_AV: (optional)
        Fractional spread in A_V along the line of sight.

    :param convert_h5: (optional)
        If true, text file downloaded from TRILEGAL will be converted
        into a ``pandas.DataFrame`` stored in an HDF file, with ``'df'``
        path.
                 
    """
    try:
        c = SkyCoord(ra,dec)
    except UnitsError:
        c = SkyCoord(ra,dec,unit='deg')
    l,b = (c.galactic.l.value,c.galactic.b.value)

    if os.path.isabs(filename):
        folder = ''

    if not re.search('\.dat$',filename):
        outfile = '{}/{}.dat'.format(folder,filename)
    else:
        outfile = '{}/{}'.format(folder,filename)
    AV = get_AV_infinity(l,b,frame='galactic')
    cmd = 'get_trilegal %s %f %f %f %i %.3f %.2f %s 1 %.1f %s' % (trilegal_version,l,b,
                                                                  area,binaries,AV,sigma_AV,
                                                                  filterset,maglim,outfile)
    sp.Popen(cmd,shell=True).wait()
    if convert_h5:
        df = pd.read_table(outfile, sep='\s+', skip_footer=1, engine='python')
        df = df.rename(columns={'#Gc':'Gc'})
        for col in df.columns:
            if col not in NONMAG_COLS:
                df.rename(columns={col:'{}_mag'.format(col)},inplace=True)
        if not re.search('\.h5$', filename):
            h5file = '{}/{}.h5'.format(folder,filename)
        else:
            h5file = '{}/{}'.format(folder,filename)
        df.to_hdf(h5file,'df')
        store = pd.HDFStore(h5file)
        attrs = store.get_storer('df').attrs
        attrs.trilegal_args = {'version':trilegal_version,
                               'l':l,'b':b,'area':area,
                               'AV':AV, 'sigma_AV':sigma_AV,
                               'filterset':filterset,
                               'maglim':maglim,
                               'binaries':binaries}
        store.close()
        os.remove(outfile)
    
    
