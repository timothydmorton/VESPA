from __future__ import print_function,division

import logging
import subprocess as sp
import os, re
import time

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

def get_trilegal(filename,ra,dec,folder='.', galactic=False,
                 filterset='TESS_2mass_kepler',area=1,maglim=27,binaries=False,
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
    if galactic:
        l, b = ra, dec
    else:
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
    #cmd = 'get_trilegal %s %f %f %f %i %.3f %.2f %s 1 %.1f %s' % (trilegal_version,l,b,
    #                                                              area,binaries,AV,sigma_AV,
    #                                                              filterset,maglim,outfile)
    #sp.Popen(cmd,shell=True).wait()
    trilegal_webcall(trilegal_version,l,b,area,binaries,AV,sigma_AV,filterset,maglim,outfile)
    if convert_h5:
        df = pd.read_table(outfile, sep='\s+', skipfooter=1, engine='python')
        df = df.rename(columns={'#Gc':'Gc'})
        for col in df.columns:
            if col not in NONMAG_COLS:
                df.rename(columns={col:'{}_mag'.format(col)},inplace=True)
        if not re.search('\.h5$', filename):
            h5file = '{}/{}.h5'.format(folder,filename)
        else:
            h5file = '{}/{}'.format(folder,filename)
        df.to_hdf(h5file,'df')
        with pd.HDFStore(h5file) as store:
            attrs = store.get_storer('df').attrs
            attrs.trilegal_args = {'version':trilegal_version,
                                   'ra':ra, 'dec':dec,
                                   'l':l,'b':b,'area':area,
                                   'AV':AV, 'sigma_AV':sigma_AV,
                                   'filterset':filterset,
                                   'maglim':maglim,
                                   'binaries':binaries}
        os.remove(outfile)

def trilegal_webcall(trilegal_version,l,b,area,binaries,AV,sigma_AV,filterset,maglim,
					 outfile):
    """Calls TRILEGAL webserver and downloads results file.
    :param trilegal_version:
        Version of trilegal (only tested on 1.6).
    :param l,b:
        Coordinates (galactic) for line-of-sight simulation.
    :param area:
        Area of TRILEGAL simulation [sq. deg]
    :param binaries:
        Whether to have TRILEGAL include binary stars.  Default ``False``.
    :param AV:
    	Extinction along the line of sight.
    :param sigma_AV:
        Fractional spread in A_V along the line of sight.
    :param filterset: (optional)
        Filter set for which to call TRILEGAL.
    :param maglim:
        Limiting magnitude in mag (by default will be 1st band of filterset)
        If want to limit in different band, then you have to
        change function directly.
    :param outfile:
        Desired output filename.
    """
    webserver = 'http://stev.oapd.inaf.it'
    args = [l,b,area,AV,sigma_AV,filterset,maglim,1,binaries]
    mainparams = ('imf_file=tab_imf%2Fimf_chabrier_lognormal.dat&binary_frac=0.3&'
    			  'binary_mrinf=0.7&binary_mrsup=1&extinction_h_r=100000&extinction_h_z='
    			  '110&extinction_kind=2&extinction_rho_sun=0.00015&extinction_infty={}&'
    			  'extinction_sigma={}&r_sun=8700&z_sun=24.2&thindisk_h_r=2800&'
    			  'thindisk_r_min=0&thindisk_r_max=15000&thindisk_kind=3&thindisk_h_z0='
    			  '95&thindisk_hz_tau0=4400000000&thindisk_hz_alpha=1.6666&'
    			  'thindisk_rho_sun=59&thindisk_file=tab_sfr%2Ffile_sfr_thindisk_mod.dat&'
    			  'thindisk_a=0.8&thindisk_b=0&thickdisk_kind=0&thickdisk_h_r=2800&'
    			  'thickdisk_r_min=0&thickdisk_r_max=15000&thickdisk_h_z=800&'
    			  'thickdisk_rho_sun=0.0015&thickdisk_file=tab_sfr%2Ffile_sfr_thickdisk.dat&'
    			  'thickdisk_a=1&thickdisk_b=0&halo_kind=2&halo_r_eff=2800&halo_q=0.65&'
    			  'halo_rho_sun=0.00015&halo_file=tab_sfr%2Ffile_sfr_halo.dat&halo_a=1&'
    			  'halo_b=0&bulge_kind=2&bulge_am=2500&bulge_a0=95&bulge_eta=0.68&'
    			  'bulge_csi=0.31&bulge_phi0=15&bulge_rho_central=406.0&'
    			  'bulge_cutoffmass=0.01&bulge_file=tab_sfr%2Ffile_sfr_bulge_zoccali_p03.dat&'
    			  'bulge_a=1&bulge_b=-2.0e9&object_kind=0&object_mass=1280&object_dist=1658&'
    			  'object_av=1.504&object_avkind=1&object_cutoffmass=0.8&'
    			  'object_file=tab_sfr%2Ffile_sfr_m4.dat&object_a=1&object_b=0&'
    			  'output_kind=1').format(AV,sigma_AV)
    cmdargs = [trilegal_version,l,b,area,filterset,1,maglim,binaries,mainparams,
    		   webserver,trilegal_version]
    cmd = ("wget -o lixo -Otmpfile --post-data='submit_form=Submit&trilegal_version={}"
    	   "&gal_coord=1&gc_l={}&gc_b={}&eq_alpha=0&eq_delta=0&field={}&photsys_file="
    	   "tab_mag_odfnew%2Ftab_mag_{}.dat&icm_lim={}&mag_lim={}&mag_res=0.1&"
    	   "binary_kind={}&{}' {}/cgi-bin/trilegal_{}").format(*cmdargs)
    complete = False
    while not complete:
        notconnected = True
        busy = True
        print("TRILEGAL is being called with \n l={} deg, b={} deg, area={} sqrdeg\n "
        "Av={} with {} fractional r.m.s. spread \n in the {} system, complete down to "
        "mag={} in its {}th filter, use_binaries set to {}.".format(*args))
        sp.Popen(cmd,shell=True).wait()
        if os.path.exists('tmpfile') and os.path.getsize('tmpfile')>0:
            notconnected = False
        else:
            print("No communication with {}, will retry in 2 min".format(webserver))
            time.sleep(120)
        if not notconnected:
            with open('tmpfile','r') as f:
                lines = f.readlines()
            for line in lines:
                if 'The results will be available after about 2 minutes' in line:
                    busy = False
                    break
            sp.Popen('rm -f lixo tmpfile',shell=True)
            if not busy:
                filenameidx = line.find('<a href=../tmp/') +15
                fileendidx = line[filenameidx:].find('.dat')
                filename = line[filenameidx:filenameidx+fileendidx+4]
                print("retrieving data from {} ...".format(filename))
                while not complete:
                    time.sleep(40)
                    modcmd = 'wget -o lixo -O{} {}/tmp/{}'.format(filename,webserver,filename)
                    modcall = sp.Popen(modcmd,shell=True).wait()
                    if os.path.getsize(filename)>0:
                        with open(filename,'r') as f:
                            lastline = f.readlines()[-1]
                        if 'normally' in lastline:
                            complete = True
                            print('model downloaded!..')
                    if not complete:
                        print('still running...')        
            else:
                print('Server busy, trying again in 2 minutes')
                time.sleep(120)
    sp.Popen('mv {} {}'.format(filename,outfile),shell=True).wait()
    print('results copied to {}'.format(outfile))
    
