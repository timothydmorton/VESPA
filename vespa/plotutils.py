
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
except ImportError:
    plt = None
    np = None
    
    
def setfig(fig=None,**kwargs):
    """
    Sets figure to 'fig' and clears; if fig is 0, does nothing (e.g. for overplotting)

    if fig is None (or anything else), creates new figure
    
    I use this for basically every function I write to make a plot.
    I give the function
    a "fig=None" kw argument, so that it will by default create a new figure.

    .. note::

      There's most certainly a better, more object-oriented
      way of going about writing functions that make figures, but
      this was put together before I knew how to think that way,
      so this stays for now as a convenience.
      
    """
    if fig:
        plt.figure(fig,**kwargs)
        plt.clf()
    elif fig==0:
        pass
    else:
        plt.figure(**kwargs)

def plot2dhist(xdata,ydata,cmap='binary',interpolation='nearest',
               fig=None,logscale=True,xbins=None,ybins=None,
               nbins=50,pts_only=False,**kwargs):
    """Plots a 2d density histogram of provided data

    :param xdata,ydata: (array-like)
        Data to plot.

    :param cmap: (optional)
        Colormap to use for density plot.

    :param interpolation: (optional)
        Interpolation scheme for display (passed to ``plt.imshow``).

    :param fig: (optional)
        Argument passed to :func:`setfig`.

    :param logscale: (optional)
        If ``True`` then the colormap will be based on a logarithmic
        scale, rather than linear.

    :param xbins,ybins: (optional)
        Bin edges to use (if ``None``, then use ``np.histogram2d`` to
        find bins automatically).

    :param nbins: (optional)
        Number of bins to use (if ``None``, then use ``np.histogram2d`` to
        find bins automatically).

    :param pts_only: (optional)
        If ``True``, then just a scatter plot of the points is made,
        rather than the density plot.

    :param **kwargs:
        Keyword arguments passed either to ``plt.plot`` or ``plt.imshow``
        depending upon whether ``pts_only`` is set to ``True`` or not.
        
    """

    setfig(fig)
    if pts_only:
        plt.plot(xdata,ydata,**kwargs)
        return

    ok = (~np.isnan(xdata) & ~np.isnan(ydata) & 
           ~np.isinf(xdata) & ~np.isinf(ydata))
    if ~ok.sum() > 0:
        logging.warning('{} x values and {} y values are nan'.format(np.isnan(xdata).sum(),
                                                                     np.isnan(ydata).sum()))
        logging.warning('{} x values and {} y values are inf'.format(np.isinf(xdata).sum(),
                                                                     np.isinf(ydata).sum()))

    if xbins is not None and ybins is not None:
        H,xs,ys = np.histogram2d(xdata[ok],ydata[ok],bins=(xbins,ybins))
    else:
        H,xs,ys = np.histogram2d(xdata[ok],ydata[ok],bins=nbins)        
    H = H.T

    if logscale:
        H = np.log(H)

    extent = [xs[0],xs[-1],ys[0],ys[-1]]
    plt.imshow(H,extent=extent,interpolation=interpolation,
               aspect='auto',cmap=cmap,origin='lower',**kwargs)
    
