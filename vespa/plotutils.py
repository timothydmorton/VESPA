import matplotlib.pyplot as plt
import numpy as np
import logging

def setfig(fig=None,**kwargs):
    """
    Sets figure to 'fig' and clears; if fig is 0, does nothing (e.g. for overplotting)

    if fig is None (or anything else), creates new figure
    
    I use this for basically every function I write to make a plot.  I give the function
    a "fig=None" kw argument, so that it will by default create a new figure.
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

    xdata,ydata : array_like
        Data to plot

    cmap : string, optional
        Colormap to use for density plot

    interpolation : string, optional
        Interpolation scheme for display (passed to ``plt.imshow``)

    fig : None or int, optional
        Argument passed to ``setfig`` function.

    logscale : bool, optional
        If ``True`` then the colormap will be based on a logarithmic
        scale, rather than linear.

    xbins, ybins : ``None`` or array-like, optional
        Bin edges to use (if ``None``, then use ``np.histogram2d`` to
        find bins automatically.

    nbins : ``None`` or int, optional
        Number of bins to use (if ``None``, then use ``np.histogram2d`` to
        find bins automatically.

    pts_only : bool
        If ``True``, then just a scatter plot of the points is made,
        rather than the density plot.

    kwargs :
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
    
