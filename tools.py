import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

'''
TODO
  xlim instead of xmin/xmax?
  kde
  set colors
  call as list of arrays
'''

def multihist(x, y=None,
              bins=None, binsize=None,
              xmin=None, xmax=None, top=None,
              kde=None,
              density=True, alpha=0.2, figsize=(12, 8), title=None,
              ax=None):
    '''
    INPUT:
    x:      numpy array; point of a distribution
    y:      numpy array of the same length; will plot histogram
            for each unique value of y.
            If None (the default), x should be a list of arrays;
            a histogram with be created for each, labeled sequentially.
    bins:   int; number of bins in histogram
    binsize:float: size of bins (overrides bins)
 
    xmin:   lower limit (or None to set to min of data)
    xmax:   upper limit (or None to set to max of data)
    top:    upper limit of y

    density: normalize number of elements in each class
    kde:    add kde plot
    alpha:  float; opacity; pass to matplotlib
    figsize:tuple; width and height of figure; pass to matplotlib
    title:  str; title of plot
    ax:     axis on which to plot histograms
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title)
    if y is None:
        y = np.concatenate([[i]*len(subx) for i, subx in enumerate(x)])
        x = np.concatenate(x)
    if xmin is None and xmax is None:
        xc = x
    else:
        xc = np.clip(x, a_min=xmin, a_max=xmax)

    if binsize is None:
        if bins is None:
            bins = 20.
        # maybe there should be logic around trying to make these integers
        binsize = (xc.max() - xc.min())/(bins+1)
    binarray = np.arange(xc.min(),
                         xc.max() + binsize,
                         binsize)

    # We need to get the default color cycle to get the same color
    # for the hist and kde line.
    props = plt.rcParams['axes.prop_cycle']

    for yval, prop in zip(np.unique(y), props):
        color = prop['color']

        h = ax.hist(list(xc[y == yval]),
                    alpha=alpha,
                    bins=binarray,
                    density=density,
                    label=str(yval),
                    color=color)
        if kde:
            kde_func = stats.gaussian_kde(xc[y == yval])
            if not density:
                kde_scale = np.sum(y == yval) * binsize
            ax.plot(xvals, kde_scale * kde_func(xvals), color=color)
        
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(top=top)
    ax.legend()


def axdline(ax, slope, intercept, **kwargs):
    """
    Add a diagonal line across the axis, similar to axhline and axvline.
    Parameters
    ----------
    slope : scalar
        slope in data coordinates of the diagonal line

    intercept : scalar
        y intercept in data coordinates of the diagonal line

    Returns
    -------
    line : :class:`~matplotlib.lines.Line2D`
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    line = ax.plot(xlim,
                   (xlim[0]*slope + intercept,
                    xlim[1]*slope + intercept),
                   **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return line[0]
