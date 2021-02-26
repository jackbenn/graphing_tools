import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix

'''
TODO
  xlim instead of xmin/xmax?
  set colors
  should line up better with integers by default
'''


def _find_aligned_bins(xmin, xmax, bins):
    diff = xmax - xmin
    target_binsize = diff / bins

    log_target_binsize = np.log10(target_binsize)
    base_unit = 10 ** np.floor(log_target_binsize)
    trial_units = [base_unit,
                   base_unit * 2,
                   base_unit * 5,
                   base_unit * 10]
    errors = []
    for trial_unit in trial_units:
        trial_min_per_unit = xmin // trial_unit  # round down to trial_unit
        trial_max_per_unit = xmax // trial_unit + (xmax % trial_unit != 0)  # round up
        trial_bins = trial_max_per_unit - trial_min_per_unit
        errors.append(np.log(bins / trial_bins) ** 2)  # how far of we are in number of bins

    unit = trial_units[np.argmin(errors)]
    return unit, np.arange(xmin // unit,
                           xmax // unit + (xmax % unit != 0) + 1,
                           1) * unit


def multihist(x, y=None,
              bins=None, binsize=None,
              align=False,
              xmin=None, xmax=None, ymax=None,
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
    align:  align bins on multiples of integers
    binsize:float: size of bins (overrides bins, align)

    xmin:   lower limit (or None to set to min of data)
    xmax:   upper limit (or None to set to max of data)
    ymax:    upper limit of y

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

    xbinmin, xbinmax = xc.min(), xc.max()
    if xmin is not None:
        xbinmin = min(xbinmin, xmin)
    if xmax is not None:
        xbinmax = max(xbinmax, xmax)

    if binsize is None:
        if bins is None:
            bins = 20
        if align:
            binsize, binarray = _find_aligned_bins(xbinmin,
                                                   xbinmax,
                                                   bins)
        else:
            binsize = (xc.max() - xc.min())/bins
            binarray = np.linspace(xbinmin, xbinmax, bins + 1)
    else:
        binarray = np.arange(xbinmin,
                             np.nextafter(xbinmax, xbinmax+1),
                             binsize)

    if kde:
        xvals = np.linspace(xc.min(), xc.max(), 100)
        kde_scale = 1

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

    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(top=ymax)
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


def plot_confusion_matrix(ax,
                          y_true,
                          y_pred,
                          color='blue',
                          counts=False,
                          grid=False,
                          area=True,
                          transpose=True,
                          shape='square',
                          normalization="maximum"):
    """
    Parameters
    ----------
    ax : matplotlib axis on which to draw graph.
    y_true : list-like object
        actual labels.
    y_pred : list-like object of the same size as y_true
        predictions based on model.
    color : str, detault: 'blue'
        color of squares
    counts : bool, default: False
        show the counts inside each square (colored white)
    grid : bool, default False
        whether to draw a grid.
    area : bool, default: True
        if True, the area of the square is proportional to the value.
        If False, the length of a side is used.
    shape : 'square' or 'circle'
        the shape of the object in the grid
    transpose : bool, default: True
        transpose from sklearn standard. if True, show the actual values
        along the vertical axis and predictions along horizontal
    normalization : string, default: 'maximum'
        How to normalize the values.
        'maximum' : normalize all values so the largest value is 1.
        'prediction' or 'precision' : normalize so that the sum of each
            prediction is 1, so the values represent the precisions
            (along columns if transpose==True)
        'true' or 'recall': normalize so that the sum of each true value is 1,
            so the values represent the recalls (along rows if transpose==True)
    """
    cnts = confusion_matrix(y_true, y_pred).astype(float)
    if normalization == 'maximum':
        cm = cnts / cnts.max()
    elif normalization == 'prediction' or normalization == 'precision':
        cm = cnts / cnts.sum(axis=0, keepdims=True)
    elif normalization == 'true' or normalization == 'recall':
        cm = cnts / cnts.sum(axis=1, keepdims=True)
    else:
        raise ValueError("`normalization` should be one of: " +
                         "`maximum`, `true`, `prediction`")
    n = cm.shape[0]

    if transpose:
        cnts = cnts.transpose()
        cm = cm.transpose()

    labels = np.unique(y_true)
    tics = np.arange(n)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(tics + 0.5, minor=True)
        axis.set(ticks=tics, ticklabels=labels)
    ax.tick_params(axis='both', which='major', length=0)

    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, n-0.5)

    ax.invert_yaxis()
    ax.grid(grid, which='minor')

    labels = ['true', 'prediction']
    if transpose:
        labels.reverse()
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    if area:
        cm **= 0.5

    for i in range(n):
        for j in range(n):
            size = cm[i, j]
            if shape == 'square':
                square = matplotlib.patches.Rectangle((i - size/2, j - size/2),
                                                      size, size, color=color)
            elif shape == 'circle':
                square = matplotlib.patches.Circle((i, j), size/2, color=color)
            ax.add_patch(square)
            if counts:
                ax.text(i, j, str(int(cnts[i, j])),
                        horizontalalignment='center',
                        verticalalignment='center', color='white')


def plot_discrete_cdf(ax, pmf, margin=1/5):
    """
    Plot the CDF of discrete random variable.
    """
    keys = sorted(pmf)
    diff = keys[-1] - keys[0]
    keys.insert(0, keys[0] - diff * margin)
    keys.append(keys[-1] + diff * margin)

    cumulative_prob = 0
    for i in range(len(keys)-1):
        if i > 0:
            ax.scatter([keys[i]],
                       [cumulative_prob],
                       s=100,
                       facecolor='w',
                       edgecolor='k',
                       zorder=3)
            cumulative_prob += pmf[keys[i]]
            ax.scatter([keys[i]],
                       [cumulative_prob],
                       s=100,
                       facecolor='k',
                       edgecolor='k',
                       zorder=3)
        ax.plot([keys[i], keys[i+1]],
                [cumulative_prob, cumulative_prob], 'k')
        ax.set_ylabel('CDF')
