# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:19:13 2018
Plotting function container

@author: danilo
"""
import numpy as np
from matplotlib import pyplot as plt

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    return ax

def figure_layout(x_in, y_in):
    
    fontsize_main = 15./4.*y_in
    fontsize = 12./4.*y_in
    linewidth_main = 1.5/4.*y_in
    linewidth = .8/6.*y_in
    markersize_main = 10./4.*y_in
    markersize = 8./4.*y_in
    return fontsize_main, fontsize, linewidth_main, linewidth, markersize_main, markersize

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
