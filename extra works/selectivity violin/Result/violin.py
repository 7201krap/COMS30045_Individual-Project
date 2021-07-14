#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:07:18 2021

@author: qq20468
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import matplotlib as mpl

#%%


def set_color(p, axes, color, facealpha, edgewidth):
    for pc in p['bodies']:
        faceclr = mpl.colors.colorConverter.to_rgba(color, alpha=facealpha)
        pc.set_facecolor(faceclr)
        pc.set_alpha(facealpha)
        path = pc.get_paths()[0]
        s = mpl.patches.PathPatch(path, linewidth=edgewidth, edgecolor=color, fill=False)
        axes.add_patch(s)
    p['cmeans'].set_facecolor(color)
    p['cmeans'].set_edgecolor(color)


def get_max_density(arr):
    x=arr
    density = gaussian_kde(x)
    minval = np.min(x)
    maxval=np.max(x)
    x_vals = np.linspace(minval,maxval,200) # Specifying the limits of our data
    density.covariance_factor = lambda : .5 #Smoothing parameter
    density._compute_covariance()
    max_density = np.max(density(x_vals))
    return max_density

#%%
# generate some normally distributed data:
var1 = [np.random.normal(1, 0.2, 200) for i in range(4)]
var2 = [np.random.normal(1.1, 0.1, 300) for i in range(4)]
norm = var1
norm.extend(var2)

# make a dummy data frame:
df = pd.DataFrame({'condition':['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                       'x_locations':[1,2,1,2,1,2,1,2],
                       'ccmas': norm})
#%%

target  ='ccmas' # target variable to plot

# np arrays cannot be aggregated if you want to get all ccmas values for multiple models
if type((df[target].sample()).values[0])==np.ndarray:
    df[target] = df[target].apply(lambda x: x.tolist())

# we want the violins to be equally spaced,
# so it's best to rank the x_location variabel we're interested in
# e.g. these could be hidden layer sizes
df['x_locations_rank'] = df['x_locations'].rank(method='dense')
# we also want to rank the condition we're plotting (in my case that's the optimizer)
df['condition_rank'] = df['condition'].rank(method='dense').apply(lambda x: int(x)-1)

sel_df = df.groupby(['x_locations', 'x_locations_rank', 'condition', 'condition_rank'], as_index = False, observed=True).agg(sel = (target, sum))
fig, ax = plt.subplots(dpi=80)

labels = []

# violinplots aren't plotted on the same width scale by default.
# in order to have comparable curves, we need to find the maximum density value
# in our data, and scale each violins width accordingly.
sel_df['max_density'] = sel_df['sel'].apply(lambda x: get_max_density(x))
sel_df['X'] = sel_df['max_density'].max()
sel_df['violin_scale'] = sel_df['max_density']/sel_df['X']

# outline colors
outlines = ['royalblue', 'mediumseagreen', 'maroon', 'darkslategray', 'gold']


grp = sel_df.groupby(['condition', 'condition_rank']) # plot each condition at once
for n, row in grp:
    cond, color_id = n
    print(cond)
    p=ax.violinplot(row.sel, positions = row['x_locations_rank'],
                     vert=True, widths = row['violin_scale'],
                     showmeans=True, showextrema=False,
                     points=100, bw_method=0.2)
    set_color(p, ax, outlines[color_id], facealpha=0.05, edgewidth=1.4)
    labels.append((mpl.patches.Patch(color=outlines[color_id]), cond))


lgd = plt.legend(*zip(*labels), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticklabels(sel_df['x_locations'].unique())
ax.set_xticks(sel_df['x_locations_rank'].unique())
plt.xlabel('x_locations'); plt.ylabel(target);
plt.show()
