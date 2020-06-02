import numpy as np
from random import seed
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage
import multiprocessing as mp
seed(0)


def make_random_array(m, n, th, p):
    if th != 0:
        data_3d = np.zeros((m, n, th))
        for i in range(m):
            for j in range(n):
                for k in range(th):
                    data_3d[i, j, k] = np.random.choice([0, 1], p=[1 - p, p])

        return data_3d

    else:
        data_2d = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                data_2d[i, j] = np.random.choice([0, 1], p=[1 - p, p])

        return data_2d


def make_distances_array(m, n):
    xx, yy = np.abs(np.meshgrid(np.arange(-m + 1, m), np.arange(-n + 1, n), sparse=False))
    zz = np.round(np.sqrt(xx ** 2 + yy ** 2), 2)

    return zz


def correlations_select_distance_pool(locs, delta_th_ref, padded_peaks_matrix, output, th, distance):
    th_locs = locs[2]

    if delta_th_ref % 36 == 0:
        print('   Completed ' + str(np.round(delta_th_ref / th * 100, 1)) + ' % of correlations')

    for k in range(th):
        ref_locs = np.where(np.abs(th_locs - k) == delta_th_ref)[0]
        n_peaks = len(ref_locs)

        for i in range(n_peaks):
            row, col = locs[0][ref_locs[i]], locs[1][ref_locs[i]]

            subset = padded_peaks_matrix[row - distance + 1:row + distance, col - distance + 1:col + distance, k]
            output[:subset.shape[0], :subset.shape[1], delta_th_ref] += subset

    return output


def correlations_select_distance_multiprocess(peaks_matrix, max_distance_nm, pixel_size, filename, min_counts=1, n_cores=8):
    # Initialization
    start_time = time.time()
    distance = int(max_distance_nm / pixel_size) + 1
    print('Calculating 3D autocorrelations up to distance {0} pixels ...'.format(distance))

    # Set parameters
    m, n, th = peaks_matrix.shape
    zz = make_distances_array(distance, distance)
    # Defining n_angles because in reality we want to find correlations in range of delta_th [0, 90]
    # We can sort values in this range because for delta_th > 90, the lowest angular misorientation is 180 - delta_th
    # (which will be below 90 degrees)
    n_angles = th
    # Output matrix for step 2
    output = np.zeros((zz.shape[0], zz.shape[1], n_angles))

    unique_distances = np.unique(zz)

    # Output array with sorted correlations with distance and delta_th
    correlations = np.zeros((unique_distances.shape[0], n_angles))

    # Pad peaks matrix
    padded_peaks_matrix = np.zeros((m + distance, n + distance, th))
    padded_peaks_matrix[distance:, distance:, :] = peaks_matrix

    # Step 1: find all locations where there is peak
    locs = np.where(padded_peaks_matrix > 0)

    # Step 2: pooling
    pool = mp.Pool(processes=n_cores)
    results = [pool.apply_async(correlations_select_distance_pool,
                                args=(locs, delta_th_ref, padded_peaks_matrix, output, th, distance))
               for delta_th_ref in range(th)]
    results = [p.get() for p in results]
    output = np.sum(results, axis=0)

    # Step 3: sort datapoints by distance and angle and normalize by number of pixels
    print('Sorting correlations ...')
    distances = []
    for i in range(unique_distances.shape[0]):
        map_locations = np.where(zz == unique_distances[i])
        counts = np.sum(output[map_locations[0], map_locations[1], :])
        if counts > min_counts:
            distances.append(i)
            for k in range(n_angles):
                plane = output[:, :, k]
                correlations[i, k] = np.sum(plane[map_locations])

    # Only select correlation rows with enough points (min_counts). Helps with doing statistics on enough samples.
    correlations = correlations[distances]
    corrs_df = pd.DataFrame(correlations, index=list(unique_distances[distances]))
    corrs_df.index = np.round(corrs_df.index * pixel_size, 2)

    # print('Done in ' + str(np.round(time.time() - start_time, 2)) + ' seconds.')

    dictionary = {}

    for d in corrs_df.index.values:
        dist = corrs_df.loc[d]
        angles = []
        for th in range(180):
            values = [th] * int(dist[th])
            angles += values

        dictionary[d] = np.array(angles)

    x = []
    y = []
    y_std = []

    for d in dictionary.keys():
        x.append(d)
        angles_radians = np.cos(dictionary[d] * np.pi / 180)
        y.append(np.mean(np.cos(dictionary[d] * np.pi / 180)))
        y_std.append(np.std(dictionary[d]))

    df = pd.DataFrame([x, y, y_std]).T
    df.columns=['L_nm', 'expected_cos_th', 'std']
    df.to_csv(filename, index=False)

    return df


def correlations_normalization_methods(corrs_df, z_score=True, binning=False):
    correlations = np.array(corrs_df)
    correlations = correlations / np.sum(correlations, axis=1).reshape(-1, 1)
    df = pd.DataFrame(correlations, index=list(corrs_df.index.values))

    if z_score:
        df = (df - df.mean()) / df.std()
    if binning:
        df.index = np.round(df.index.values, 0).astype(int)
        output = {}

        for d in df.index.values:
            row = df.loc[d]
            if len(row.shape) > 1:
                row = np.mean(row, axis=0)
            output[d] = row

        df = pd.DataFrame.from_dict(output, orient='index')

    return df


def plot_correlation_heatmap(corrs, d_max, xtick_separation=50, sigma=0, show=False, save_fig=''):
    """
    Generates 2D heatmap plot of correlations using seaborn package. This method
    :param corrs: correlations dataframe of counts or intensity as function of angles (columns) and distances (rows).
                  Note: * distances provided in nm!
    :param pixel_size: conversion from pixel to nm (i.e.  nm / pixel)
    :param d_max: maximum distance in nm to be plotted in heatmap
    :param sigma: breadth of gaussian filter
    :param show: boolean (default=False) to determine if figure should be shown in Jupyter/interface
    :param save_fig: string (default=empty '') to determine if figure should be saved. If string is not empty, this will
    be filename of saved figure.
    :return:
    """

    def find_ticks(indices, d_max, xtick_separation):
        ticks = np.arange(0, d_max + 1, step=xtick_separation)
        locs, values = [], []
        for tick in ticks:
            idx = np.argmin(np.abs(indices - tick))
            locs.append(idx)
            values.append(int(np.round(indices[idx], 0)))

        return locs, values

    corrs_filtered = scipy.ndimage.gaussian_filter(corrs, sigma=sigma)

    corrs_dist = pd.DataFrame(corrs_filtered, index=corrs.index)
    tick_locs, tick_vals = find_ticks(corrs_dist.index, d_max, xtick_separation)

    ax = sns.heatmap(corrs_dist.loc[:d_max].T, cmap='RdBu_r', robust=True, cbar=True,
                     cbar_kws={'label': 'Standard Score'}, square=False, yticklabels=10, center=0)

    ax.set_xlabel('|Δd| / nm', fontsize=14)
    ax.set_ylabel('|Δθ| / degrees', fontsize=14)
    ax.set_ylim([0, 90.5])
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_vals, rotation=0)
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def plot_example_linecuts(corrs, d_vals, sigma=0, show=False, save_fig=''):
    """
    Generates 1D scatter plot with correlations at various distances. Useful for visualization.
    :param corrs:
    :param d_vals:
    :param sigma:
    :param show:
    :param save_fig:
    :return:
    """
    corrs_dist = pd.DataFrame(scipy.ndimage.gaussian_filter(corrs, sigma=sigma))
    corrs_dist.index = corrs.index

    plt.figure()
    for d in d_vals:
        plt.scatter(np.arange(91), corrs_dist.loc[d], label=str(d), s=5)
        plt.plot(np.arange(91), corrs_dist.loc[d], linewidth=0.2)

    plt.legend(title='|Δd| / nm', bbox_to_anchor=(1.2, 1), fontsize=14)
    plt.xlabel('|Δθ| / degrees', fontsize=14)
    plt.ylabel('Standard Score', fontsize=14)
    plt.xlim([-2, 90])
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

