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


def distance_correlations_2d(data_2d, zz, distance_range):
    # Initialization
    start_time = time.time()
    print('Calculating 2D correlations ...')

    # Set parameters
    m, n = data_2d.shape
    output = np.zeros(zz.shape)
    center_row = m - 1
    center_col = n - 1

    unique_distances = np.unique(zz)
    unique_distances = unique_distances[unique_distances <= distance_range]

    correlations = []
    dists = []

    # Step 1: find all locations where there is peak
    locs = np.where(data_2d > 0)
    n_peaks = len(locs[0])

    # Step 2: go over all points and add up according to distance difference
    for case in range(n_peaks):
        row, col = locs[0][case], locs[1][case]

        # shift matrix
        diff_row = center_row - row
        diff_col = center_col - col

        output[diff_row:diff_row + m, diff_col:diff_col + m] += data_2d

    plt.imshow(output)
    plt.title('Sum of all correlations')
    plt.show()

    # Step 3: sort datapoints by distance and normalize by number of pixels
    for i in range(unique_distances.shape[0]):
        map_locations = np.where(zz == unique_distances[i])
        counts = np.sum(output[map_locations])
        correlations.append(counts / len(output[map_locations]))
        dists.append(unique_distances[i])

    # Step 4: generate 2d corrrelations dataframe, rows are different distances
        corrs_df = pd.DataFrame(list(correlations), index=dists)

    print('Done in ' + str(np.round(time.time() - start_time, 2)) + ' seconds')

    return corrs_df


def correlations_3d(data_3d, zz, pixel_size, min_counts=1000, z_score=True, bin_rows=True):
    # Initialization
    start_time = time.time()
    print('Calculating 3D correlations ...')

    # Set parameters
    m, n, th = data_3d.shape
    # Defining n_angles because in reality we want to find correlations in range of delta_th [0, 90]
    # We can sort values in this range because for delta_th > 90, the lowest angular misorientation is 180 - delta_th
    # (which will be below 90 degrees)
    n_angles = int(th/2) + 1
    # Output matrix for step 2
    output = np.zeros((zz.shape[0], zz.shape[1], n_angles))

    unique_distances = np.unique(zz)

    # Output array with sorted correlations with distance and delta_th
    correlations = np.zeros((unique_distances.shape[0], n_angles))

    center_row = m - 1
    center_col = n - 1

    # Step 1: find all locations where there is peak
    locs = np.where(data_3d > 0)
    th_locs = locs[2]    # th_locs is the plane of datacube with certain theta orientation

    # Step 2: go over all points and add up according to distance and angle difference
    for delta_th_ref in range(th):
        if delta_th_ref % 20 == 0:
            print('   Completed ' + str(np.round(delta_th_ref / th * 100, 1)) + ' % of correlations')
        for k in range(th):
            ref_locs = np.where(np.abs(th_locs - k) == delta_th_ref)[0]
            n_peaks = len(ref_locs)

            for i in range(n_peaks):
                row, col = locs[0][ref_locs[i]], locs[1][ref_locs[i]]
                diff_row = center_row - row
                diff_col = center_col - col
                if delta_th_ref > 90:
                    delta_th_ref = 180 - delta_th_ref

                output[diff_row:diff_row + m, diff_col:diff_col + m, delta_th_ref] += data_3d[:, :, k]

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

    # Step 4: divide total volume of datacube to complete volume average normalization
    correlations = correlations / np.sum(correlations, axis=1).reshape(-1, 1)

    # Step 5: generate 3d corrrelations dataframe, rows are different distances and columns delta_th
    corrs_df = pd.DataFrame(correlations, index=list(unique_distances[distances]))
    corrs_df.index = np.round(corrs_df.index * pixel_size, 2)

    # Deal with data normalization conditions
    # Standarize values to have mean zero and 1std of +/- one (Z_socre). Doing it for each column
    if z_score:
        corrs_df = (corrs_df - corrs_df.mean()) / corrs_df.std()

    # Bin (get mean) of distance rows that are very close. This helps with smoothing of data at large distance values
    if bin_rows:
        corrs_df.index = np.round(corrs_df.index.values, 0).astype(int)
        output = {}

        for d in corrs_df.index.values:
            row = corrs_df.loc[d]
            if len(row.shape) > 1:
                row = np.mean(row, axis=0)
            output[d] = row

        corrs_df = pd.DataFrame.from_dict(output, orient='index')

    print('Done in ' + str(np.round(time.time() - start_time, 2)) + ' seconds.')

    return corrs_df


def correlations_pool(locs, delta_th_ref, data, output, m, n, th):
    th_locs = locs[2]

    if delta_th_ref % 20 == 0:
        print('   Completed ' + str(np.round(delta_th_ref / th * 100, 1)) + ' % of correlations')

    for k in range(th):
        ref_locs = np.where(np.abs(th_locs - k) == delta_th_ref)[0]
        n_peaks = len(ref_locs)

        for i in range(n_peaks):
            row, col = locs[0][ref_locs[i]], locs[1][ref_locs[i]]
            diff_row = m - 1 - row
            diff_col = n - 1 - col
            if delta_th_ref > 90:
                delta_th_ref = 180 - delta_th_ref

            output[diff_row:diff_row + m, diff_col:diff_col + m, delta_th_ref] += data[:, :, k]
    return output


def correlations_multiprocess(data, zz, pixel_size, min_counts=1000, z_score=True, bin_rows=True):
    # Initialization
    start_time = time.time()
    print('Calculating 3D correlations ...')

    # Set parameters
    m, n, th = data.shape
    # Defining n_angles because in reality we want to find correlations in range of delta_th [0, 90]
    # We can sort values in this range because for delta_th > 90, the lowest angular misorientation is 180 - delta_th
    # (which will be below 90 degrees)
    n_angles = int(th/2) + 1
    unique_distances = np.unique(zz)

    output = np.zeros((zz.shape[0], zz.shape[1], n_angles))

    # Output array with sorted correlations with distance and delta_th
    correlations = np.zeros((unique_distances.shape[0], n_angles))

    # Step 1: find all locations where there is peak
    locs = np.where(data > 0)

    # Step 2: pooling
    pool = mp.Pool(processes=8)
    results = [pool.apply_async(correlations_pool, args=(locs, delta_th_ref, data, output, m, n, th)) for delta_th_ref in
               range(th)]
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

    # Step 4: divide total volume of datacube to complete volume average normalization
    correlations = correlations / np.sum(correlations, axis=1).reshape(-1, 1)

    # Step 5: generate 3d corrrelations dataframe, rows are different distances and columns delta_th
    corrs_df = pd.DataFrame(correlations, index=list(unique_distances[distances]))
    corrs_df.index = np.round(corrs_df.index * pixel_size, 2)

    # Deal with data normalization conditions
    # Standarize values to have mean zero and 1std of +/- one (Z_socre). Doing it for each column
    if z_score:
        corrs_df = (corrs_df - corrs_df.mean()) / corrs_df.std()

    # Bin (get mean) of distance rows that are very close. This helps with smoothing of data at large distance values
    if bin_rows:
        corrs_df.index = np.round(corrs_df.index.values, 0).astype(int)
        output = {}

        for d in corrs_df.index.values:
            row = corrs_df.loc[d]
            if len(row.shape) > 1:
                row = np.mean(row, axis=0)
            output[d] = row

        corrs_df = pd.DataFrame.from_dict(output, orient='index')

    print('Done in ' + str(np.round(time.time() - start_time, 2)) + ' seconds.')

    return corrs_df


def correlations_intensity(data, zz, pixel_size, min_counts, z_score=True, bin_rows=True):
    # Initialization
    start_time = time.time()
    print('Calculating 3D correlations ...')

    # Set parameters
    m, n, th = data.shape
    # Defining n_angles because in reality we want to find correlations in range of delta_th [0, 90]
    # We can sort values in this range because for delta_th > 90, the lowest angular misorientation is 180 - delta_th
    # (which will be below 90 degrees)
    n_angles = int(th/2) + 1
    # Output matrix for step 2
    output = np.zeros((zz.shape[0], zz.shape[1], n_angles))

    unique_distances = np.unique(zz)
    # unique_distances = unique_distances[unique_distances <= distance_range]   # Need to think about distance range

    # Output array with sorted correlations with distance and delta_th
    correlations = np.zeros((unique_distances.shape[0], n_angles))

    center_row = m - 1
    center_col = n - 1

    # Step 1: find all locations where there is peak
    locs = np.where(data > 0)
    th_locs = locs[2]    # th_locs is the plane of datacube with certain theta orientation

    # Step 2: go over all points and add up according to distance and angle difference
    for delta_th_ref in range(th):
        if delta_th_ref % 20 == 0:
            print('   Completed ' + str(np.round(delta_th_ref / th * 100, 1)) + ' % of correlations')
        for k in range(th):
            ref_locs = np.where(np.abs(th_locs - k) == delta_th_ref)[0]
            n_peaks = len(ref_locs)

            for i in range(n_peaks):
                row, col, angle = locs[0][ref_locs[i]], locs[1][ref_locs[i]], locs[2][ref_locs[i]]
                intensity = data[row, col, angle]
                diff_row = center_row - row
                diff_col = center_col - col
                if delta_th_ref > 90:
                    delta_th_ref = 180 - delta_th_ref

                output[diff_row:diff_row + m, diff_col:diff_col + m, delta_th_ref] += data[:, :, k] * intensity

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

    # Step 4: divide total volume of datacube to complete volume average normalization
    correlations = correlations / np.sum(correlations, axis=1).reshape(-1, 1)

    # Step 5: generate 3d corrrelations dataframe, rows are different distances and columns delta_th
    corrs_df = pd.DataFrame(correlations, index=list(unique_distances[distances]))
    corrs_df.index = np.round(corrs_df.index * pixel_size, 2)

    # Deal with data normalization conditions
    # Standarize values to have mean zero and 1std of +/- one (Z_socre). Doing it for each column
    if z_score:
        corrs_df = (corrs_df - corrs_df.mean()) / corrs_df.std()

    # Bin (get mean) of distance rows that are very close. This helps with smoothing of data at large distance values
    if bin_rows:
        corrs_df.index = np.round(corrs_df.index.values, 0).astype(int)
        output = {}

        for d in corrs_df.index.values:
            row = corrs_df.loc[d]
            if len(row.shape) > 1:
                row = np.mean(row, axis=0)
            output[d] = row

        corrs_df = pd.DataFrame.from_dict(output, orient='index')

    print('Done in ' + str(np.round(time.time() - start_time, 2)) + ' seconds.')

    return corrs_df


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
                     cbar_kws={'label': 'Standard Score'}, square=False, yticklabels=10)

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

    plt.legend(title='|Δd| / nm', bbox_to_anchor=(1.2, 1))
    plt.xlabel('|Δθ| / degrees', fontsize=14)
    plt.ylabel('Standard Score', fontsize=14)
    plt.xlim([-2, 90])
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()