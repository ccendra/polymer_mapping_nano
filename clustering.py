import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib import colors
import pandas as pd

import plot_functions as plot


def find_clusters(data, threshold, min_cluster_size, max_separation):
    """Computes (x, y) map of clusters for input numpy array (data) as specified by argument conditions. Returns
    a numbered map of clusters and averaged orientation map of each cluster.
    Arguments:
        data: 3D peaks array with 1 at (x, y, th) locations with peak center and 0 otherwise
        threshold: maximum angular deviation (+/-) relative to average orientation of a cluster allowed for
                    considering a neighboring point belongs to the same cluster
        min_cluster_size: minimum size of a cluster (in total number of pixels) for it to be considered a cluster
        max_separation: maximum separation allowed between neighboring points with similar orientation for them to be
                    considered part of the same cluster.
    Returns:
        cluster_map: 2D numpy array with cluster number at each (x,y) formed based on arguments above. NaN values in
                (x, y) locations where no cluster was found.
        orientation_map: 2D numpy array with average orientation angle (average theta) for found clusters. NaN values
                in (x, y) locations where no cluster was found.
        cluster_orientation_std: 2D numpy array with calculated 1 standard deviation from average orientation angle for
                found clusters. Nan values in (x, y) locations where no cluster was found.
        cluster_properties: dictionary storing the extracted information for each cluster. To be used for eithter
                troubleshooting or more analysis.
    """
    # Initialize outputs
    m, n, _ = data.shape

    rows, cols, angles = np.where(data > 0)
    k = int(np.max(np.sum(data, axis=2)))

    # Make input arrays
    input_array = np.full(shape=(m, n, k), fill_value=-1, dtype=np.int16)
    # input_counter = np.zeros((m, n), dtype=np.int16)
    input_counter = np.full(shape=(m, n), fill_value=-1, dtype=np.int16)

    # Fill input arrays with values
    for i in range(len(rows)):
        row, col, th = rows[i], cols[i], angles[i]
        input_array[row, col, input_counter[row, col]] = th
        input_counter[row, col] += 1

    estimation = int(m * n * 2 / min_cluster_size)
    output = np.full(shape=(estimation, m, n), fill_value=-1, dtype=np.int16)
    # Cluster map
    cluster_map = np.full(shape=(k, m, n), fill_value=-1, dtype=np.int16)
    # Dictionary to track properties for each cluster
    cluster_properties = {}
    # Tracker for cluster number
    cluster_number = 0

    # Keep track of computation time
    start_time = time.time()
    num_pixels = []

    for row in range(m):
        if row % 10 == 0:
            print('     ...Row: ', row)
        for col in range(n):
            for i in range(input_counter[row, col]):
                theta = input_array[row, col, i]
                if input_counter[row, col] > 0:
                    theta_array, x_coords, y_coords = try_forming_cluster(input_array, input_counter, theta, threshold,
                                                                         row, col, max_separation)
                    # Determine if found cluster is large enough to be considered a cluster. If yes, save it in outputs.
                    if len(x_coords) >= min_cluster_size:
                        cluster_map[input_counter[x_coords, y_coords]-1, x_coords, y_coords] = cluster_number
                        output[cluster_number, x_coords, y_coords] = theta_array
                        median_theta = np.median(theta_array)
                        cluster_properties[cluster_number] = \
                            {'median_theta': median_theta, 'MAD_theta': np.median(np.abs(theta_array - median_theta)),
                             'number_pixels': len(x_coords)}
                        cluster_number += 1
                        input_counter[x_coords, y_coords] -= 1
                        num_pixels.append(len(x_coords))

    output = output[:cluster_number, :, :]
    print('     ...Formed {0} clusters'.format(cluster_number))
    if cluster_number > 0:
        print('     ...Mean size of clusters is {0} pixels'.format(np.round(np.mean(num_pixels), 2)))
    print('     ...Clustering time(s): ', np.round((time.time() - start_time), 1))

    return cluster_map, output, cluster_properties


def try_forming_cluster(input_array, input_counter, theta, threshold, start_row, start_col, separation):
    """Iterates over data array at a certain starting point (start row, start col) and searches for neighboring points
    that can form a cluster. Returns a single cluster and list of orientation values.
    Arguments:
        data: 2D numpy array providing maximum orientation theta at each (x, y)
        threshold: maximum angular deviation (+/-) relative to average orientation of a cluster allowed for
                    considering a neighboring point belongs to the same cluster
        start_row: initial row in data to start searching for cluster
        start_col: initial column in data to start searching for cluster
        separation: maximum separation allowed between neighboring points with similar orientation for them to be
                    considered part of the same cluster.
        cluster_map: 2D numpy array with cluster number at each (x,y) formed based on arguments above. NaN values in
                (x, y) locations where no cluster was found.
    Returns:
        theta_list: list of theta values at each point belonging to cluster
        x_coords, y_coords = list of x and y coordinates, respectively, of points belonging to the cluster
    """
    m, n, _ = input_array.shape
    # Initialization
    theta_list = np.array([theta])
    x_coords = np.array([start_row])
    y_coords = np.array([start_col])

    for row in range(start_row, m):
        if row == start_row:
            col_start = start_col + 1
        else:
            col_start = 0

        if (np.abs(x_coords - row) > separation).all():
            break

        for col in range(col_start, n):
            thetas = input_array[row, col]  # get possible theta values
            if (~np.isnan(thetas)).any():
                closest_theta_index = np.nanargmin(np.abs(np.mean(theta_list) - thetas))
                th = input_array[row, col, closest_theta_index]
                if input_counter[row, col] > 0 and point_belongs_to_cluster(th, x_coords, y_coords, theta_list, threshold,
                                                                            row, col, separation):
                    x_coords = np.append(x_coords, row)
                    y_coords = np.append(y_coords, col)
                    theta_list = np.append(theta_list, th)  # Update orientation list

    return theta_list, x_coords, y_coords


def point_belongs_to_cluster(value, x_coords, y_coords, theta_list, threshold, row, col, separation):
    """Determine if new point is neighbor to the cluster.
     Arguments:
         value: theta value at [row, col]
         x_coords: list of row coordinates of points already in cluster
         y_coords: list of col coordinates of points already in cluster
         theta_list: list of theta values corresponding to points already assigned to cluster
         threshold: maximum allowed angle misalignment
         row: current row of point being evaluated
         col: current column of point being evaluated
         separation: maximum separation allowed for point (row, col) to be considered part of the same cluster
    Returns:
        True/False: T/F of whether point (x,y) belongs to cluster.
    """
    if value >= 0:
        if len(x_coords) == 0:
            return True
        else:
            if np.abs(value - np.median(theta_list)) <= threshold:
                distance = np.sqrt((np.array(x_coords) - row) ** 2 + (np.array(y_coords) - col) ** 2) <= separation
                if distance.any():
                    return True
    return False


def plot_cluster_map(output, angles, xlength, ylength, save_fig='', show_plot=False):

    cmap = colors.ListedColormap(plot.get_colors(angles + 90))
    fig = plt.figure(figsize=(10, 10))
    for i in range(output.shape[0]):
        plt.imshow(output[i, :, :], vmin=0, vmax=180, alpha=0.5, cmap=cmap, extent=[0, xlength, 0, ylength])
    # plt.xticks([])
    # plt.yticks([])
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300, transparent=True)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def cumulative_step_histogram(cluster_size, title='', save_fig=''):

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(cluster_size, density=True, histtype='step', cumulative=True, bins=100)

    ax.set_title('Cumulative step histogram', fontsize=14)
    ax.set_xlabel('Estimated domain size /nm', fontsize=14)
    ax.set_ylabel('Likelihood of occurrence', fontsize=14)
    ax.set_title(title + ' total # domains: ' + str(np.round(len(cluster_size), 2)))
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300)
    plt.show()


def density_histogram(cluster_size, title='', save_fig=''):

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(cluster_size, density=True, bins=80)

    ax.set_title('Histogram', fontsize=14)
    ax.set_xlabel('Estimated domain size /nm', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(title + ' total # domains: ' + str(np.round(len(cluster_size), 2)))
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300)
    plt.show()


def get_average_bin(bins):
    x = []
    for i in range(len(bins) - 1):
        x.append((bins[i] + bins[i + 1]) / 2)

    return np.array(x)


def area_distribution(data, n_bins=30):
    n, bins = np.histogram(data, bins=n_bins)

    x = get_average_bin(bins)
    area = np.array(n) * x ** 2
    area = area / np.sum(area) * 100

    return x, area

# Code below still needs troubleshooting, I think it's because it's meant to work for multiple datasets


def plot_area_distribution(data, n_bins=80, save_fig='', title='', fit=False):
    fig, ax = plt.subplots()
    for key in data.keys():
        x, y = area_distribution(data[key], n_bins)
        if fit:
            ax.scatter(x, y, s=5)
            popt, _ = optimize.curve_fit(gaussian, x, y)
            popt = np.round(popt, 2)
            xnew = np.linspace(5, 100)

            ax.plot(xnew, gaussian(xnew, *popt), linewidth=1, label=make_label(key, popt))
        else:
            ax.scatter(x, y, s=5, label=key)

    ax.set_ylabel('area contribution (%)', fontsize=14)
    ax.set_xlabel('domain size / nm', fontsize=14)

    plt.legend()
    plt.title(title)
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)


def make_label(system, popt):
    return system + '\n [Fit: μ = ' + str(popt[1]) + ', σ = ' + str(popt[2]) + ']'


def get_single_domain(image, mask):
    def domain_coordinates(m):
        rows, cols = np.where(m > 0)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        return min_row, max_row + 1, min_col, max_col + 1

    min_row, max_row, min_col, max_col = domain_coordinates(mask)
    domain_image = image[min_row:max_row, min_col:max_col]
    domain_mask = mask[min_row:max_row, min_col:max_col]

    return domain_image, domain_mask












# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from scipy import optimize
# from matplotlib import colors
# import pandas as pd
#
# import plot_functions as plot
#
#
# def find_clusters(data, threshold, min_cluster_size, max_separation):
#     """Computes (x, y) map of clusters for input numpy array (data) as specified by argument conditions. Returns
#     a numbered map of clusters and averaged orientation map of each cluster.
#     Arguments:
#         data: 3D peaks array with 1 at (x, y, th) locations with peak center and 0 otherwise
#         threshold: maximum angular deviation (+/-) relative to average orientation of a cluster allowed for
#                     considering a neighboring point belongs to the same cluster
#         min_cluster_size: minimum size of a cluster (in total number of pixels) for it to be considered a cluster
#         max_separation: maximum separation allowed between neighboring points with similar orientation for them to be
#                     considered part of the same cluster.
#     Returns:
#         cluster_map: 2D numpy array with cluster number at each (x,y) formed based on arguments above. NaN values in
#                 (x, y) locations where no cluster was found.
#         orientation_map: 2D numpy array with average orientation angle (average theta) for found clusters. NaN values
#                 in (x, y) locations where no cluster was found.
#         cluster_orientation_std: 2D numpy array with calculated 1 standard deviation from average orientation angle for
#                 found clusters. Nan values in (x, y) locations where no cluster was found.
#         cluster_properties: dictionary storing the extracted information for each cluster. To be used for eithter
#                 troubleshooting or more analysis.
#     """
#     # Initialize outputs
#     m, n, _ = data.shape
#
#     rows, cols, angles = np.where(data > 0)
#     k = int(np.max(np.sum(data, axis=2)))
#
#     # Make input arrays
#     input_array = np.full(shape=(m, n, k), fill_value=np.nan)
#     input_counter = np.zeros((m, n), dtype=np.int)
#
#     # Fill input arrays with values
#     for i in range(len(rows)):
#         row, col, th = rows[i], cols[i], angles[i]
#         input_array[row, col, input_counter[row, col]] = th
#         input_counter[row, col] += 1
#
#     estimation = int(m * n * 2 / min_cluster_size)
#     output = np.full(shape=(estimation, m, n), fill_value=np.nan)
#     # Cluster map
#     cluster_map = np.full(shape=(k, m, n), fill_value=np.nan)
#     # Dictionary to track properties for each cluster
#     cluster_properties = {}
#     # Tracker for cluster number
#     cluster_number = 0
#
#     # Keep track of computation time
#     start_time = time.time()
#     num_pixels = []
#
#     for row in range(m):
#         if row % 10 == 0:
#             print('     ...Row: ', row)
#         for col in range(n):
#             for i in range(input_counter[row, col]):
#                 theta = input_array[row, col, i]
#                 if input_counter[row, col] > 0:
#                     theta_array, x_coords, y_coords = try_forming_cluster(input_array, input_counter, theta, threshold,
#                                                                          row, col, max_separation)
#                     # Determine if found cluster is large enough to be considered a cluster. If yes, save it in outputs.
#                     if len(x_coords) >= min_cluster_size:
#                         cluster_map[input_counter[x_coords, y_coords]-1, x_coords, y_coords] = cluster_number
#                         output[cluster_number, x_coords, y_coords] = theta_array
#                         median_theta = np.median(theta_array)
#                         cluster_properties[cluster_number] = \
#                             {'median_theta': median_theta, 'MAD_theta': np.median(np.abs(theta_array - median_theta)),
#                              'number_pixels': len(x_coords)}
#                         cluster_number += 1
#                         input_counter[x_coords, y_coords] -= 1
#                         num_pixels.append(len(x_coords))
#
#     output = output[:cluster_number, :, :]
#     print('     ...Formed {0} clusters'.format(cluster_number))
#     if cluster_number > 0:
#         print('     ...Mean size of clusters is {0} pixels'.format(np.round(np.mean(num_pixels), 2)))
#     print('     ...Clustering time(s): ', np.round((time.time() - start_time), 1))
#
#     return cluster_map, output, cluster_properties
#
#
# def try_forming_cluster(input_array, input_counter, theta, threshold, start_row, start_col, separation):
#     """Iterates over data array at a certain starting point (start row, start col) and searches for neighboring points
#     that can form a cluster. Returns a single cluster and list of orientation values.
#     Arguments:
#         data: 2D numpy array providing maximum orientation theta at each (x, y)
#         threshold: maximum angular deviation (+/-) relative to average orientation of a cluster allowed for
#                     considering a neighboring point belongs to the same cluster
#         start_row: initial row in data to start searching for cluster
#         start_col: initial column in data to start searching for cluster
#         separation: maximum separation allowed between neighboring points with similar orientation for them to be
#                     considered part of the same cluster.
#         cluster_map: 2D numpy array with cluster number at each (x,y) formed based on arguments above. NaN values in
#                 (x, y) locations where no cluster was found.
#     Returns:
#         theta_list: list of theta values at each point belonging to cluster
#         x_coords, y_coords = list of x and y coordinates, respectively, of points belonging to the cluster
#     """
#     m, n, _ = input_array.shape
#     # Initialization
#     theta_list = np.array([theta])
#     x_coords = np.array([start_row])
#     y_coords = np.array([start_col])
#
#     for row in range(start_row, m):
#         if row == start_row:
#             col_start = start_col + 1
#         else:
#             col_start = 0
#
#         if (np.abs(x_coords - row) > separation).all():
#             break
#
#         for col in range(col_start, n):
#             thetas = input_array[row, col]  # get possible theta values
#             if (~np.isnan(thetas)).any():
#                 closest_theta_index = np.nanargmin(np.abs(np.mean(theta_list) - thetas))
#                 th = input_array[row, col, closest_theta_index]
#                 if input_counter[row, col] > 0 and point_belongs_to_cluster(th, x_coords, y_coords, theta_list, threshold,
#                                                                             row, col, separation):
#                     x_coords = np.append(x_coords, row)
#                     y_coords = np.append(y_coords, col)
#                     theta_list = np.append(theta_list, th)  # Update orientation list
#
#     return theta_list, x_coords, y_coords
#
#
# def point_belongs_to_cluster(value, x_coords, y_coords, theta_list, threshold, row, col, separation):
#     """Determine if new point is neighbor to the cluster.
#      Arguments:
#          value: theta value at [row, col]
#          x_coords: list of row coordinates of points already in cluster
#          y_coords: list of col coordinates of points already in cluster
#          theta_list: list of theta values corresponding to points already assigned to cluster
#          threshold: maximum allowed angle misalignment
#          row: current row of point being evaluated
#          col: current column of point being evaluated
#          separation: maximum separation allowed for point (row, col) to be considered part of the same cluster
#     Returns:
#         True/False: T/F of whether point (x,y) belongs to cluster.
#     """
#     if not np.isnan(value):
#         if len(x_coords) == 0:
#             return True
#         else:
#             if np.abs(value - np.median(theta_list)) <= threshold:
#                 distance = np.sqrt((np.array(x_coords) - row) ** 2 + (np.array(y_coords) - col) ** 2) <= separation
#                 if distance.any():
#                     return True
#     return False
#
#
# def plot_cluster_map(output, angles, xlength, ylength, save_fig='', show_plot=False):
#
#     cmap = colors.ListedColormap(plot.get_colors(angles + 90))
#     fig = plt.figure(figsize=(10, 10))
#     for i in range(output.shape[0]):
#         plt.imshow(output[i, :, :], vmin=0, vmax=180, alpha=0.5, cmap=cmap, extent=[0, xlength, 0, ylength])
#     # plt.xticks([])
#     # plt.yticks([])
#     if save_fig:
#         plt.savefig(save_fig + '.png', dpi=300, transparent=True)
#     if show_plot:
#         plt.show()
#     else:
#         plt.close(fig)
#
#
# def cumulative_step_histogram(cluster_size, title='', save_fig=''):
#
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(cluster_size, density=True, histtype='step', cumulative=True, bins=100)
#
#     ax.set_title('Cumulative step histogram', fontsize=14)
#     ax.set_xlabel('Estimated domain size /nm', fontsize=14)
#     ax.set_ylabel('Likelihood of occurrence', fontsize=14)
#     ax.set_title(title + ' total # domains: ' + str(np.round(len(cluster_size), 2)))
#     if save_fig:
#         plt.savefig(save_fig + '.png', dpi=300)
#     plt.show()
#
#
# def density_histogram(cluster_size, title='', save_fig=''):
#
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(cluster_size, density=True, bins=80)
#
#     ax.set_title('Histogram', fontsize=14)
#     ax.set_xlabel('Estimated domain size /nm', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)
#     ax.set_title(title + ' total # domains: ' + str(np.round(len(cluster_size), 2)))
#     if save_fig:
#         plt.savefig(save_fig + '.png', dpi=300)
#     plt.show()
#
#
# def get_average_bin(bins):
#     x = []
#     for i in range(len(bins) - 1):
#         x.append((bins[i] + bins[i + 1]) / 2)
#
#     return np.array(x)
#
#
# def area_distribution(data, n_bins=30):
#     n, bins = np.histogram(data, bins=n_bins)
#
#     x = get_average_bin(bins)
#     area = np.array(n) * x ** 2
#     area = area / np.sum(area) * 100
#
#     return x, area
#
# # Code below still needs troubleshooting, I think it's because it's meant to work for multiple datasets
#
#
# def plot_area_distribution(data, n_bins=80, save_fig='', title='', fit=False):
#     fig, ax = plt.subplots()
#     for key in data.keys():
#         x, y = area_distribution(data[key], n_bins)
#         if fit:
#             ax.scatter(x, y, s=5)
#             popt, _ = optimize.curve_fit(gaussian, x, y)
#             popt = np.round(popt, 2)
#             xnew = np.linspace(5, 100)
#
#             ax.plot(xnew, gaussian(xnew, *popt), linewidth=1, label=make_label(key, popt))
#         else:
#             ax.scatter(x, y, s=5, label=key)
#
#     ax.set_ylabel('area contribution (%)', fontsize=14)
#     ax.set_xlabel('domain size / nm', fontsize=14)
#
#     plt.legend()
#     plt.title(title)
#     if save_fig:
#         plt.savefig(save_fig + '.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# def gaussian(x, amplitude, mean, stddev):
#     return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)
#
#
# def make_label(system, popt):
#     return system + '\n [Fit: μ = ' + str(popt[1]) + ', σ = ' + str(popt[2]) + ']'
#
#
# def get_single_domain(image, mask):
#     def domain_coordinates(m):
#         rows, cols = np.where(m > 0)
#         min_row, max_row = np.min(rows), np.max(rows)
#         min_col, max_col = np.min(cols), np.max(cols)
#
#         return min_row, max_row + 1, min_col, max_col + 1
#
#     min_row, max_row, min_col, max_col = domain_coordinates(mask)
#     domain_image = image[min_row:max_row, min_col:max_col]
#     domain_mask = mask[min_row:max_row, min_col:max_col]
#
#     return domain_image, domain_mask
#
