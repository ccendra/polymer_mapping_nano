import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib import colors
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
    input_array = np.full(shape=(m, n, k), fill_value=np.nan)
    input_counter = np.zeros((m, n), dtype=int)

    # Fill input arrays with values
    for i in range(len(rows)):
        row, col, th = rows[i], cols[i], angles[i]
        input_array[row, col, input_counter[row, col]] = th
        input_counter[row, col] += 1

    estimation = int(m * n * 2 / min_cluster_size)
    output = np.full(shape=(m, n, estimation), fill_value=np.nan)
    # Cluster map
    cluster_map = np.full(shape=(m, n, k), fill_value=np.nan)
    # Dictionary to track properties for each cluster
    cluster_properties = {}
    # Tracker for cluster number
    cluster_number = 0

    # Keep track of computation time
    start_time = time.time()
    num_pixels = []

    for row in range(m):
        if row % 10 == 0:
            print('row: ', row)
        for col in range(n):
            for i in range(input_counter[row, col]):
                theta = input_array[row, col, i]
                if input_counter[row, col] > 0:
                    theta_list, x_coords, y_coords = try_forming_cluster(input_array, input_counter, theta, threshold,
                                                                         row, col, max_separation)
                    # Determine if found cluster is large enough to be considered a cluster. If yes, save it in outputs.
                    if len(x_coords) >= min_cluster_size:
                        cluster_map[x_coords, y_coords, input_counter[x_coords, y_coords]-1] = cluster_number
                        output[x_coords, y_coords, cluster_number] = theta_list
                        cluster_properties[cluster_number] = \
                            {'median_theta': np.median(theta_list), 'stdev_theta': np.std(theta_list),
                             'number_pixels': len(x_coords), 'theta_list': theta_list}
                        cluster_number += 1
                        input_counter[x_coords, y_coords] -= 1
                        num_pixels.append(len(x_coords))

    output = output[:, :, :cluster_number]
    print('formed {0} clusters'.format(cluster_number))
    print('Mean size of clusters is {0} pixels'.format(np.round(np.mean(num_pixels), 2)))
    print('clustering time(s): ', np.round((time.time() - start_time), 1))

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
    if not np.isnan(value):
        if len(x_coords) == 0:
            return True
        else:
            if np.abs(value - np.median(theta_list)) <= threshold:
                distance = np.sqrt((np.array(x_coords) - row) ** 2 + (np.array(y_coords) - col) ** 2) <= separation
                if distance.any():
                    return True
    return False


def plot_cluster_map(output, angles, xlength, ylength):

    cmap = colors.ListedColormap(plot.get_colors(angles + 90))
    plt.figure(figsize=(10, 10))
    for i in range(output.shape[2]):
        plt.imshow(output[:, :, i], vmin=0, vmax=180, alpha=0.5, cmap=cmap, extent=[0, xlength, 0, ylength])
    plt.show()


def cumulative_step_histogram(data, nbins=500, title='', save_fig=''):
    if isinstance(data, pd.DataFrame):
        data = data.cluster_size_nm

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(data, density=True, histtype='step', cumulative=True, bins=nbins)

    ax.set_title('Cumulative step histogram', fontsize=14)
    ax.set_xlabel('Estimated domain size /nm', fontsize=14)
    ax.set_ylabel('Likelihood of occurrence', fontsize=14)
    ax.set_title(title + ' total # domains: ' + str(np.round(len(data), 2)))
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300)
    plt.show()


def density_histogram(data, title='', save_fig=''):
    if isinstance(data, pd.DataFrame):
        data = data.cluster_size_nm

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(data, density=True, bins=80)

    ax.set_title('Histogram', fontsize=14)
    ax.set_xlabel('Estimated domain size /nm', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xlim([0, 80])
    ax.set_title(title + ' total # domains: ' + str(np.round(len(data), 2)))
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300)
    plt.show()


def get_average_bin(bins):
    x = []
    for i in range(len(bins) - 1):
        x.append((bins[i] + bins[i + 1]) / 2)

    return np.array(x)


def area_distribution(data, n_bins=80):
    n, bins = np.histogram(data, bins=n_bins)

    x = get_average_bin(bins)
    area = np.array(n) * x ** 2
    area = area / np.sum(area) * 100

    return x, area


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





# import numpy as np
# import pandas as pd
# import time
# import matplotlib.pyplot as plt
# from scipy import optimize
#
#
# def find_clusters(peaks_matrix, pixel_size, threshold=10, min_cluster_size=5, max_separation=3):
#     """Computes (x, y) map of clusters for input numpy array (data) as specified by argument conditions. Returns
#     a numbered map of clusters and averaged orientation map of each cluster.
#     Arguments:
#         data: 2D numpy array providing maximum orientation theta at each (x, y)
#         threshold: maximum angular deviation (+/-) relative to average orientation of a cluster allowed for
#                     considering a neighboring point belongs to the same cluster
#         min_cluster_size: minimum size of a cluster (in total number of pixels) for it to be considered a cluster
#         max_separation: maximum separation allowed between neighboring points with similar orientation for them to be
#                     considered part of the same cluster.
#         pixel_size: length of one pixel in nm
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
#     m, n, _ = peaks_matrix.shape
#     rows, cols, angles = np.where(peaks_matrix > 0)
#     k = int(np.max(np.sum(peaks_matrix, axis=2)))
#
#     # Make input arrays
#     input_array = np.full(shape=(m, n, k), fill_value=np.nan)
#     input_counter = np.zeros((m, n), dtype=int)
#
#     # Fill input arrays with values
#     for i in range(len(rows)):
#         row, col, th = rows[i], cols[i], angles[i]
#         input_array[row, col, input_counter[row, col]] = th
#         input_counter[row, col] += 1
#
#     # Cluster map
#     cluster_map = np.full(shape=(m, n, k), fill_value=np.nan)
#     # Cluster average orientation map
#     cluster_orientation_map = np.full(shape=(m, n, k), fill_value=np.nan)
#     # Dictionary to track properties for each cluster
#     cluster_properties = {}
#     # Tracker for cluster number
#     cluster_number = 1
#
#     # Keep track of computation time
#     start_time = time.time()
#
#     for row in range(m):
#         if row % 10 == 0:
#             print('row: ', row)
#         for col in range(n):
#             # print('col: ', col)
#             for i in range(input_counter[row, col]):
#                 # print(i)
#                 if np.isnan(cluster_map[row, col, i]) and input_counter[row, col] > 0:
#                     theta_list, x_coords, y_coords, th_coords = \
#                         try_forming_cluster(input_array, input_counter, threshold, row, max_separation, cluster_map)
#                     # Determine if found cluster is large enough to be considered a cluster. If yes, save it in outputs.
#                     if len(x_coords) >= min_cluster_size:
#                         cluster_map[x_coords, y_coords, th_coords] = cluster_number
#                         cluster_orientation_map[x_coords, y_coords, th_coords] = np.mean(theta_list)
#                         cluster_properties[cluster_number] = \
#                             {'mean_theta': np.mean(theta_list), 'stdev_theta': np.std(theta_list),
#                              'number_pixels': len(x_coords), 'cluster_size_nm': np.sqrt(len(x_coords)) * pixel_size,
#                              'theta_list': theta_list}
#                         cluster_number += 1
#                         input_array[x_coords, y_coords, th_coords] = np.nan
#                         input_counter[x_coords, y_coords] -= 1
#                     # # Else, mark locations as belonging to a cluster that is too small
#                     else:
#                         cluster_map[x_coords, y_coords, th_coords] = -1
#
#     # # Convert cluster map to dataframe to remove zeros (locations where no cluster could be found)
#     # # and convert again to numpy
#     # cluster_map_df = pd.DataFrame(cluster_map)
#     # cluster_map = cluster_map_df[cluster_map_df > 0].to_numpy()
#
#     print('clustering time(min): ', np.round((time.time() - start_time) / 60, 1))
#
#     return cluster_map, cluster_orientation_map, cluster_properties
#
#
# def try_forming_cluster(input_array, input_counter, threshold, start_row, max_separation, cluster_map):
#     """Iterates over data array at a certain starting point (start row, start col) and searches for neighboring points
#     that can form a cluster. Returns a single cluster and list of orientation values.
#     """
#     m, n, k = input_array.shape
#     # Initialization
#     theta_list = []
#     x_coords = []
#     y_coords = []
#     th_coords = []
#
#     for row in range(start_row, m):
#         for col in range(n):
#             for i in range(input_counter[row, col]):
#                 theta = input_array[row, col, i]  # get theta value
#                 if np.isnan(cluster_map[row, col, i]) and point_belongs_to_cluster(theta, x_coords, y_coords, theta_list, threshold, row, col, max_separation):
#                     x_coords.append(row)
#                     y_coords.append(col)
#                     theta_list.append(theta)  # Update orientation list
#                     th_coords.append(i)
#
#     return theta_list, x_coords, y_coords, th_coords
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
#             if np.abs(value - np.mean(theta_list)) <= threshold:
#                 distance_x = np.abs(np.array(x_coords) - row) <= separation
#                 distance_y = np.abs(np.array(y_coords) - col) <= separation
#                 if distance_x.any() and distance_y.any():
#                     return True
#
#     return False
#
#
# def cumulative_step_histogram(data, nbins=500, title='', save_fig=''):
#     if isinstance(data, pd.DataFrame):
#         data = data.cluster_size_nm
#
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(data, density=True, histtype='step', cumulative=True, bins=nbins)
#
#     ax.set_title('Cumulative step histogram', fontsize=14)
#     ax.set_xlabel('Estimated domain size /nm', fontsize=14)
#     ax.set_ylabel('Likelihood of occurrence', fontsize=14)
#     ax.set_title(title + ' total # domains: ' + str(np.round(len(data), 2)))
#     if save_fig:
#         plt.savefig(save_fig + '.png', dpi=300)
#     plt.show()
#
#
# def density_histogram(data, title='', save_fig=''):
#     if isinstance(data, pd.DataFrame):
#         data = data.cluster_size_nm
#
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(data, density=True, bins=80)
#
#     ax.set_title('Histogram', fontsize=14)
#     ax.set_xlabel('Estimated domain size /nm', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)
#     ax.set_xlim([0, 80])
#     ax.set_title(title + ' total # domains: ' + str(np.round(len(data), 2)))
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
# def area_distribution(data, n_bins=80):
#     n, bins = np.histogram(data, bins=n_bins)
#
#     x = get_average_bin(bins)
#     area = np.array(n) * x ** 2
#     area = area / np.sum(area) * 100
#
#     return x, area
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



# def point_belongs_to_cluster(angle, x_coords, y_coords, th_list, row, col, threshold, separation):
#     if not np.isnan(angle):
#         if len(x_coords) == 0:
#             return True
#         else:
#             if np.abs(angle - np.mean(th_list)) <= threshold:
#                 distance = np.sqrt((np.array(x_coords) - row)**2 + (np.array(y_coords) - col)**2) <= separation
#                 if distance.any():
#                     return True
#
#     return False
#
#
# def try_forming_cluster(i, input_array, input_counter, threshold, separation):
#     m, n, p = input_array.shape
#     x_coords, y_coords, th_coords, th_values = \
#         np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
#
#     for row in range(i, m):
#         for col in range(n):
#             for angle in range(input_counter[row, col]):
#                 value = input_array[row, col, angle]
#                 if point_belongs_to_cluster(value, x_coords, y_coords, th_values, row, col, threshold, separation):
#                     x_coords = np.append(x_coords, row)
#                     y_coords = np.append(y_coords, col)
#                     th_coords = np.append(th_coords, angle)
#                     th_values = np.append(th_values, value)
#
#         if (row - x_coords > separation).all():
#             break
#
#     return x_coords, y_coords, th_coords, th_values
#
#
# def find_clusters(data, threshold, separation, min_cluster_size):
#     start_time = time.time()
#     m, n, th = data.shape
#     rows, cols, angles = np.where(data > 0)
#     n_max_clusters_gridpoint = int(np.max(np.sum(data, axis=2)))
#
#     # Make input arrays
#     input_array = np.full(shape=(m, n, n_max_clusters_gridpoint), fill_value=np.nan)
#     input_counter = np.zeros((m, n), dtype=int)
#
#     # Fill input arrays with values
#     for i in range(len(rows)):
#         row, col, th = rows[i], cols[i], angles[i]
#         input_array[row, col, input_counter[row, col]] = th
#         input_counter[row, col] += 1
#
#     # Output array
#     output = []
#
#     for i in range(m):
#         if i % 10 == 0:
#             print('row: ', i)
#         for j in range(n):
#             n_peaks = input_counter[i, j]
#             for k in range(n_peaks):
#                 if np.isnan(input_array[i, j, k]):
#                     continue   # do next iteration of innermost for loop
#
#                 x_coords, y_coords, th_coords, th_list = \
#                     try_forming_cluster(i, input_array, input_counter, threshold, separation)
#
#                 if x_coords.shape[0] >= min_cluster_size:
#                     belongs_to_cluster_angle = np.abs(th_list - np.mean(th_list)) <= threshold
#                     #
#                     # x_grid = np.reshape(np.repeat(x_coords, x_coords.shape[0]), (x_coords.shape[0], x_coords.shape[0]))
#                     # belongs_to_distance_x = np.any((x_grid - x_coords) <= separation, axis=1)
#                     # y_grid = np.reshape(np.repeat(y_coords, y_coords.shape[0]), (y_coords.shape[0], y_coords.shape[0]))
#                     # belongs_to_distance_y = np.any((y_grid - y_coords) <= separation, axis=1)
#
#                     points = np.where(belongs_to_cluster_angle)
#
#                     x, y, z, th_values = x_coords[points], y_coords[points], th_coords[points], th_list[points]
#
#                     input_array[x, y, z] = np.nan
#                     input_counter[x, y] -= 1
#                     output.append([x, y, z, th_values])
#
#     print('Found {0} clusters'.format(len(output)))
#     print('Processing time: {0}'.format(np.round(time.time() - start_time, 2)))
#     return output, input_array, input_counter
#
#
# def make_cluster_mask(x_coords, y_coords, m, n):
#     mask = np.zeros((m, n))
#     for i in range(len(x_coords)):
#         mask[x_coords[i], y_coords[i]] = 1
#
#     return mask
#
#
# def cumulative_step_histogram(data, nbins=500, title='', save_fig=''):
#     if isinstance(data, pd.DataFrame):
#         data = data.cluster_size_nm
#
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(data, density=True, histtype='step', cumulative=True, bins=nbins)
#
#     ax.set_title('Cumulative step histogram', fontsize=14)
#     ax.set_xlabel('Estimated domain size /nm', fontsize=14)
#     ax.set_ylabel('Likelihood of occurrence', fontsize=14)
#     ax.set_title(title + ' total # domains: ' + str(np.round(len(data), 2)))
#     if save_fig:
#         plt.savefig(save_fig + '.png', dpi=300)
#     plt.show()
#
#
# def density_histogram(data, title='', save_fig=''):
#     if isinstance(data, pd.DataFrame):
#         data = data.cluster_size_nm
#
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(data, density=True, bins=80)
#
#     ax.set_title('Histogram', fontsize=14)
#     ax.set_xlabel('Estimated domain size /nm', fontsize=14)
#     ax.set_ylabel('Frequency', fontsize=14)
#     ax.set_xlim([0, 80])
#     ax.set_title(title + ' total # domains: ' + str(np.round(len(data), 2)))
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
# def area_distribution(data, n_bins=80):
#     n, bins = np.histogram(data, bins=n_bins)
#
#     x = get_average_bin(bins)
#     area = np.array(n) * x ** 2
#     area = area / np.sum(area) * 100
#
#     return x, area
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
