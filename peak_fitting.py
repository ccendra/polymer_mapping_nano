import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from itertools import combinations


def find_datacube_peaks(data, threshold_function, width=1, plot_freq=100):
    """
    Finds peak position of I vs. theta lineouts at each datacube gridpoint. Returns peaks matrix with ones in
    locations where a peak was found. Function uses a user-defined threshold function.
    :param data: datacube numpy array of size (n_rows, n_cols, n_angles)
    :param threshold_function: Required threshold of peaks, the vertical distance to its neighbouring samples.
    More info at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    :param distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    More info at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    :param width: Required width of peaks in samples.
    More info at https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    :param plot_freq: frequency at which to plot as examples the I vs. theta lineouts and resulting peak finding
    :return: peaks_matrix: 3D numpy array of same shape as datacube with ones in angular values at every grid point
    where a peak was found, and zeroes elsewhere.
    """
    m, n, th = data.shape
    # peaks_matrix = np.zeros(data.shape)  # Output matrix. Zeros will be replaced by ones where there is a peak.
    peaks_matrix = np.zeros((m, n, 180))
    overlap_angles_list = []

    f = int(1.5 * th)  # Extending signal by 50%
    x = np.linspace(0, f - 1, f)
    ct = 0  # counter for plotting examples

    for row in range(m):
        for col in range(n):
            d = data[row, col]
            if d.any() > 0:
                # Extend the signal in case there are peaks in edges of signal
                I = extend_signal(d, f)
                # Find peaks using scipy.signal.find_peaks() function.
                # More info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
                # Parameters:
                # height: required height of peaks -- use user-defined threshold function
                # threshold: required threshold, vertical distance to neighboring samples
                # width: required width of peaks

                # peaks_extended, properties = signal.find_peaks(I, distance=distance,
                #                                                prominence=threshold_function(I), width=width)
                # This version seems to work better
                peaks_extended, properties = signal.find_peaks(I, height=threshold_function(I), width=width)

                # Remove duplicate peaks, which occur because we extended signal. They could also occur when peak is
                # close to 0 or 180 degrees
                peaks, _ = remove_duplicates(peaks_extended)
                overlap_angles = find_overlap_angles(peaks)
                overlap_angles_list += overlap_angles

                # Determine location of peaks and place ones in peaks_matrix in theta values where there is a peak
                # peaks_loc = np.zeros(th) !!!!!!!!!!!!
                peaks_loc = np.zeros(180)
                peaks_loc[peaks] = 1
                peaks_matrix[row, col, peaks] = 1

                # Plotting examples
                if plot_freq !=0 and ct % plot_freq == 0:
                    print('PEAKS: ', peaks)
                    print('OVERLAPS: ', overlap_angles)

                    fig, ax = plt.subplots()
                    # Plot extended I vs. theta signal, both as scatter plot and lineplot (IMO seeing the point density
                    # and tracing a line helps with visualization)
                    ax.scatter(x, I, s=2, color='blue')
                    ax.plot(x, I, color='blue', linewidth=0.5)

                    # Add 'x' where in theta values where there is a peak
                    ax.plot(peaks_extended, I[peaks_extended], "x", color='gray')
                    # Add markers to show peak features found by find_peaks() function
                    ax.vlines(x=peaks_extended, ymin=I[peaks_extended] - properties["prominences"],
                              ymax=I[peaks_extended], color="gray")
                    ax.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                              xmax=properties["right_ips"], color="gray")
                    # Add theta value of peaks as text
                    for i in range(len(peaks_extended)):
                        ax.text(peaks_extended[i], I[peaks_extended[i]] * 1.05, str(peaks_extended[i]), fontsize=10)

                    # ax.set_ylim([np.amin(I) * 0.9, np.amax(I) * 1.3])
                    ax.set_ylim([0, np.amax(I) * 1.3])
                    ax.set_xlabel('θ / degrees')
                    ax.set_ylabel('Counts / a.u.')
                    plt.show()
            ct += 1

    return peaks_matrix, overlap_angles_list


def find_max_peak_in_lineout(lineout, threshold, width=1):
    th = len(lineout)
    f = int(1.5 * th)  # Extending signal by 50%
    I = extend_signal(lineout, f)
    peaks_extended, properties = signal.find_peaks(I, height=threshold(I), width=1)
    peak, _ = remove_duplicates(peaks_extended)

    if len(peak) > 1:
        peak_index = np.argmax(lineout[peak])
        peak = peak[peak_index]
    else:
        peak = peak[0]

    return peak

def extend_signal(intensity, f):
    """
    Extends/repeats 1D signal (i.e. I vs theta) to account for peaks present in edges of signal, which peak fitting
    algorithm sometimes misses.
    Args:
        intensity: signal (i.e. I vs th)
        f: final length of signal
    Returns:
        y: extended signal
    """
    y = np.zeros(f)
    th = len(intensity)
    y[:th] = intensity
    y[th:] = intensity[:(f - th)]

    return y


def remove_duplicates(peaks):
    """
    Removes peaks that were found twice. This happens because we extend the signal.
    Args:
        :param peaks: List of peaks found by peak fitting algorithm
    Returns:
        peak_list: trimmmed list of peaks with removed duplicates
        peak_idx: index of found peaks
    """
    peak_list = []
    peak_idx = []
    ct = 0
    for peak in peaks:
        if peak >= 180:
            peak = peak - 180
        if peak not in peak_list:
            peak_list.append(peak)
            peak_idx.append(ct)
        ct += 1

    return peak_list, peak_idx


def find_overlap_angles(peaks):
    """
    Finds all possible angular difference from a list of unique angles.
    Args:
        peaks: list of angular values where peak was found
    Returns:
        overlap_angles: list angular differences between peaks
    """
    angle_combinations = list(combinations(peaks, 2))
    overlap_angles = []

    for comb in angle_combinations:
        angle_difference = np.abs(comb[0] - comb[1])
        if angle_difference > 90:
            angle_difference = 180 - angle_difference

        overlap_angles.append(angle_difference)

    return overlap_angles
