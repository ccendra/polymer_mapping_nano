# General packages
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Custom packages - each performs different task
import reduce_data as reduce    # Reduce data in fourier space and get datacube
import auxiliary_functions as aux   # Simple auxiliary functions
import plot_functions as plot   # Methods to plot data
import drift_correction as drift    # Drift correction
import peak_fitting as peaks    # Peak fitting methods
import autocorrelations as acf  # Autocorrelations using MultiProcess
import clustering as cluster    # Find clusters
import director_fields as director    # Director fields visualization
import flow_fields as flow     # Flow fields visualization

#######################################################################################################################
# Functions

def read_raw_data(input_folder, filename, subregion=None, s0=0):
    """
    Reads raw .mrc or .tif data frames and stores them as torch tensor (on CPU). User can specify to select a subregion
    of image. Subregion variable selects top left corner of image for specified range, or user can specify to get
    subregion with a starting row and column (s0).
    :param input_folder: location of .mrc or .tif image
    :param filename: filename of image at input_folder location
    :param subregion: integer specifying shape of selected subregion. Default is None.
    :param s0: starting index of subregion. Default is zero.
    :return: Pytorch tensor of raw_data in CPU. Shape is (number_of_frames, 3838, 3720) for K2 detector images.
    """
    file_type = filename.split('.')[-1]     # Determine file ending (.mrc or .tif accepted ATM).
    fn = input_folder + filename
    raw_data = None
    if file_type == 'mrc':
        print('\n...Opening .mrc file')
        raw_data = aux.read_mrc(fn)     # Calls auxiliary function to read .mrc
    elif file_type == 'tif':
        print('\n...Opening .tif file')
        raw_data = aux.read_tif(fn)     # Calls auxiliary function to read .tif
    else:   # User here could add options to open other file types.
        print('Invalid file type. Accepted files are either .mrc or .tif')

    if subregion:   # Select subregion. Top left corner as default, unless s0 is specified
        raw_data = raw_data[:, s0:subregion+s0, s0:subregion+s0]

    return torch.from_numpy(raw_data)   # Tensor in CPU


def initial_visualization(data, preliminary_num_frames, size_fft, dx, gamma_images=1, plot_lineout=False,
                          show_figures=False, save_results=False, output_folder=''):
    """
    Perfoms initial visualization of raw data prior to performing any transformations. Method stacks a
    pre-determined number of raw frames and plots stacked image, FFT, and azimuthally integrated powder
    lineout (optional).
    :param data: data frame PyTorch tensor (either in CPU or GPU).
    :param preliminary_num_frames: number of frames to use for initial visualization. Typically 4 tp 8 work well.
    :param size_fft: size of Fourier transform. Prior to computing Fourier transform, image is padded with zeros to
    match this size. Recommended to use binary-power values for fast execution (FFT computes faster with 2^x sizes).
    :param dx: Image resolution in Angstrom/pixel.
    :param gamma_images: Gamma value for contrast tuning of images. Default to one.
    :param plot_lineout: Boolean to compute azimuthally integrated powder spectra (this is a medium-expensive
    computation in GPU). Default is False. Can make it faster by coarsening of the bandpass mask.
    :param show_figures: Boolean to show figures.
    :param save_results: Boolean to save results (figures and lineouts).
    :param output_folder: Folder to store results. Governed by save_results variable.
    """
    print('\n...Performing initial visualizations \n '
          'Note: transformations here are temporary and only for visualization purposes.')
    if not data.is_cuda:
        # Send data frames to GPU
        data = data.to('cuda')

    # Stack n first frames of data
    data = torch.sum(data[:preliminary_num_frames, :, :], dim=0)
    print('     ...The first {0} image frames have been stacked and image size is: {1}'.
          format(preliminary_num_frames, data.shape))
    # Determine output filenames in case figures are saved
    if save_results:
        save_res = [output_folder + 'initial_visualization_hrtem',
                    output_folder + 'initial_visualization_fft',
                    output_folder + 'initial_visualization_IvsQ_lineout',
                    output_folder + 'aziumuthally_integrated_lineout.csv']
    else:
        save_res = ['', '', '', '']

    # Get FFT (using GPU)
    img_fft_gpu = reduce.tensor_fft(data, size_fft)

    # Remove stacked_data tensor from GPU
    data = data.to('cpu')

    # Plot HRTEM
    plot.hrtem(data.numpy(), size=10, gamma=gamma_images, vmax=0, colorbar=False, dx=dx, save_fig=save_res[0],
               show_plot=show_figures)

    # Plot FFT
    plot.fft(img_fft_gpu.to('cpu'), size_fft, q_contour_list=[], dx=dx, save_fig=save_res[1], show_plot=show_figures)

    # Calculate azimuthally integrated powder lineout (using FFT in GPU)
    if plot_lineout:
        q_increments = 0.01  # Can be increased to 0.01 for coarser (but faster) calculation
        q_bandwidth = 0.01   # Can be increased to 0.01 for coarser (but faster) calculation
        x_powder, y_powder = reduce.extract_intensity_q_lineout(img_fft_gpu, q_increments, q_bandwidth, dx)
        if save_results:
            lineout = pd.DataFrame(np.array([x_powder, y_powder]).T, columns=['q', 'intensity'])
            lineout.to_csv(save_res[3], index=False)

        plot.intensity_q_lineout(x_powder, y_powder, save_fig=save_res[2], show_plot=show_figures)

    img_fft_gpu = img_fft_gpu.to('cpu')
    torch.cuda.empty_cache()    # housekeeping


def apply_bandpass_filter_to_image_stack(data_frames, q_center, q_bandwidth, dx, beta=0.1):
    """
    Apply bandpass filter to all frames in raw data. Returns stack of bandpass filtered real space frames.
    :param data_frames: tensor with image frames. Tensor will be modified with bandpass-filtered analogue.
    :param q_center: q center position for bandpass filter (in inverse Angstrom).
    :param q_bandwidth: bandwidth for bandpass filter in inverse Angstrom. Filter will use as range +/- bandwidth.
    :param dx: image resolution in Angstrom / pixel.
    :param beta: parameter for raised cosine window. Default is 0.1.
    :return: bandpass-filtered stack of frames. PyTorch tensor in CPU.
    """
    print('\n...Filtering raw data frames with bandpass filter')
    n_frames, m, n = data_frames.shape

    if not data_frames.is_cuda:
        # Send data frames to GPU
        data_frames = data_frames.to('cuda')

    # Apply bandpass filter to each individual frame
    # reduce.bandpass_filtering_image takes care of:
    # 1) Applying raised cosine window
    # 2) Adding padding to image such that image is shape is max(m,n) x max(m, n)
    # 3) Removing padding to bandpass filtered image
    print('   ...Applying bandpass filter to all frames in image')
    for i in range(n_frames):
        data_frames[i, :, :] = reduce.bandpass_filtering_image(data_frames[i, :, :],
                                                                    q_center, q_bandwidth, dx, beta=beta)
    # Bring tensor to CPU
    data_frames = data_frames.to('cpu')
    torch.cuda.empty_cache()    # housekeeping

    return data_frames


def analyze_stack_properties(data_frames, plot_color, dx, show_figures=False, save_results=False, output_folder=''):
    """
    Characterize properties of the stack in terms of integrated powder intensity and drift motion.
    STRONGLY RECOMMENDED to use bandpass filtered image, such that analysis is performed solely on information
    contained in q range of interest. Function returns drift in x and y directions of image.
    :param data_frames: image dataframes (ideally bandpass filtered). PyTorch tensor.
    :param plot_color: color for scatter plots.
    :param dx: image resolution. In Angstrom / pixel.
    :param show_figures: Boolean to show figures.
    :param save_results:Boolean to save results (figures and lineouts).
    :param output_folder: Folder to store results. Governed by save_results variable.
    :return:
    """
    print('\n...Analyzing full-stack behavior')

    n_frames, m, n = data_frames.shape
    s = max(m, n)
    # Pad image. If m = n, no padding will be performed.
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)

    # Integrated powder intensities
    print('     ...Getting integrated powder intensities vs. frame number.')
    fft_integrated_intensity = torch.zeros(n_frames)
    for i in range(n_frames):
        # Do Fourier transform of bandpass filtered image
        frame_fft_gpu = reduce.tensor_fft(pad(data_frames[i, :, :]).to('cuda'), s)
        # Store integrated powder spectrum at bandwidth q
        fft_integrated_intensity[i] = torch.sum(frame_fft_gpu).to('cpu')

    torch.cuda.empty_cache()  # housekeeping

    fig = plt.figure()
    plt.scatter(np.arange(n_frames), fft_integrated_intensity.cpu(), color=plot_color)
    plt.xlabel('frame number')
    plt.ylabel('Integrated counts')
    plt.ylim([torch.min(fft_integrated_intensity) * 0.8, torch.max(fft_integrated_intensity) * 1.1])
    if save_results:
        plt.savefig(output_folder + 'stack_analysis_IntegratedCounts_vs_FrameNumber.png', bbox_inches='tight')
    if show_figures:
        plt.show()
    else:
        plt.close(fig)

    # Tracking drift
    print('     ...Tracking drift between frames.')
    x_drift, y_drift = drift.track_drift(pad(data_frames).to('cuda'), s, verbose=False)

    fig = plt.figure()
    plt.scatter(np.arange(n_frames), x_drift, color=plot_color)
    plt.plot(np.arange(n_frames), x_drift, color='black', linewidth=0.5)
    plt.ylabel('Image drift in x̄ / pixels', fontsize=14)
    plt.xlabel('image #', fontsize=14)
    if save_results:
        plt.savefig(output_folder + 'stack_analysis_ImageDrift_x_direction.png', bbox_inches='tight')
    if show_figures:
        plt.show()
    else:
        plt.close(fig)

    fig = plt.figure()
    plt.scatter(np.arange(n_frames), y_drift, color=plot_color)
    plt.plot(np.arange(n_frames), y_drift, color='black', linewidth=0.5)
    plt.ylabel('Image drift in ȳ / pixels', fontsize=14)
    plt.xlabel('image #', fontsize=14)
    if save_results:
        plt.savefig(output_folder + 'stack_analysis_ImageDrift_y_direction.png', bbox_inches='tight')
    if show_figures:
        plt.show()
    else:
        plt.close(fig)

    drift.plot_2d_drift(x_drift, y_drift, dx=dx, lines=False,
                        save_fig=output_folder + 'stack_analysis_2D_drift', show_plot=show_figures)

    # Send data back to CPU
    data_frames = data_frames.to('cpu')
    torch.cuda.empty_cache()

    return x_drift, y_drift


def correct_drift(data_frames, data_frames_bp_filtered, max_drift_allowed, first_frame, last_frame, x_drift, y_drift,
                  dx, size_fft, show_figures=False, save_results=False, output_folder='', gamma_images=1,
                  save_corrected_results=False):

    # Select range for images to be drift corrected
    data_frames_bp_filtered = data_frames_bp_filtered[first_frame:last_frame, :, :]
    # raw data to visualize image after drift correction
    data_frames = data_frames[first_frame:last_frame, :, :]

    print('\n...Correcting drift between frames. Maximum drift allowed is {0} pixels in either '
          'x or y directions.'.format(max_drift_allowed))

    n_frames, m, n = data_frames_bp_filtered.shape
    padding = max_drift_allowed + 1

    data_corrected = np.zeros((n_frames, m + 2 * padding, n + 2 * padding))
    data_corrected_bp_filtered = torch.zeros(n_frames, m + 2 * padding, n + 2 * padding)

    ct = 0
    for i in range(n_frames):
        ux = x_drift[i]
        uy = y_drift[i]
        if np.abs(ux) <= max_drift_allowed and np.abs(uy) <= max_drift_allowed:
            a, b, c, d = int(padding - ux), -int(padding + ux), int(padding - uy), -int(padding + uy)
            data_corrected[ct, a:b, c:d] = data_frames[i, :m, :n]
            data_corrected_bp_filtered[ct, a:b, c:d] = data_frames_bp_filtered[i, :m, :n]
            ct += 1

    size = padding + max_drift_allowed
    data_corrected = data_corrected[:ct, size:-size, size:-size]
    data_corrected_bp_filtered = data_corrected_bp_filtered[:ct, size:-size, size:-size]
    print('     ...Data has been drift-corrected and new shape is: ', data_corrected.shape)

    # Overwrite with stacked data for image visualization
    if save_results:
        save_res = [output_folder + 'drift_corrected_hrtem',
                    output_folder + 'drift_corrected_hrtem_bp_filtered',
                    output_folder + 'drift_corrected_fft']
    else:
        save_res = ['', '', '']

    if save_corrected_results:
        np.save(output_folder + 'image_drift_corrected.npy', data_corrected)
        print('     ...Drift corrected stacked image has been saved.')


        np.save(output_folder + 'image_bp_filter_drift_corrected.npy', data_corrected_bp_filtered)
        print('     ...Drift corrected stacked bandpass image has been saved.')

    # Visualization of drift-corrected data.
    data_corrected = np.sum(data_corrected, axis=0)  # Stack frames for case of drift corrected raw image (no filter).

    # HRTEM of image with all spatial frequencies accounted
    plot.hrtem(data_corrected, size=10, gamma=gamma_images, vmax=0, colorbar=False, dx=dx,
               save_fig=save_res[0], show_plot=show_figures)
    # HRTEM of bandpass filtered image
    plot.hrtem((torch.sum(data_corrected_bp_filtered, dim=0)).numpy(), size=10, gamma=gamma_images, vmax=0,
               colorbar=False, dx=dx, save_fig=save_res[1], show_plot=show_figures)

    # FFT of image
    img_fft_gpu = reduce.tensor_fft(torch.from_numpy(data_corrected).to('cuda'), size_fft)
    img_fft_gpu = img_fft_gpu.to('cpu')
    plot.fft(img_fft_gpu, size_fft, q_contour_list=[], save_fig=save_res[2], show_plot=show_figures)

    torch.cuda.empty_cache()    # housekeeping

    return data_corrected_bp_filtered


def stack_selected_frames(data_frames_bp_filtered, frames, dx, size_fft,
                          show_figures=False, save_results=False, output_folder='', gamma_images=1):
    """

    :param data_frames_bp_filtered:
    :param frames:
    :param dx:
    :param size_fft:
    :param show_figures:
    :param save_results:
    :param output_folder:
    :param gamma_images:
    :return:
    """
    if len(frames) == 2:
        first_frame = frames[0]
        last_frame = frames[1]
        print('\n...Selecting frame stack from frame # {0} to frame # {1}'.format(first_frame, last_frame))
        data = data_frames_bp_filtered[first_frame:last_frame, :, :]
    else:
        print('\n...Selecting frames {0} in stack.'.format(frames))
        data = data_frames_bp_filtered[frames, :, :]

    data = torch.sum(data, dim=0)

    if save_results:
        save_res = [output_folder + 'selected_stack_sum',
                    output_folder + 'selected_stack_sum']
    else:
        save_res = ['', '']

    plot.hrtem(data.numpy(), size=10, gamma=gamma_images, vmax=0, colorbar=False, dx=dx, save_fig=save_res[0],
               show_plot=show_figures)

    img_fft_gpu = reduce.tensor_fft(data.to('cuda'), size_fft)
    plot.fft(img_fft_gpu.to('cpu'), size_fft, q_contour_list=[], save_fig=save_res[1], show_plot=show_figures)

    return data


def reduce_data(data, q_center, sigma_q, sigma_th, dx, bandwidth_q, angles, N, M, step_size, number_frames=None,
                    save_datacube=False, plot_frequency=0, show_figures=False, save_results=False, output_folder=''):
    """

    :param data:
    :param q_center:
    :param sigma_q:
    :param sigma_th:
    :param dx:
    :param bandwidth_q:
    :param angles:
    :param N:
    :param M:
    :param step_size:
    :param number_frames:
    :param save_datacube:
    :param plot_frequency:
    :param show_figures:
    :param save_results:
    :param output_folder:
    :return:
    """
    print('\nPerforming data reduction.')

    if not data.is_cuda:
        data = data.to('cuda')

    # Stack frames and/or account for different situations based on previous data preprocessing
    if len(data.shape) == 3:
        if number_frames:
            # Case where we want to stack a select number of frames. This happens when there was no drift
            # correction or if want to stack fewer frames than the drift-corrected data.
            print('     ...Stacking first {0} frames'.format(number_frames))
            data = torch.sum(data[:number_frames, :, :], dim=0)
        else:
            print('     ...Stacking all {0} frames'.format(data.shape[0]))
            # Stack all frames together
            data = torch.sum(data, dim=0)

    print('\n...Getting datacube.')
    # Getting filters (bandpass and gaussian, then combine)
    gaussian_filter = reduce.gaussian_q_filter(q_center, sigma_q, sigma_th, M, dx)
    bandpass_filter = reduce.bandpass_filter(M, q_center - bandwidth_q, q_center + bandwidth_q, dx)
    selected_filter = gaussian_filter * bandpass_filter

    q_max = np.pi / dx

    fig = plt.figure()
    plt.imshow(selected_filter, cmap='gray', extent=[-q_max, q_max, -q_max, q_max])
    plt.xlabel('q / ${Å^{-1}}$')
    plt.ylabel('q / ${Å^{-1}}$')
    plt.title('Bandpass filter')
    if save_results:
        plt.savefig(output_folder + 'bandpass_filter.png', bbox_inches='tight')
    if show_figures:
        plt.show()
    plt.close(fig)

    # Get datacube
    datacube = reduce.get_datacube(data.double(), angles, step_size, selected_filter, N, M,
                                   dx=dx, plot_freq=plot_frequency, device='cuda')

    # Send datacube and data back to CPU to save GPU memory
    datacube = datacube.to('cpu')
    data = data.to('cpu')
    torch.cuda.empty_cache()    # housekeeping

    if save_datacube:
        np.save(output_folder + 'datacube_step_size_N' + str(step_size) + '.npy', datacube.numpy())
    if save_results:
        np.save(output_folder + 'stacked_images.npy', data.numpy())

    return datacube, (data.shape)


def find_peaks(datacube, threshold_function, step_size, get_overlap_angles=False, plot_frequency=0, peak_width=1,
               save_peaks_matrix=False, show_figures=False, save_results=False, output_folder=''):
    """

    :param datacube:
    :param threshold_function:
    :param plot_frequency:
    :param peak_width:
    :param save_peaks_matrix:
    :param show_figures:
    :param save_results:
    :param output_folder:
    :return:
    """
    if torch.is_tensor(datacube):
        if datacube.is_cuda:
            datacube = datacube.to('cpu')
        datacube = datacube.numpy()

    print('\n...Finding peaks in datacube')
    peaks_matrix, overlap_angles = peaks.find_datacube_peaks(datacube, threshold_function, width=peak_width,
                                                plot_freq=plot_frequency)

    if save_peaks_matrix:
        np.save(output_folder + 'peaks_matrix_step_size_N' + str(step_size) + '.npy', peaks_matrix)

    # Average number of peaks
    m, n, th = peaks_matrix.shape
    print('     ...Average number of peaks per grid point: ', np.round(np.sum(peaks_matrix) / (m * n), 2))
    print('     ...Maximum number of peaks per grid point: ', np.max(np.sum(peaks_matrix, axis=2)))

    if get_overlap_angles:
        sns.distplot(overlap_angles, hist=True, kde=True, bins=len(set(overlap_angles)), color='darkblue',
                     hist_kws={'edgecolor': 'black', 'linewidth': 0.5})
        plt.xlim([0, 90])
        plt.xlabel('Relative overlap angle / degrees', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        if save_results:
            plt.savefig(output_folder + 'overlap_angle_N64.png')
            np.save(output_folder + 'overlap_angles_step_size_N' + str(step_size) + '.npy', overlap_angles)
        if show_figures:
            plt.show()
        plt.close()


    return peaks_matrix


def calculate_autocorrelations(peaks_matrix, maximum_distance_nm, pixel_size, minimum_counts=1, n_cores=8,
                               show_figures=False, save_results=False, output_folder=''):
    """

    :param peaks_matrix:
    :param maximum_distance_nm:
    :param pixel_size:
    :param minimum_counts:
    :param n_cores:
    :param show_figures:
    :param save_results:
    :param output_folder:
    :return:
    """
    corrs_df = acf.correlations_select_distance_multiprocess(peaks_matrix, maximum_distance_nm, pixel_size,
                                                             min_counts=minimum_counts, n_cores=n_cores)
    corrs_df = corrs_df.loc[:maximum_distance_nm]
    corrs_df.index.name = 'distance_nm'

    ax = sns.heatmap(corrs_df.T, cmap='magma', robust=True, cbar=True,
                     cbar_kws={'label': 'Counts'}, square=False, yticklabels=10)
    ax.set_xlabel('|Δd| / nm', fontsize=14)
    ax.set_ylabel('|Δθ| / degrees', fontsize=14)
    ax.set_ylim([0, 90.5])
    if save_results:
        plt.savefig(output_folder + 'correlations_heatmap_raw.png')
        corrs_df.to_csv(output_folder + 'correlations_raw.csv', index=True)
    if show_figures:
        plt.show()
    plt.close()
    return corrs_df


def normalize_autocorrelations(corrs_df, z_score, binning=False, robust=True,
                               show_figures=False, save_results=False, output_folder=''):
    """

    :param corrs_df:
    :param z_score:
    :param binning:
    :param robust:
    :param show_figures:
    :param save_results:
    :param output_folder:
    :return:
    """
    # Volume normalization only (with or without binning)
    if not z_score:
        corrs_prob_r = acf.correlations_normalization_methods(corrs_df, z_score=False, binning=binning)

        ax = sns.heatmap(corrs_prob_r.T * 100, cmap='magma', robust=robust, cbar=True,
                         cbar_kws={'label': 'P(d) / %'}, square=False, yticklabels=10, vmin=0)
        ax.set_xlabel('|Δd| / nm', fontsize=14)
        ax.set_ylabel('|Δθ| / degrees', fontsize=14)
        ax.set_ylim([0, 90.5])
        if save_results:
            binning_fn = 'binned' if binning else ''
            plt.savefig(output_folder + 'correlations_heatmap_P_r_' + binning_fn + '.png')
            corrs_df.to_csv(output_folder + 'correlations_P_r' + binning_fn + '.csv', index=True)
        if show_figures:
            plt.show()
        plt.close()

        return corrs_prob_r

    # Z-score normalization (with or without binning):
    corrs_z_score = acf.correlations_normalization_methods(corrs_df, z_score=True, binning=binning)

    ax = sns.heatmap(corrs_z_score.T, cmap='RdBu_r', robust=robust, cbar=True,
                     cbar_kws={'label': 'Standard Score'}, square=False, yticklabels=10, center=0)
    ax.set_xlabel('|Δd| / nm', fontsize=14)
    ax.set_ylabel('|Δθ| / degrees', fontsize=14)
    ax.set_ylim([0, 90.5])
    if save_results:
        binning_fn = 'binned' if binning else ''
        plt.savefig(output_folder + 'correlations_heatmap_z_score_' + binning_fn + '.png')
        corrs_df.to_csv(output_folder + 'correlations_z_score_' + binning_fn + '.csv', index=True)
    if show_figures:
        plt.show()
    plt.close()

    return corrs_z_score


def find_clusters(peaks_matrix, maximum_bending, minimum_cluster_size, maximum_separation_pixels,
                  save_results=False, output_folder=''):
    """

    :param peaks_matrix:
    :param maximum_bending:
    :param minimum_cluster_size:
    :param maximum_separation_pixels:
    :param save_results:
    :param output_folder:
    :return:
    """
    print('\n...Finding clusters')
    cluster_map, output, cluster_properties = cluster.find_clusters(peaks_matrix, maximum_bending,
                                                                    minimum_cluster_size, maximum_separation_pixels)

    if save_results:
        df = pd.DataFrame.from_dict(cluster_properties, orient='index')
        df.to_csv(output_folder + 'cluster_properties.csv', index=False)
        np.save(output_folder + 'cluster_map.npy', cluster_map)
        np.save(output_folder + 'cluster_output.npy', output)

    return cluster_map, output, df


def plot_cluster_output(cluster_output, x_length_nm, y_length_nm, angles, perpendicular,
                        show_figures=False, save_results=False, output_folder=''):

    print('     ...Plotting clusters')
    fn = output_folder + 'visualization_cluster_map' if save_results else ''
    angles = angles if perpendicular else angles + 90
    cluster.plot_cluster_map(cluster_output, angles, x_length_nm, y_length_nm, save_fig=fn, show_plot=show_figures)


def plot_director_fields(peaks_matrix, x_length_nm, y_length_nm, angles, perpendicular, colored=False,
                         show_figures=False, save_results=False, output_folder=''):

    print('     ...Plotting director fields')
    fn = output_folder + 'visualization_director_fields' if save_results else ''

    director.plot_director_field(peaks_matrix, angles, x_length_nm, y_length_nm, perpendicular=perpendicular,
                                 colored_lines=colored, save_fig=fn, show_plot=show_figures)


def plot_flow_fields(datacube, peaks_matrix, perpendicular, step_size, seed_density=2, bend_tolerance=10, curve_resolution=2,
                     preview_sparsity=20, line_spacing=1, spacing_resolution=5, angle_spacing_degrees=10,
                     max_overlap_fraction=0.5, show_preview=False, show_figures=False, save_results=False,
                     output_folder=''):

    if torch.is_tensor(datacube):
        if datacube.is_cuda:
            datacube = datacube.to('cpu')
        datacube = datacube.numpy()


    if torch.is_tensor(peaks_matrix):
        if peaks_matrix.is_cuda:
            peaks_matrix = peaks_matrix.to('cpu')
        peaks_matrix = peaks_matrix.numpy()

    m, n, th = datacube.shape
    k = np.min([m, n])

    # Prepare intensity matrix and peaks matrix
    intensity_matrix = datacube[:k, :k, :]
    peaks_matrix_mod = peaks_matrix[:k, :k, :]

    # If the diffraction peaks are perpendicular to the chain direction, rotate the matrix 90 degrees
    prepped_intensity_matrix = flow.prepare_intensity_matrix(intensity_matrix, rotate=perpendicular)

    # Create line seeds at each peak
    line_seeds = flow.seed_lines(peaks_matrix_mod, step_size, seed_density=seed_density)

    # Extend line seeds to create full lines
    propagated_lines = flow.propagate_lines(line_seeds, peaks_matrix_mod, step_size, bend_tolerance,
                                            curve_resolution=curve_resolution, max_grid_length=100)

    if show_preview:
        # Show a preview, using a subset of the propagated lines
        propagated_image = flow.plot_solid_lines(propagated_lines, min_length=2, sparsity=preview_sparsity)
        plt.xlabel('distance / nm')
        plt.ylabel('distance / nm')
        if save_results:
            plt.savefig(output_folder + 'propagated_lines_preview')
        if show_figures:
            plt.show()
        plt.close()

    # Thin out lines, reducing overlap between lines and creating a more homogeneous line density.
    # This prevents the illusion of high density in regions with good alignment, and makes the image more readable.

    trimmed_lines = flow.trim_lines(propagated_lines, prepped_intensity_matrix.shape, step_size,
                                    line_spacing, spacing_resolution, angle_spacing_degrees,
                                    max_overlap_fraction=max_overlap_fraction, min_length=5, verbose=False)
    if show_preview:
        trimmed_image = flow.plot_solid_lines(trimmed_lines)
        plt.xlabel('distance / nm')
        plt.ylabel('distance / nm')
        if save_results:
            plt.savefig(output_folder + 'propagated_lines_preview')
        if show_figures:
            plt.show()
        plt.close()

    # Add intensity data to lines
    line_data = flow.prepare_line_data(trimmed_lines, prepped_intensity_matrix, step_size)
    angle_data = line_data[2, :, :]
    intensity_data = np.array(line_data[4, :, :])
    n_dims, max_length, n_lines = line_data.shape

    # Create amd Format Flow Plots

    # There are many ways to format the plots.  I suggest keeping settings organized in the format below.
    format_codes = [0, 1, 2, 3, 4]

    contrast = 0.1
    gamma = 0.1
    brightness = 1
    for i, format_code in enumerate(format_codes):
        if format_code == 0:
            # Constant color, linewidth, and alpha
            r, g, b = np.zeros((max_length, n_lines)), np.zeros((max_length, n_lines)), np.zeros(
                (max_length, n_lines))
            linewidth = np.ones((max_length, n_lines)) * 0.5
            alpha = np.ones((max_length, n_lines))
        elif format_code == 1:
            # Color by angle, alpha by intensity
            r, g, b = flow.color_by_angle(angle_data)
            linewidth = np.ones((max_length, n_lines)) * 2
            alpha = flow.scale_values(flow.smooth(intensity_data, 9), contrast=contrast, gamma=gamma,
                                      brightness=brightness)
        elif format_code == 2:
            # Solid Color, alpha by intensity
            r, g, b = np.zeros((max_length, n_lines)), np.zeros((max_length, n_lines)), np.zeros(
                (max_length, n_lines))
            linewidth = np.ones((max_length, n_lines)) * 2
            alpha = flow.scale_values(flow.smooth(intensity_data, 9), contrast=contrast, gamma=gamma,
                                      brightness=brightness)
        elif format_code == 3:
            # Color by angle, constant linewidth and alpha
            r, g, b = flow.color_by_angle(angle_data)
            linewidth = np.ones((max_length, n_lines))
            alpha = np.ones((max_length, n_lines))
        elif format_code == 4:
            # Solid Color, alpha by intensity
            r, g, b = np.zeros((max_length, n_lines)), np.zeros((max_length, n_lines)), np.zeros(
                (max_length, n_lines))
            linewidth = flow.scale_values(flow.smooth(intensity_data, 9), contrast=contrast, gamma=gamma,
                                          brightness=brightness)
            alpha = np.ones((max_length, n_lines))

        flow.plot_graded_lines(trimmed_lines, r, g, b, alpha, linewidth)
        # plt.autoscale(enable=True, axis='both', tight=True)
        if save_results:
            plt.savefig(output_folder + 'flow_plots_code_' + str(i) + '.png', dpi=300)
        if show_figures:
            plt.show()
        plt.close()