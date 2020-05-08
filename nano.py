# General packages
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom packages
import reduce_data as reduce
import auxiliary_functions as aux
import plot_functions as plot
import drift_correction as drift
import peak_fitting as peaks
import director_fields as director
import flow_fields as flow
import clustering as cluster

# Pytorch setup
device = torch.device('cuda')
# print('CUDA device: ', torch.cuda.get_device_name(0))


class Nano(object):
    def __init__(self, params):
        print('Creating object of class Nano.')

        ###############################################################################################################
        # Load initial parameters
        self.input_folder = params['input_folder']
        self.filename = params['filename']
        self.output_folder = params['output_folder']

        self.dx = params['dx']
        self.N = params['N']
        self.M = params['M']
        self.step_size_pixels = params['step_size_pixels']
        self.angles = np.arange(-90, 90, step=1)

        self.q_center = params['q_center']
        self.sigma_q = params['sigma_q']
        self.sigma_th = params['sigma_th']
        self.bandwidth_q = params['bandwidth_q']

        self.preliminary_num_frames = params['preliminary_num_frames']
        self.size_fft_full = params['size_fft_full']

        # Set optional parameters with set choice or default
        self.gamma_images = params['gamma_images'] if 'gamma_images' in params.keys() else 1
        self.subregion = params['subregion'] if 'subregion' in params.keys() else None
        self.subregion_s0 = params['subregion_s0'] if 'subregion_s0' in params.keys() else 0
        self.plot_color = params['plot_color'] if 'plot_color' in params.keys() else 'black'
        self.show_figures = params['show_figures'] if 'show_figures' in params.keys() else True
        self.save_figures = params['save_figures'] if 'show_figures' in params.keys() else False
        self.perpendicular = params['peaks_perpendicular'] if 'peaks_perpendicular' in params.keys() else True
        self.colored_lines = params['colored_lines'] if 'colored_lines' in params.keys() else False

        ###############################################################################################################
        # Load raw data
        # Returns tensor in CPU with raw data
        self.data_frames = read_raw_data(self.input_folder, self.filename, subregion=self.subregion, s0=self.subregion_s0)
        # NOTE: during computations with raw data using the various methods described below, send data to GPU before
        # computation and then retrieve to CPU. This has to be done in order to save memory in GPU.

        ###############################################################################################################
        # Initialization of object variables (to be calculated later on)
        self.x_drift, self.y_drift = np.zeros(self.data_frames.shape[0]), np.zeros(self.data_frames.shape[0]) # for drift analysis
        self.data_stacked = None
        self.datacube = None
        self.peaks_matrix = None
        self.cluster_map = None
        self.cluster_output=None
        self.cluster_properties = {}

    ###################################################################################################################
    # Class Nano Methods

    def initial_visualization(self, plot_lineout=True):
        """ Perfoms initial visualization of raw data prior to performing any transformations. Method stacks a
        pre-determined number of raw frames and plots stacked image, FFT, and azimuthally integrated powder lineout
        (optional).
        :param plot_lineout: whether to compute azimuthally integrated powder spectra (this is a medium expensive
        computation in GPU). Default is True.
        """
        print('\n...Performing initial visualizations \n '
              'Note: transformations here are temporary and only for visualization purposes.')
        if not self.data_frames.is_cuda:
            # Send data to GPU
            stacked_data = self.data_frames.to(device)

        # Stack n first frames of data.
        stacked_data = torch.sum(stacked_data[:self.preliminary_num_frames, :, :], dim=0)
        print('     ...The first {0} image frames have been stacked and image size is: {1}'.format(self.preliminary_num_frames,
                                                                                           stacked_data.shape))
        # Determine output filenames in case figures are saved
        if self.save_figures:
            save_fig = [self.output_folder + 'initial_visualization_hrtem',
                        self.output_folder + 'initial_visualization_fft',
                        self.output_folder + 'initial_visualization_IvsQ_lineout']
        else:
            save_fig = ['', '', '']

        # Plot HRTEM
        plot.hrtem(stacked_data.cpu().numpy(), size=10, gamma=self.gamma_images, vmax=0,
                   colorbar=False, dx=self.dx, save_fig=save_fig[0], show_plot=self.show_figures)

        # Get FFT (using GPU)
        img_fft_gpu = reduce.tensor_fft(stacked_data, self.size_fft_full)
        plot.fft(img_fft_gpu.cpu(), self.size_fft_full, q_contour_list=[], dx=self.dx,
                 save_fig=save_fig[1], show_plot=self.show_figures)

        # Calculate azimuthally integrated powder lineout (using GPU)
        if plot_lineout:
            q_increments = 0.005  # Can be increased to 0.01 for coarser (but faster) calculation
            q_bandwidth = 0.005   # Can be increased to 0.01 for coarser (but faster) calculation
            x_powder, y_powder = reduce.extract_intensity_q_lineout(img_fft_gpu, q_increments, q_bandwidth, self.dx)
        plot.intensity_q_lineout(x_powder, y_powder, save_fig=save_fig[2], show_plot=self.show_figures)

    def bandpass_filter_data(self):
        """
        Apply bandpass filter to all frames in raw data. Computes stack of banpass filtered real space frames and
        stores it in object.data_frames variable.
        """
        print('\n...Filtering raw data with bandpass filter')
        # Send raw data to GPU
        data = self.data_frames.to(device)

        # Make raised cosine window
        n_frames, m, n = data.shape
        _, rc_window_m = reduce.raised_cosine_window_np(m, beta=0.1)
        _, rc_window_n = reduce.raised_cosine_window_np(n, beta=0.1)
        window = torch.from_numpy(np.outer(rc_window_m, rc_window_n)).to(device)  # window shape is (m, n)

        # Apply raised cosine window to 3D data
        data = data * torch.reshape(window, (1, m, n)).double()
        # window = window.cpu()
        del window

        # Pad image if m != n (case for full images)
        s = max(m, n)
        if m != n:
            pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
            data = pad(data)

        # Apply bandpass filter to each individual frame
        print('   ...Applying bandpass filter to all frames in image')
        for i in range(n_frames):
            data[i, :, :] = reduce.bandpass_filtering_image(data[i, :, :], self.q_center,
                                                            self.bandwidth_q, self.dx, beta=0.1)
        # Remove padding
        data = data[:, :m, :n]
        # Send data back to CPU
        self.data_frames = data.cpu()
        print('   ...Data has been modified to bandpass filtered images and has shape: {0}'.format(data.shape))

    def stack_analysis(self, plot_fft=False):
        print('\n...Analyzing full-stack behavior')
        if not self.data_frames.is_cuda:
            # Send data to GPU
            self.data_frames = self.data_frames.to(device)

        n_frames, m, n = self.data_frames.shape
        s = max(m, n)

        # Pad image if m != n (case for full images)
        if m != n:
            pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
            self.data_frames = pad(self.data_frames)
            # print('Data has temporarily been padded to shape: ', self.data_frames.shape)

        print('     ...Getting integrated powder intensities vs. frame number.')
        fft_integrated_intensity = torch.zeros(n_frames)
        for i in range(n_frames):
            # Do Fourier transform of bandpass filtered image
            frame_fft_gpu = reduce.tensor_fft(self.data_frames[i, :, :], s)
            if plot_fft:
                plot.fft(frame_fft_gpu.cpu(), s, q_contour_list=[], size=5, show=True)
            # Store integrated powder spectrum at bandwidth q
            fft_integrated_intensity[i] = torch.sum(frame_fft_gpu)

        fig = plt.figure()
        plt.scatter(np.arange(n_frames), fft_integrated_intensity.cpu(), color=self.plot_color)
        plt.xlabel('frame number')
        plt.ylabel('Integrated counts')
        plt.ylim([torch.min(fft_integrated_intensity)*0.8, torch.max(fft_integrated_intensity)*1.1])
        plt.savefig(self.output_folder + 'stack_analysis_IntegratedCounts_vs_FrameNumber.png', bbox_inches='tight')
        if self.show_figures:
            plt.show()
        else:
            plt.close(fig)

        print('     ...Tracking drift between frames.')
        self.x_drift, self.y_drift = drift.track_drift(self.data_frames, s, verbose=False)

        fig = plt.figure()
        plt.scatter(np.arange(n_frames), self.x_drift, color=self.plot_color)
        plt.plot(np.arange(n_frames), self.x_drift, color='black', linewidth=0.5)
        plt.ylabel('Image drift in x̄ / pixels', fontsize=14)
        plt.xlabel('image #', fontsize=14)
        plt.savefig(self.output_folder + 'stack_analysis_ImageDrift_x_direction.png', bbox_inches='tight')
        if self.show_figures:
            plt.show()
        else:
            plt.close(fig)

        fig = plt.figure()
        plt.scatter(np.arange(n_frames), self.y_drift, color=self.plot_color)
        plt.plot(np.arange(n_frames), self.y_drift, color='black', linewidth=0.5)
        plt.ylabel('Image drift in ȳ / pixels', fontsize=14)
        plt.xlabel('image #', fontsize=14)
        plt.savefig(self.output_folder + 'stack_analysis_ImageDrift_y_direction.png', bbox_inches='tight')
        if self.show_figures:
            plt.show()
        else:
            plt.close(fig)

        drift.plot_2d_drift(self.x_drift, self.y_drift, dx=self.dx, lines=False,
                            save_fig=self.output_folder + 'stack_analysis_2D_drift', show_plot=self.show_figures)

        # Send data back to CPU
        self.data_frames = self.data_frames.cpu()

    def correct_drift(self, max_drift_allowed, first_frame, last_frame, save_array=True):
        print('\n...Correcting drift between frames. Maximum drift allowed is {0} pixels in either '
              'x or y directions.'.format(max_drift_allowed))

        if not self.data_frames.is_cuda:
            # Send data to GPU
            self.data_frames = self.data_frames[first_frame:last_frame, :, :].to(device)

        # Reading raw data again to illustrate behavior post-drift
        data = read_raw_data(self.input_folder, self.filename, subregion=self.subregion, s0=self.subregion_s0)
        data = data[first_frame:last_frame, :, :]

        n_frames, m, n = data.shape
        padding = max_drift_allowed + 1

        data_corrected = np.zeros((n_frames, m + 2*padding, n + 2*padding))
        data_corrected_bp_filtered = torch.zeros(n_frames, m + 2 * padding, n + 2 * padding).to(device)

        ct = 0
        for i in range(n_frames):
            ux = self.x_drift[i]
            uy = self.y_drift[i]
            if np.abs(ux) <= max_drift_allowed and np.abs(uy) <= max_drift_allowed:
                data_corrected[ct, (padding - ux):-(padding + ux), (padding - uy):-(padding + uy)] = data[i, :m, :n]
                data_corrected_bp_filtered[ct, (padding - ux):-(padding + ux), (padding - uy):-(padding + uy)] = \
                                                                                                self.data_frames[i, :m, :n]
                ct += 1

        size = padding + max_drift_allowed
        data_corrected = data_corrected[:ct, size:-size, size:-size]
        # Send data back to CPU
        self.data_frames = data_corrected_bp_filtered[:ct, size:-size, size:-size].cpu()
        print('     ...Data has been drift-corrected and new shape is: ', data_corrected.shape)

        # Overwrite with stacked data for image
        data_corrected = np.sum(data_corrected, axis=0)
        plot.hrtem(data_corrected, size=10, gamma=self.gamma_images, vmax=0, colorbar=False, dx=self.dx,
                   save_fig=self.output_folder + 'drift_corrected_hrtem', show_plot=self.show_figures)
        # Plot FFT
        img_fft_gpu = reduce.tensor_fft(torch.from_numpy(data_corrected).to(device), self.size_fft_full)
        plot.fft(img_fft_gpu.cpu(), self.size_fft_full, q_contour_list=[],
                 save_fig=self.output_folder+'drift_corrected_fft', show_plot=self.show_figures)

        if save_array:
            np.save(self.output_folder + 'data_frames_drift_corrected.npy', data_corrected)
            print('     ...Drift corrected image frames have been saved.')

    def select_frames(self, frames):
        if not self.data_frames.is_cuda:
            # Send data to GPU
            self.data_frames = self.data_frames.to(device)

        data = read_raw_data(self.input_folder, self.filename, subregion=self.subregion, s0=self.subregion_s0)

        if len(frames) == 2:
            first_frame = frames[0]
            last_frame = frames[1]
            print('\n...Selecting frame stack from frame # {0} to frame # {1}'.format(first_frame, last_frame))
            self.data_frames = self.data_frames[first_frame:last_frame, :, :]
            data = torch.sum(data[first_frame:last_frame, :, :], dim=0)
        else:
            print('\n...Selecting frames {0} in stack.'.format(frames))
            self.data_frames = self.data_frames[frames, :, :]
            data = torch.sum(data[frames, :, :], dim=0)

        # Send data back to CPU
        self.data_frames = self.data_frames.cpu()

        plot.hrtem(data.numpy(), size=10, gamma=self.gamma_images, vmax=0, colorbar=False, dx=self.dx,
                   save_fig=self.output_folder + 'selected_stack_sum', show_plot=self.show_figures)

        img_fft_gpu = reduce.tensor_fft(data, self.size_fft_full)
        plot.fft(img_fft_gpu.cpu(), self.size_fft_full, q_contour_list=[],
                 save_fig=self.output_folder+'selected_stack_sum_fft', show_plot=self.show_figures)

    def reduce_data(self, number_frames=None, plot_frequency=0, save_datacube=True):
        print('\nPerforming data reduction.')

        if not self.data_frames.is_cuda:
            self.data_frames = self.data_frames.to(device)

        # Stack frames and account for different situations
        if len(self.data_frames.shape) == 3:
            if number_frames:
                # Case where we want to stack a select number of frames. This happens when there was no drift
                # correction or if want to stack fewer frames than the drift-corrected data.
                print('     ...Stacking first {0} frames'.format(number_frames))
                self.data_stacked = torch.sum(self.data_frames[:number_frames, :, :], dim=0)
            else:
                print('     ...Stacking all {0} frames'.format(self.data_frames.shape[0]))
                # Stack all frames together
                self.data_stacked = torch.sum(self.data_frames, dim=0)

        print('\n...Getting datacube.')
        # Getting filters (bandpass and gaussian, then combine)
        gaussian_filter = reduce.gaussian_q_filter(self.q_center, self.sigma_q, self.sigma_th, self.M, self.dx)
        bandpass_filter = reduce.bandpass_filter(self.M, self.q_center - self.bandwidth_q,
                                                 self.q_center + self.bandwidth_q, self.dx)
        selected_filter = gaussian_filter * bandpass_filter

        q_max = np.pi / self.dx

        fig = plt.figure()
        plt.imshow(selected_filter, cmap='gray', extent=[-q_max, q_max, -q_max, q_max])
        plt.xlabel('q / ${Å^{-1}}$')
        plt.ylabel('q / ${Å^{-1}}$')
        plt.title('Bandpass filter')
        plt.savefig(self.output_folder + 'bandpass_filter.png', bbox_inches='tight')
        if self.show_figures:
            plt.show()
        else:
            plt.close(fig)

        # Get datacube
        self.datacube = reduce.get_datacube(self.data_stacked.double(), self.angles, self.step_size_pixels,
                                            selected_filter, self.N, self.M, dx=self.dx,
                                            plot_freq=plot_frequency, device=device)

        # Send datacube and data back to CPU to save GPU memory
        self.datacube = self.datacube.cpu().numpy()
        self.data_frames = self.data_frames.cpu()
        self.data_stacked = self.data_stacked.cpu()

        if save_datacube:
            np.save(self.output_folder + 'datacube.npy', self.datacube)

    def find_peaks(self, threshold_function, plot_frequency=0):

        print('\n...Finding peaks in datacube')
        peaks_matrix, _ = peaks.find_datacube_peaks(self.datacube, threshold_function, width=10,
                                                          plot_freq=plot_frequency)
        np.save(self.output_folder + 'peaks_matrix.npy', peaks_matrix)

        # Average number of peaks
        m, n, th = peaks_matrix.shape
        print('     ...Average number of peaks per grid point: ', np.round(np.sum(peaks_matrix) / (m * n), 2))
        print('     ...Maximum number of peaks per grid point: ', np.max(np.sum(peaks_matrix, axis=2)))

        self.peaks_matrix = peaks_matrix

    def find_clusters(self, threshold, min_cluster_size, max_separation, save_output=True):
        print('\n...Finding clusters')
        cluster_map, output, cluster_properties = cluster.find_clusters(self.peaks_matrix, threshold,
                                                                        min_cluster_size, max_separation)

        if save_output:
            df = pd.DataFrame.from_dict(cluster_properties, orient='index')
            df.to_csv(self.output_folder + 'cluster_properties.csv', index=False)
            np.save(self.output_folder + 'cluster_map.npy', cluster_map)
            np.save(self.output_folder + 'cluster_output.npy', output)

        self.cluster_map = cluster_map
        self.cluster_output = output
        self.cluster_properties = cluster_properties

    def final_visualizations(self, clusters=True, director_fields=True, flow_fields=False):
        x_length_nm = self.data_stacked.shape[1] * self.dx / 10
        y_length_nm = self.data_stacked.shape[0] * self.dx / 10
        print('\n...Plotting final visualizations')

        if clusters:
            print('     ...Plotting clusters')
            cluster.plot_cluster_map(self.cluster_output, self.angles, x_length_nm, y_length_nm,
                                     save_fig=self.output_folder + 'final_visualizations_cluster_map',
                                     show_plot=self.show_figures)

        if director_fields:
            print('     ...Plotting director fields')
            director.plot_director_field(self.peaks_matrix, self.angles, x_length_nm, y_length_nm,
                                          perpendicular=self.perpendicular, colored_lines=self.colored_lines,
                                          save_fig=self.output_folder + 'final_visualizations_director_fields',
                                          show_plot=self.show_figures)


def read_raw_data(input_folder, filename, subregion=None, s0=0):
    file_type = filename.split('.')[-1]
    fn = input_folder + filename
    raw_data = None
    if file_type == 'mrc':
        print('\n...Opening .mrc file')
        raw_data = aux.read_mrc(fn)
    elif file_type == 'tif':
        print('\n...Opening .tif file')
        raw_data = aux.read_tif(fn)
    else:
        print('Invalid file type. Accepted files are either .mrc or .tif')

    if subregion:
        raw_data = raw_data[:, s0:subregion+s0, s0:subregion+s0]

    return torch.from_numpy(raw_data)

#
# def raw_data_preprocesing(data, q_center, bandwidth_q, dx):
#     print('...Preprocessing')
#     # Send raw data to GPU
#     data = torch.from_numpy(data).to(device)
#     n_frames, m, n = data.shape
#
#     # Make raised cosine window
#     _, rc_window_m = reduce.raised_cosine_window_np(m, beta=0.1)
#     _, rc_window_n = reduce.raised_cosine_window_np(n, beta=0.1)
#     window = torch.from_numpy(np.outer(rc_window_m, rc_window_n)).to(device) # window shape is (m, n)
#
#     data = data * torch.reshape(window, (1, m, n)).double()
#     window = window.cpu()
#     del window
#
#     # Pad image if m != n (case for full images)
#     s = max(m, n)
#     if m != n:
#         pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
#         data = pad(data)
#
#     # Get bandpass filtered image
#     print('   ...Applying bandpass filter to all frames in image')
#
#     # Apply bandpass filter to each individual frame
#     for i in range(n_frames):
#         data[i, :, :] = reduce.bandpass_filtering_image(data[i, :, :], q_center,
#                                                                      bandwidth_q, dx, beta=0.1)
#     data = data[:, :m, :n]
#     data = data.cpu()
#     print('   ...Data has been modified to bandpass filtered images and has shape: {0}'.format(data.shape))
#
#     return data

