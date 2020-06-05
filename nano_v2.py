# General packages
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Custom packages - each performs different task
import nano_functions as pm


class Nano(object):
    def __init__(self, params):
        print('Creating object of class Nano.')

        ###############################################################################################################
        # Load initial parameters

        # Loading and saving parameters
        self.input_folder = params['input_folder']
        self.filename = params['filename']
        self.output_folder = params['output_folder']
        # IMPORTANT - define if FFT peaks are parallel or perpendicular to the direction of backbone.
        # Parameter should be set as True for tracking lamellar peaks, or False for tracking backbone peaks.
        self.perpendicular = params['peaks_perpendicular']

        # Image reduction conditions
        self.dx = params['dx']  # resolution in Angstrom / pixel
        self.N = params['N']    # Size of sliding window
        self.M = params['M']    # Size of FFT for sliding window
        self.step_size_pixels = params['step_size_pixels']  # Step size of sliding window
        self.angles = np.arange(-90, 90, step=1) if self.perpendicular else np.arange(0, 180, step=1)   # Angles
        self.q_center = params['q_center']  # Q of interested. In inverse Angstroms.
        self.sigma_q = params['sigma_q']    # FWHM in q of bandwidth of interest. In inverse Angstroms.
        self.sigma_th = params['sigma_th']  # FWHM in theta of bandwidth of interest. In inverse Angstroms.
        self.bandwidth_q = params['bandwidth_q']    # Q-bandwidth for bandpass filter. In inverse Angstroms.
        # Image visualization parameters
        self.preliminary_num_frames = params['preliminary_num_frames']  # Number of frames to use for starting plots.
        self.size_fft_full = params['size_fft_full']    # Size of FFT for full image. Recommended 4096 (always 2^X)

        # Set optional parameters with set choice or default

        # Gamma of HRTEM image visualization
        self.gamma_images = params['gamma_images'] if 'gamma_images' in params.keys() else 1
        # Case of selecting subregion of image (top left corner by default)
        self.subregion = params['subregion'] if 'subregion' in params.keys() else None
        # Case of selecting subregion that is not top left corner
        self.subregion_s0 = params['subregion_s0'] if 'subregion_s0' in params.keys() else 0
        # Color of scatter plots
        self.plot_color = params['plot_color'] if 'plot_color' in params.keys() else 'black'
        # Show figures option - disable when running scripts. Default is true.
        self.show_figures = params['show_figures'] if 'show_figures' in params.keys() else True
        # Save figures option. Default is False
        self.save_results = params['save_results'] if 'save_results' in params.keys() else False

        ###############################################################################################################
        # Initialization of object variables (to be calculated later on)
        # For drift analysis
        self.x_drift, self.y_drift = None, None
        self.x_length_nm, self.y_length_nm = None, None
        self.pixel_size_after_reduction = None

    ###################################################################################################################
    # Class Nano Methods
    # NOTE: during computations with raw data using the various methods described below, send data to GPU before
    # computation and then retrieve to CPU. This has to be done in order to save memory in GPU.

    def read_image(self):

        raw_data = pm.read_raw_data(self.input_folder, self.filename, subregion=self.subregion, s0=self.subregion_s0)

        return raw_data

    def visualize(self, data_frames, plot_lineout=False):

        pm.initial_visualization(data_frames, self.preliminary_num_frames, self.size_fft_full, self.dx,
                              self.gamma_images, plot_lineout=plot_lineout, show_figures=self.show_figures,
                              save_results=self.save_results, output_folder=self.output_folder)

    def bandpass_filter_frames(self, data_frames, beta=0.1):

        # Overwriting self.data_frames variable with bandpass filter version of stack. This operation does not increase
        # allocated memory in GPU.
        bp_filtered_data = pm.apply_bandpass_filter_to_image_stack(data_frames, self.q_center, self.bandwidth_q,
                                                                self.dx, beta=beta)
        print('   ...Data has been modified to bandpass filtered images of shape: {0}'.format(data_frames.shape))

        return bp_filtered_data

    def stack_properties(self, data_frames_bp_filtered):

        self.x_drift, self.y_drift = \
            pm.analyze_stack_properties(data_frames_bp_filtered, self.plot_color, self.dx, show_figures=self.show_figures,
                                     save_results=self.save_results, output_folder=self.output_folder)

    def correct_drift_frames(self, data_frames, data_frames_bp_filtered, max_drift_allowed, first_frame, last_frame,
                             save_corrected_results=False):

        data = pm.correct_drift(data_frames, data_frames_bp_filtered, max_drift_allowed, first_frame, last_frame,
                             self.x_drift, self.y_drift, self.dx, self.size_fft_full, show_figures=self.show_figures,
                             save_results=self.save_results, output_folder=self.output_folder,
                             gamma_images=self.gamma_images, save_corrected_results=save_corrected_results)

        return data

    def stack_selected_list_frames(self, data_frames_bp_filtered, frames):

        data = pm.stack_selected_frames(data_frames_bp_filtered, frames, self.dx, self.size_fft_full, self.show_figures,
                                        self.save_results, self.output_folder, self.gamma_images)

        return data

    def get_datacube(self, data, plot_frequency=0, number_frames=None, save_datacube=False):

        datacube, (data_shape) = pm.reduce_data(data, self.q_center, self.sigma_q, self.sigma_th, self.dx, self.bandwidth_q,
                                  self.angles, self.N, self.M, self.step_size_pixels,
                                  number_frames=number_frames ,save_datacube=save_datacube,
                                  plot_frequency=plot_frequency, show_figures=self.show_figures,
                                  save_results=self.save_results, output_folder=self.output_folder)

        # Get pixel size after data reduction
        self.x_length_nm = data_shape[0] * self.dx / 10
        self.y_length_nm = data_shape[1] * self.dx / 10
        self.pixel_size_after_reduction = np.round(self.x_length_nm / datacube.shape[0], 2)

        print('pixel size after datacube reduction: ' , self.pixel_size_after_reduction)

        return datacube

    def get_peaks(self, datacube, threshold_function, get_overlap_angles=False, plot_frequency=0, peak_width=1,
                  save_peaks_matrix=False):

        peaks_matrix = pm.find_peaks(datacube, threshold_function, self.N, self.step_size_pixels,
                                     get_overlap_angles=get_overlap_angles, plot_frequency=plot_frequency,
                                     peak_width=peak_width, save_peaks_matrix=save_peaks_matrix,
                                     show_figures=self.show_figures, save_results=self.save_results,
                                     output_folder=self.output_folder)

        return peaks_matrix

    def get_autocorrelations(self, peaks_matrix, maximum_distance_nm, minimum_counts=1, n_cores=8):

        corrs_df = pm.calculate_autocorrelations(peaks_matrix, maximum_distance_nm, self.pixel_size_after_reduction,
                                                 minimum_counts=minimum_counts, n_cores=n_cores,
                                                 show_figures=self.show_figures, save_results=self.save_results,
                                                 output_folder=self.output_folder)

        return corrs_df

    def autocorrelations_normalize(self, corrs_df, z_score=False, binning=False, robust=True):

        corrs_norm = pm.normalize_autocorrelations(corrs_df, z_score=z_score, binning=binning, robust=robust,
                                                   show_figures=self.show_figures,save_results=self.save_results,
                                                   output_folder=self.output_folder)

        return corrs_norm

    def get_domains(self, peaks_matrix, maximum_bending, minimum_cluster_size, maximum_separation_pixels):

        cluster_map, output, cluster_df = pm.find_clusters(peaks_matrix, maximum_bending, minimum_cluster_size,
                                                             maximum_separation_pixels, save_results=self.save_results,
                                                             output_folder=self.output_folder)
        if self.pixel_size_after_reduction is not None:
            cluster_df['domain_size_nm'] = np.sqrt(cluster_df['number_pixels'] * self.pixel_size_after_reduction**2)

        return cluster_map, output, cluster_df

    def plot_domains(self, cluster_output):

        pm.plot_cluster_output(cluster_output, self.x_length_nm, self.y_length_nm, self.angles, self.perpendicular,
                               self.show_figures, self.save_results, self.output_folder)

    def director_fields(self, peaks_matrix, colored=False):

        pm.plot_director_fields(peaks_matrix, self.x_length_nm, self.y_length_nm, self.angles, self.perpendicular,
                                colored=colored, show_figures=self.show_figures, save_results=self.save_results,
                                output_folder=self.output_folder)

    def flow_fields(self, datacube, peaks_matrix, seed_density=2, bend_tolerance=10, curve_resolution=2,
                     preview_sparsity=20, line_spacing=1, spacing_resolution=5, angle_spacing_degrees=10,
                     max_overlap_fraction=0.5, show_preview=False):

        step_size = self.step_size_pixels * self.dx / 10
        pm.plot_flow_fields(datacube, peaks_matrix, self.perpendicular, step_size, seed_density=seed_density,
                            bend_tolerance=bend_tolerance, curve_resolution=curve_resolution,
                            preview_sparsity=preview_sparsity, line_spacing=line_spacing,
                            spacing_resolution=spacing_resolution, angle_spacing_degrees=angle_spacing_degrees,
                            max_overlap_fraction=max_overlap_fraction, show_preview=show_preview,
                            show_figures=self.show_figures, save_results=self.save_results,
                            output_folder=self.output_folder)