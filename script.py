import sys
# To load custom packages stored in general directory
sys.path.append('/home/camila/hrtem_python_packages/branch_definingNanoClass')
import nano
import numpy as np

params = {}
params['input_folder'] = '/media/super/Toshiba8Tb/camila/kornberg_TEM_data/20190714_Bao_C01_FF-DIO/'
params['filename'] = 'C01_FF-DIO_area5_00005.mrc'
params['save_figures'] = True

params['dx'] = 1.924 # Angstrom/pixel
params['N'] = 128
params['M'] = 512
params['step_size_pixels'] = 16
params['sigma_q'] = 0.02
params['sigma_th'] = 0.02
params['bandwidth_q'] = 0.03  # original 0.03

params['preliminary_num_frames'] = 8
params['size_fft_full'] = 4096

# Optional parameters
params['gamma_images'] = 0.75
params['show_figures'] = False ## Default is True

params['subregion'] = 1060
params['subregion_s0'] = 1040

params['plot_color'] = 'red'
params['output_folder'] = '/home/camila/hrtem_python_packages/branch_definingNanoClass/FFDIO_test/donor/'
params['q_center'] = 0.29

threshold = 10
min_cluster_size = 3
max_separation = 2.5

def threshold_function(y):
    """Defines minimum threshold for prominence of peak during peak searching algorithm.
    Can change functionality as desired.
    This function is passed as an argument to peaks.find_datacube_peaks()
    """
    return np.percentile(y, 90)


donor = nano.Nano(params)
donor.initial_visualization(plot_lineout=True)
donor.bandpass_filter_data()
donor.stack_analysis()
donor.correct_drift(max_drift_allowed=5, first_frame=0, last_frame=8, save_array=True)
donor.reduce_data()
donor.find_peaks(threshold_function)
donor.find_clusters(threshold, min_cluster_size, max_separation, save_output=True)
donor.final_visualizations(clusters=True, director_fields=True)
donor.flow_fields_visualization(peaks_parallel_to_chain=False, seed_density=2, bend_tolerance=20,
                                  curve_resolution=2, preview_sparsity=20, line_spacing=1, spacing_resolution=5,
                                  angle_spacing_degrees=10, max_overlap_fraction=0.5)
del donor

print('\n \nAnalyzing acceptor')

params['plot_color'] = 'blue'
params['output_folder'] = '/home/camila/hrtem_python_packages/branch_definingNanoClass/test_large/acceptor/'
params['q_center'] = 0.235

acceptor = nano.Nano(params)
acceptor.initial_visualization(plot_lineout=True)
acceptor.bandpass_filter_data()
acceptor.stack_analysis()
acceptor.correct_drift(max_drift_allowed=5, first_frame=0, last_frame=8, save_array=True)
acceptor.reduce_data()
acceptor.find_peaks(threshold_function)
acceptor.find_clusters(threshold, min_cluster_size, max_separation, save_output=True)
acceptor.final_visualizations(clusters=True, director_fields=True)
acceptor.flow_fields_visualization(peaks_parallel_to_chain=False, seed_density=2, bend_tolerance=20,
                                  curve_resolution=2, preview_sparsity=20, line_spacing=1, spacing_resolution=5,
                                  angle_spacing_degrees=10, max_overlap_fraction=0.5)
