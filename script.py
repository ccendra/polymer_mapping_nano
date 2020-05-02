import sys
# To load custom packages stored in general directory
sys.path.append('/home/camila/hrtem_python_packages/branch_definingNanoClass')
import nano

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
params['output_folder'] = '/home/camila/hrtem_python_packages/branch_definingNanoClass/test/donor/'
params['q_center'] = 0.29

threshold = 5
min_cluster_size = 10
max_separation = 2.5

donor = nano.Nano(params)
donor.raw_data_processing()
donor.stack_analysis()
donor.correct_drift(5)
donor.reduce_data()
donor.find_peaks()
donor.find_clusters(threshold, min_cluster_size, max_separation, save_output=True)
donor.final_visualizations()
del donor

print('\n \nAnalyzing acceptor')

params['plot_color'] = 'blue'
params['output_folder'] = '/home/camila/hrtem_python_packages/branch_definingNanoClass/test_large/acceptor/'
params['q_center'] = 0.235

acceptor = nano.Nano(params)
acceptor.raw_data_processing()
acceptor.stack_analysis()
acceptor.correct_drift(5)
acceptor.reduce_data()
acceptor.find_peaks()

acceptor.find_clusters(threshold, min_cluster_size, max_separation, save_output=True)
acceptor.final_visualizations()

# add finding peaks matrix
# add getting clusters --> this is all we need withouth representation
# QS: best way to store and organize clusters ?

# Once have cluster, want to start looking at different types of clusters