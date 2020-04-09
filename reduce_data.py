import numpy as np
import time
import torch
from scipy import ndimage
import matplotlib.pyplot as plt


def tensor_fft(tensor, s, print_time=False):
    """
    Returns powder spectrum of 2D tensor (image) using PyTorch implementation.
    NOTE: location of operation (GPU or CPU) is determined by location of input tensor.
    Send tensor to GPU prior to using this function to perform operations in GPU (i.e. tensor.to(cuda))
    Args:
        tensor: 2D tensor (image)
        s: output size of FFT (s x s). tensor is padded with zeros prior to performing FFT operation
        to specified output size.
    Returns:
        fft: powder spectra (real^2 + complex^2) tensor of size (s x s) with Fourier Transform.
             DC frequency component is set in center of tensor.
    """
    start = time.time()
    m, n = tensor.shape
    # normalize tensor prior to performing FFT
    tensor = normalize_tensor(tensor)
    # pad tensor with zeros function to get (s x s) tensor
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
    padded_tensor = pad(tensor)
    # get fft of padded tensor using torch.rfft function
    hrtem_fft_gpu = torch.rfft(padded_tensor, 2, normalized=True, onesided=False)
    # adding up real and imaginary components in FT
    fft = hrtem_fft_gpu[:, :, 0]**2 + hrtem_fft_gpu[:, :, 1]**2
    # shift zero frequency to center of image
    fft = tensor_shift_fft(fft)
    if print_time:
        print('Time: ', np.round(time.time() - start, 5))
    return fft


def normalize_tensor(tensor):
    """
    Tensor normalization operation. Tensor/mean - 1."""
    tensor = tensor / torch.mean(tensor) - 1
    return tensor


def tensor_shift_fft(fft):
    """
    Shift zero frequency spatial frequency component to center of 2D image. For Pytorch implementation
    Args:
        fft: 2D FFT obtained using torch_fft function
    Returns:
        shifted FFT with DC frequency component in center of image.
    """
    m, n = fft.shape
    out = torch.cat((fft[-m//2:], fft[:-m//2]), dim=0)
    return torch.cat((out[:, -n//2:], out[:, :-n//2]), dim=1)


def bandpass_filter(M, q_low, q_high, dx):
    """
    Generates bandpass filter of size MxM for range of frequencies (q_low, q_high).
    Arguments:
        M: size in pixels of output bandpass filter(in fourier space)
        q_low: minimum q value in inverse Angstroms
        q_high: maximum q value in inverse Angstroms
        dx: HRTEM image resolution in Angstroms/pixel
    Returns:
        mask: numpy array size (M, M) with ones for pixels between selected bandwidth of q values and zeros elsewhere.
    """
    rows, cols = M, M
    crow, ccol = (M-1) / 2, (M-1) / 2

    f_low_pixels = get_q_pixels(q_low, M, dx)
    f_high_pixels = get_q_pixels(q_high, M, dx)

    mask = np.zeros((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= f_low_pixels ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= f_high_pixels ** 2))
    mask[mask_area] = 1

    return mask


def extract_intensity_q_lineout(fft_gpu, q_increments, q_bandwidth, dx, device='cuda'):
    """
    Function performs azimuthal intensity integrations as function of q for selected q bandwidth.
    :param fft_gpu: Pytorch tensor fourier transform
    :param q_increments: integration coarseness
    :param q_bandwidth: width of bandpass filter
    :param dx: resolution in Angstrom / pixel
    :param device: cuda device
    :return: x, y lists with values for x (q) and y (counts) to construct lineout
    """
    qi = 0
    x = []
    y = []
    M, M = fft_gpu.shape

    while qi < 1.6:
        bp_filter = torch.from_numpy(bandpass_filter(M, qi - q_bandwidth, qi + q_bandwidth, dx)).to(device)
        product = bp_filter.double() * fft_gpu / torch.sum(bp_filter.double())
        x.append(qi)
        y.append(torch.sum(product).cpu().numpy())
        qi += q_increments

    return x, y


def make_gaussian_filter_tensor(single_filter, angles, device='cuda'):
    """
    Generates 3D tensor of size (M, M, number_angles) with stack of rotated Fourier filters to be used for
    datacube extraction.
    :param single_filter: Initial filter as numpy array of size (M, M)
    :param angles: list or numpy array with rotation angles
    :param device: CUDA device send final output tensor
    :return: 3D filters tensor with series of filters for azimuthal information extraction
    """
    M = single_filter.shape[0]
    filters_np = np.zeros((M, M, len(angles)))
    for i in range(len(angles)):
        filters_np[:, :, i] = ndimage.rotate(single_filter, angles[i], reshape=False)

    return torch.from_numpy(filters_np).to(device)


def get_datacube(img_gpu, angles, step_size, selected_filter, bandpass, N, M, dx=1.924, plot_freq=0, device='cuda'):
    """ Get intensity - theta 4D array. Saves 4D array output.
    Arguments:
        img_gpu: GPU tensor of raw image
        angles: np array with angles to probe
        step_size: size of steps during 'rastering'
        selected_filter: gaussian filter - numpy of size (M, M) (potentially combined with bandpass filter) to rotate and use for analysis
        bandpass: bandpass filter - numpy of size (M, M)
        N: Size of nano-image in pixels
        M: size of FFT
        device: name of GPU CUDA device
        dx: pixel resolution in Angstrom/pixel
        plot_freq: integer to determine frequency to plot window analysis. No plots shown if set to zero.
    Returns:
        datacube: 4D pyTorch tensor containing integrated intesity for every (row, col, theta)
    """
    start_time = time.time()

    filters_tensor = make_gaussian_filter_tensor(selected_filter, angles, device)

    size_rows = int((img_gpu.shape[0] - N) / step_size + 1)
    size_cols = int((img_gpu.shape[1] - N) / step_size + 1)

    datacube = torch.from_numpy(np.zeros((size_rows, size_cols, len(angles)))).to(device)

    hanning_window = torch.from_numpy(np.outer(np.hanning(N), np.hanning(N))).to(device)
    bandpass = torch.from_numpy(bandpass).to(device).double()

    i0 = 0
    m, n = img_gpu.shape
    ct = 0
    row = 0

    for i in range(N, m + 1, step_size):
        j0 = 0
        col = 0

        for j in range(N, n + 1, step_size):
            # Normalize window signal (divide by mean and substract 1) and multiply by hanning window
            mini = normalize_tensor(img_gpu[i0:i, j0:j]) * hanning_window
            fft = tensor_fft(mini, s=M)
            intensity_theta = get_orientation_torch(fft, filters_tensor, device)
            datacube[row, col, :] = intensity_theta

            if plot_freq != 0 and ct % plot_freq == 0:
                fft_masked = fft * bandpass
                subplot_mini(mini.cpu().numpy(), fft.cpu().numpy(), fft_masked.cpu().numpy(),
                             [angles, intensity_theta], 'count = ' + str(ct), dx, save_fig='')

            j0 += step_size
            ct += 1
            col += 1

        i0 += step_size
        row += 1

    print('Processing time to get 4D datacube [seconds]: ' + str(time.time() - start_time))

    return datacube


def gaussian_q_filter(q, sigma_q, sigma_th, M, dx):
    """Generates gaussian filter in Fourier space for designated q center and bandwidth.
    Arguments:
        q: q vector in angstroms
        sigma_q: q bandwidth in angstroms
        sigma_th: theta bandwidth in degrees
        M: size of output fourier transform image
        dx: HRTEM image resolution in Angstroms/pixel
    Return:
        matrix: numpy array of size (NN, NN) with gaussian filter. Normalized by area (sum = 2).
    """
    f_pixels = get_q_pixels(q, M, dx)
    f_sigma_q_pixels = get_q_pixels(sigma_q, M, dx)
    f_sigma_th_pixels = get_q_pixels(sigma_th, M, dx)

    grid = np.linspace(-(M - 1) / 2, (M - 1) / 2, M)  # Array centering

    out = []  # temporary list to generate 2d output matrix

    for i in grid:
        a = 1 / (2 * np.pi * f_sigma_q_pixels * f_sigma_th_pixels)  # front in gaussian function
        # exponential term
        sub = ((grid - f_pixels) ** 2 / (2 * f_sigma_q_pixels ** 2) + (i) ** 2 / (2 * f_sigma_th_pixels ** 2))
        out.append(a * np.exp(-sub))

    matrix = np.array(out)
    matrix = matrix / np.sum(matrix)
    matrix = matrix + ndimage.rotate(matrix, 180, reshape=False)

    return matrix


def subplot_mini(image, fft_raw, fft_masked, I_vs_th, title='', dx=1.924, save_fig=''):
    """ Plots stack of figures to describe nano-FFT extraction process. From left to right,
    the following figures are plot: real space  nano-image, raw FFT, and filtered FFT.
    Args:
        image: real-space image
        fft_raw: calculated FFT
        fft_masked: FFT multiplied with bandpass filter
        title: optional title for first subplot
        dx: image resolution in Angstrom/pixel
        save_fig: filename in case user wants to save plot
    """
    q_max = np.pi / dx
    th, I = I_vs_th[0], I_vs_th[1]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.set_figheight(2)
    fig.set_figwidth(8)
    # Plot nano image
    ax1.imshow(image, cmap='gray', extent=[0, image.shape[1]*dx/10, 0, image.shape[0]*dx/10])
    ax1.set_xlabel('nm')
    ax1.set_ylabel('nm')
    ax1.set_title(title, fontsize=10)
    # plot TEM nanoimage with window
    ax2.imshow(fft_raw, cmap='gray', extent=[-q_max, q_max, -q_max, q_max])
    ax2.set_title('FFT', fontsize=10)
    ax2.set_xlabel('${Å^{-1}}$')
    # plot FFT of nanoimage
    ax3.imshow(fft_masked, cmap='gray', extent=[-q_max, q_max, -q_max, q_max])
    ax3.set_title('FFT masked', fontsize=10)
    ax3.set_xlabel('${Å^{-1}}$')
    # plot I vs theta
    ax4.scatter(th, I, s=2, color='blue')
    ax4.set_xlabel('θ / degrees')
    ax4.yaxis.tick_right()
    ax4.set_ylabel('Counts / a.u. ')
    ax4.set_xticks(np.arange(th[0], th[-1]+2, step=45))
    ax4.yaxis.set_label_position('right')

    plt.show()


def get_orientation_torch(fft, filters, device):
    """ Gets Intensity tensor for different angles at any grid point (x,y). Uses broadcasting and torch
    operations to speed 2x process with respect to loop.
    :param fft: GPU torch tensor of fourier transform
    :param filters: GPU torch tensor of gaussian filters to be applied (different angles)
    :param device: CUDA device
    :return: intensity tensor at grid point x,y
    """
    m, n = fft.shape
    fft_broadcast = torch.empty(m, n, 1).to(device).double()
    fft_broadcast[:, :, 0] = fft

    intensity_theta = torch.sum(torch.mul(fft_broadcast, filters), [0, 1])

    return intensity_theta


def get_q_pixels(q, M, dx):
    """Conversion of reciprocal space component in angstroms to spatial frequency in number of pixels.
    Arguments:
        q: reciprocal space vector (2pi/d)
        M: size of Fourier transform
        dx: HRTEM image resolution in Angstroms/pixel
    Returns:
        f_pixels: q in pixel size
    """
    f_angstrom = q / (2 * np.pi)   # q = 1/d (d in Angstrom)
    freq_resolution = 1 / (M * dx)
    f_pixels = f_angstrom / freq_resolution

    return f_pixels


def median_filter(data, size=1, device='cuda'):
    """
    Median filter operation for n-dimension tensor or array. Function first checks if data is pyTorch tensor or numpy,
    then performs median filter operation using numpy and returns same datatype as input.
    Can be relatively slow operation because only performed once per image/datacube.
    :param data: torch tensor or numpy array to apply median filter to
    :param device: CUDA device
    :param size: size of sliding window. Default is size = 1
    :return: Median filtered tensor or numpy array
    """
    if type(data) == torch.Tensor:
        median_np = ndimage.median_filter(data.cpu().numpy(), size=size)
        return torch.from_numpy(median_np).to(device)
    else:
        return ndimage.median_filter(data, size=size)


def raised_cosine_window_np(s, beta=0.2):
    """
    Generate 1D raised cosine window of length s and with roll-factor beta.
    :param s: pixel lenght of signal
    :param beta: roll-factor beta. Default=0.2 mainly zeroes down signal close to edges of image.
    :return:
         frequencies, window: frequency values and corresponding raised cosine signal
    """

    def rrc(f, b, T):
        """Finds raised cosine value for single frequency."""
        if f < (1.0 - b) / (2.0 * T):
            return 1.0
        elif f <= (1.0 + b) / (2.0 * T):
            return 0.5 * (1 + np.cos(np.pi * T / b * (f - (1 - b) / (2 * T))))
        else:
            return 0.

    T = 1 / s   # period
    frequencies = np.arange(-s / 2, s / 2, step=1)   # Distribute frequencies
    window = []
    for freq in frequencies:
        window.append(rrc(np.abs(freq), beta, T))   # Go over all frequencies

    window = window - np.amin(window)   # Set minimum to zero
    window = window / np.amax(window)   # Normalize so rc signal goes from 0 to 1

    return frequencies, window


def bandpass_filtering_image(img_gpu, q, q_bandwidth, dx, beta=0.1, device='cuda'):
    """
    Computes FT of image, multiplies by user-defined filter, and computes the inverse FT to get
    a filtered version of the image.
    :param img_gpu: image in Torch. Image can be any size and function allows for m != n
    :param q: spatial frequency center
    :param q_bandwidth: bandwidth of spatial frequency filter
    :param dx: pixel size in Angstrom/pixel
    :param beta: raised cosine beta coefficient
    :param device: default to 'cuda'
    :return: inverse fourier transform of image after applying raised cosine window and bandpass filter. Torch tensor.
    """
    # Pad image if shape[0] != shape[1]
    m, n = img_gpu.shape

    # Make raised cosine window
    _, rc_window_m = raised_cosine_window_np(m, beta=beta)
    _, rc_window_n = raised_cosine_window_np(n, beta=beta)
    window = torch.from_numpy(np.outer(rc_window_m, rc_window_n)).to(device)   # window shape is (m, n)

    # Multiply real space image by window
    img_gpu = img_gpu * window

    s = max(m, n)
    if m != n:
        print('padding tensor')
        pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
        img_gpu = pad(img_gpu)

    # Make bandpass filter
    bp_filter = torch.from_numpy(bandpass_filter(s, q - q_bandwidth, q + q_bandwidth, dx)).to(device)
    # Shift DC component to edges and reshape for broad casting with fft_gpu
    bp_filter = tensor_shift_fft(bp_filter).reshape(s, s, 1).double()

    plt.imshow(bp_filter[:, :, 0], cmap='gray')
    plt.title('Bandpass filter of {0}nm feature'.format(np.round(2 * np.pi / q / 10, 2)))
    plt.show()

    # Do FFT of img_gpu
    fft_gpu = torch.rfft(img_gpu, 2, normalized=False, onesided=False)

    # Do inverse FFT of FFT of image multiplied by bandpass filter
    ifft = torch.irfft(fft_gpu * bp_filter, 2, normalized=False, onesided=False)
    ifft = ifft + torch.abs(torch.min(ifft))  # Rescale values such that no negative intensity values

    return ifft
