import numpy as np
import torch
import mrcfile
import matplotlib.pyplot as plt


def track_drift(data_tensor, img_size, verbose=False):
    n_frames = data_tensor.shape[0]

    x_center, y_center = img_size // 2, img_size // 2

    x_drift_list = [0]
    y_drift_list = [0]

    img_prev = data_tensor[0]

    for i in range(1, n_frames):
        img_curr = data_tensor[i]

        xc_results = calculate_xc(img_prev, img_curr, img_size)
        a, b = np.where(xc_results == np.amax(xc_results))
        if verbose:
            print('Drift b/w image #' + str(i-1) + ' and #' + str(i) + ' :', x_center - a, y_center - b)
        x_drift_list.append(x_center - a[0])
        y_drift_list.append(y_center - b[0])

    return np.array(x_drift_list), np.array(y_drift_list)


def calculate_xc(tensor1, tensor2, img_size, device='cuda'):
    fft_size = img_size
    fft_t1 = complex_fft(tensor1, fft_size, normalize=True)
    fft_t2 = complex_fft(tensor2, fft_size, normalize=True)

    prod = fft_t1 * np.conjugate(fft_t2)

    out = np.zeros((fft_size, fft_size, 2))
    out[:, :, 0] = np.real(prod)
    out[:, :, 1] = np.imag(prod)
    out_tensor = torch.from_numpy(out).to(device)

    xc = torch.irfft(out_tensor, signal_ndim=2, normalized=False, onesided=False)
    xc = tensor_shift_fft(xc).cpu().numpy()

    return xc


def correct_data(data, x_drift, y_drift, max_drift_allowed, img_size, save_array=''):
    padding = max_drift_allowed * 2
    n_frames = data.shape[0]
    data_corrected = np.zeros((n_frames, img_size + 2 * padding, img_size + 2 * padding), dtype=np.uint16)
    ct = 0
    for i in range(n_frames):
        ux = x_drift[i]
        uy = y_drift[i]
        if np.abs(ux) <= max_drift_allowed and np.abs(uy) <= max_drift_allowed:
            data_corrected[ct, (padding - ux):-(padding + ux), (padding - uy):-(padding + uy)] = data[i, :img_size, :img_size]
            ct += 1

    size = padding + max_drift_allowed
    data_corrected = data_corrected[:ct, size:-size, size:-size]
    print('Data has been corrected and has shape: ', data_corrected.shape)

    if save_array:
        np.save(save_array + '.npy', data_corrected)
        print('Corrected image array has been saved.')
    return data_corrected


def plot_2d_drift(x_drift, y_drift, dx=1.924, lines=False, save_fig=''):
    n_frames = len(x_drift)
    plt.figure(figsize=(5, 4))
    plt.scatter(x_drift * dx, y_drift * dx, s=20, c=np.arange(n_frames))
    range = [np.max(np.abs(x_drift)), np.max(np.abs(y_drift))]
    range_max = max(range) * dx * 1.3
    if lines:
        plt.plot(x_drift * dx, y_drift * dx, color='black', linewidth=0.25)
    plt.xlim([-range_max, range_max])
    plt.ylim([-range_max, range_max])
    plt.colorbar(label='frame #')
    plt.xlabel('Image drift in x̄ / Å', fontsize=14)
    plt.ylabel('Image drift in ȳ / Å', fontsize=14)
    if save_fig:
        plt.savefig(save_fig + '_2d_drift.png', dpi=300, bbox_inches='tight')
    plt.show()


def complex_fft(tensor, s=5000, normalize=False):
    """Returns powder spectra of 2D tensor (image) using PyTorch implementation.
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
    m, n = tensor.shape
    pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)

    if normalize:
        tensor = tensor / torch.mean(tensor) - 1

    tensor = tensor * hanning(m)

    fft_tensor = torch.rfft(pad(tensor).float(), signal_ndim=2, normalized=True, onesided=False)
    fft_real = fft_tensor[:, :, 0].cpu().numpy()
    fft_complex = fft_tensor[:, :, 1].cpu().numpy()

    fft = fft_real + 1j * fft_complex

    return fft


def tensor_shift_fft(fft):
    """Shift zero frequency spatial frequency component to center of 2D image. For Pytorch implementation
    Args:
        fft: 2D FFT obtained using torch_fft function
    Returns:
        shifted FFT with DC frequency component in center of image.
    """
    m, n = fft.shape
    out = torch.cat((fft[-m // 2:], fft[:-m // 2]), dim=0)
    return torch.cat((out[:, -n // 2:], out[:, :-n // 2]), dim=1)


def hanning(N, device='cuda'):
    hanning_window = torch.from_numpy(np.outer(np.hanning(N), np.hanning(N))).to(device)

    return hanning_window


def read_mrc(fn):
    """Opens .mrc file containing single stack of images and returns numpy array.
    Args:
        fn: image filename
    Returns:
        np array of size (n, x, y)
        n is number of stacked images (generally 24)
        x, y is # pixels horizontally and vertically
    """
    mrc = mrcfile.open(fn, mode='r')
    img = np.flip(mrc.data, axis=1)
    mrc.close()

    return img.astype('float64')


def normalize_img(img_tensor):
    mean_tensor = torch.mean(img_tensor)

    return img_tensor / mean_tensor - 1


# import numpy as np
# import torch
# import mrcfile
# import matplotlib.pyplot as plt
#
#
# def track_drift(data_tensor, verbose=False):
#     n_frames, m, n = data_tensor.shape
#     img_size = max(m, n)
#     x_center, y_center = m // 2, n // 2
#
#     x_drift_list = [0]
#     y_drift_list = [0]
#
#     img_prev = data_tensor[0]
#
#     for i in range(1, n_frames):
#         img_curr = data_tensor[i]
#
#         xc_results = calculate_xc(img_prev, img_curr, img_size)
#         a, b = np.where(xc_results == np.amax(xc_results))
#         if verbose:
#             print('Drift b/w image #' + str(i-1) + ' and #' + str(i) + ' :', x_center - a, y_center - b)
#         x_drift_list.append(x_center - a[0])
#         y_drift_list.append(y_center - b[0])
#
#     return np.array(x_drift_list), np.array(y_drift_list)
#
#
# def calculate_xc(tensor1, tensor2, img_size, device='cuda'):
#     fft_size = img_size
#     fft_t1 = complex_fft(tensor1, fft_size, normalize=True)
#     fft_t2 = complex_fft(tensor2, fft_size, normalize=True)
#
#     prod = fft_t1 * np.conjugate(fft_t2)
#
#     out = np.zeros((fft_size, fft_size, 2))
#     out[:, :, 0] = np.real(prod)
#     out[:, :, 1] = np.imag(prod)
#     out_tensor = torch.from_numpy(out).to(device)
#
#     xc = torch.irfft(out_tensor, signal_ndim=2, normalized=False, onesided=False)
#     xc = tensor_shift_fft(xc).cpu().numpy()
#
#     return xc
#
#
# def correct_data(data, x_drift, y_drift, max_drift_allowed, device='cuda'):
#     padding = max_drift_allowed * 2
#     n_frames, m, n = data.shape
#     data_corrected = torch.zeros((n_frames, m + 2 * padding, n + 2 * padding)).to(device)
#     ct = 0
#     for i in range(n_frames):
#         ux = x_drift[i]
#         uy = y_drift[i]
#         if np.abs(ux) <= max_drift_allowed and np.abs(uy) <= max_drift_allowed:
#             data_corrected[ct, (padding - ux):-(padding + ux), (padding - uy):-(padding + uy)] = data[i, :m, :n]
#             ct += 1
#
#     size = padding + max_drift_allowed
#     data_corrected = data_corrected[:ct, size:-size, size:-size]
#     print('Data has been corrected and has shape: ', data_corrected.shape)
#
#     return data_corrected
#
#
# def plot_2d_drift(x_drift, y_drift, dx=1.924, lines=False, save_fig=''):
#     n_frames = len(x_drift)
#     plt.figure(figsize=(5, 4))
#     plt.scatter(x_drift * dx, y_drift * dx, s=20, c=np.arange(n_frames))
#     range = [np.max(np.abs(x_drift)), np.max(np.abs(y_drift))]
#     range_max = max(range) * dx * 1.3
#     if lines:
#         plt.plot(x_drift * dx, y_drift * dx, color='black', linewidth=0.25)
#     plt.xlim([-range_max, range_max])
#     plt.ylim([-range_max, range_max])
#     plt.colorbar(label='frame #')
#     plt.xlabel('Image drift in x̄ / Å', fontsize=14)
#     plt.ylabel('Image drift in ȳ / Å', fontsize=14)
#     if save_fig:
#         plt.savefig(save_fig + '_2d_drift.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# def complex_fft(tensor, s=5000, normalize=False):
#     """Returns powder spectra of 2D tensor (image) using PyTorch implementation.
#     NOTE: location of operation (GPU or CPU) is determined by location of input tensor.
#     Send tensor to GPU prior to using this function to perform operations in GPU (i.e. tensor.to(cuda))
#     Args:
#         tensor: 2D tensor (image)
#         s: output size of FFT (s x s). tensor is padded with zeros prior to performing FFT operation
#         to specified output size.
#     Returns:
#         fft: powder spectra (real^2 + complex^2) tensor of size (s x s) with Fourier Transform.
#              DC frequency component is set in center of tensor.
#     """
#     m, n = tensor.shape
#     pad = torch.nn.ConstantPad2d(padding=(0, s - n, 0, s - m), value=0)
#
#     if normalize:
#         tensor = tensor / torch.mean(tensor) - 1
#
#     tensor = tensor * hanning(m)
#
#     fft_tensor = torch.rfft(pad(tensor).float(), signal_ndim=2, normalized=True, onesided=False)
#     fft_real = fft_tensor[:, :, 0].cpu().numpy()
#     fft_complex = fft_tensor[:, :, 1].cpu().numpy()
#
#     fft = fft_real + 1j * fft_complex
#
#     return fft
#
#
# def tensor_shift_fft(fft):
#     """Shift zero frequency spatial frequency component to center of 2D image. For Pytorch implementation
#     Args:
#         fft: 2D FFT obtained using torch_fft function
#     Returns:
#         shifted FFT with DC frequency component in center of image.
#     """
#     m, n = fft.shape
#     out = torch.cat((fft[-m // 2:], fft[:-m // 2]), dim=0)
#     return torch.cat((out[:, -n // 2:], out[:, :-n // 2]), dim=1)
#
#
# def hanning(N, device='cuda'):
#     hanning_window = torch.from_numpy(np.outer(np.hanning(N), np.hanning(N))).to(device)
#
#     return hanning_window
#
#
# def read_mrc(fn):
#     """Opens .mrc file containing single stack of images and returns numpy array.
#     Args:
#         fn: image filename
#     Returns:
#         np array of size (n, x, y)
#         n is number of stacked images (generally 24)
#         x, y is # pixels horizontally and vertically
#     """
#     mrc = mrcfile.open(fn, mode='r')
#     img = np.flip(mrc.data, axis=1)
#     mrc.close()
#
#     return img.astype('float64')
#
#
# def normalize_img(img_tensor):
#     mean_tensor = torch.mean(img_tensor)
#
#     return img_tensor / mean_tensor - 1