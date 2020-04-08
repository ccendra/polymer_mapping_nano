import reduce_data as reduce
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def fft(ft, M, q_contour_list=[], size=10, dx=1.924, color='blue', save_fig='', show=False, alpha=0.5):
    """Plots Fourier transform and optionally radial contours of q space.
    Arguments:
        ft: fourier transform numpy array. If using pyTorch tensor, must be send to cpu and converted to numpy.
        M: size of FFT
        q_contour_list: list of q values to be drawn as contours in figure
        color: color of contours
        alpha: transparency. Default=0.5
    """

    fig, ax = plt.subplots(figsize=(size, size))

    if len(q_contour_list) > 0:
        ax.imshow(ft, cmap='gray', vmax=np.percentile(ft, 99))
        for q in q_contour_list:
            f_pixels = reduce.get_q_pixels(q, M, dx)
            ax.add_patch(plt.Circle(((M-1) / 2, (M-1) / 2), f_pixels, facecolor='none',
                                    edgecolor=color, alpha=alpha, linewidth=1, linestyle=':'))
            ax.annotate(str(np.round(q, 2)), xy=(M/2, M/2 + f_pixels), color=color, alpha=alpha, fontsize=12)

        ax.plot()  # Causes an auto scale update.

    else:
        q_max = np.pi / dx
        ax.imshow(ft, cmap='gray', extent=[-q_max, q_max, -q_max, q_max], vmax=np.percentile(ft, 99))
        ax.set_xlabel('q / ${Å^{-1}}$', fontsize=18)
        ax.set_ylabel('q / ${Å^{-1}}$', fontsize=18)

    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300)
    if show:
        plt.show()


def hrtem(img, size=15, gamma=1, vmax=0, colorbar=False, dx=1.924, save_fig='', show=False):
    """Plots 2D TEM image in gray scale.
    Arguments:
        img: 2D numpy array
        size: output figure size
        gamma: image contrast enhancer. Gamma = 1 as default (i.e no enhancement)
        vmax:
        colorbar:
        dx:
        save_fig:
    """
    plt.figure(figsize=(size, size))

    gamma_corrected = exposure.adjust_gamma(img, gamma)
    x_size = img.shape[1] * dx/10
    y_size = img.shape[0] * dx/10

    if vmax == 0:
        # Use 99th percentile
        vmax = np.percentile(gamma_corrected, 99)

    plt.imshow(gamma_corrected, extent=[0, x_size, 0, y_size], cmap='gray', vmax=vmax)
    plt.xlabel('distance / nm', fontsize=18)
    plt.ylabel('distance / nm', fontsize=18)

    if colorbar:
        plt.colorbar()
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=600)
    if show:
        plt.show()


def color_by_angle(theta):
    """Tuple of RGB values to generate circular palette.
    Arguments:
        theta: theta angle
    Returns:
        (R, G, B): tuple of doubles with R, G, B coordinates based on input angle
    """
    radians = theta * 3.14 / 180
    red = np.cos(radians) ** 2  # Range is max 1 not max 255
    green = np.cos(radians + 3.14 / 3) ** 2 * 0.7  # Makes green darker - helps it stand out equally to r and b
    blue = np.cos(radians + 2 * 3.14 / 3) ** 2
    return (red, green, blue)


def get_colors(angles):
    """Generates list of (R, G, B) colors for a list of angles (i.e. angular resolution)
    Arguments:
        angles: list or numpy array with angular values used for datacube analysis
    Returns:
        output: list of (R,G,B) tuples for every single angle used during analysis.
    """
    output = []
    for angle in angles:
        output.append(color_by_angle(angle))
    return output


def plot_orientation_map(orientation_map, angles, size=10, title='', save_fig='', show=True):
    """Plots orientation map using a circular color palette to describe angular orientation.
    Arguments:
        """
    cmap = colors.ListedColormap(get_colors(angles + 90))

    plt.figure(figsize=(size, size))
    plt.imshow(orientation_map, cmap=cmap)
    # plt.colorbar()
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(save_fig + '.png', dpi=300)
    if show:
        plt.show()


def intensity_q_lineout(x, y, q_range=[0, 1.5], save_fig='', show=True):
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, s=0.5, color='blue')
    plt.plot(x, y, linewidth=0.2, color='black')
    plt.yscale('log')
    plt.xlim([0, 1.6])
    plt.xlabel('q / Å$^{-1}$', fontsize=14)
    plt.ylabel('Counts / a.u.', fontsize=14)
    plt.xlim(q_range)
    plt.autoscale(axis='y')
    if save_fig:
        plt.savefig(save_fig + '.png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()


