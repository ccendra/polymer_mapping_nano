from skimage import io
import numpy as np
import os
import mrcfile
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def read_tif(fn, gray=True):
    """Opens raw .TIF image and returns numpy array.
    Args:
        fn: image filename
    Returns:
        np array of size (n, x, y)
        n is number of stacked images (generally 24)
        x, y is # pixels horizontally and vertically
    """
    img = io.imread(fn, as_gray=gray)

    return img.astype('float64')


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


def stack_image(img_raw):
    """Returns sum of n stacked images
    Args:
        img_raw: np array of (n, x, y) OR (x,y) size
    Returns:
        np array size (x,y)
    """
    den = len(img_raw.shape)
    if den > 2:
        return np.sum(img_raw, axis=0)
    return img_raw


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


def make_color_wheel(size=5, save_fig=''):
    theta_values = np.arange(360, step=0.1)
    # Generate  listed colormap
    # values already normalized ([0,1])
    cmap = colors.ListedColormap(get_colors(theta_values))

    # Function found in Stack overflow:
    # https://stackoverflow.com/questions/31940285/plot-a-polar-color-wheel-based-on-a-colormap-using-python-matplotlib
    # Generate a figure with a polar projection
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')

    # Plot a color mesh on the polar plot
    # with the color set by the angle

    n = 200  # the number of secants for the mesh
    t = np.linspace(0, 2 * np.pi, n)  # theta values
    r = np.linspace(.6, 1, 2)  # radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
    c = tg  # define color values as theta value

    im = ax.pcolormesh(t, r, c.T, cmap=cmap)  # plot the colormesh on axis with colormap
    ax.set_yticklabels([])  # turn of radial tick labels (yticks)
    ax.tick_params(pad=15, labelsize=24)  # cosmetic changes to tick labels
    ax.spines['polar'].set_visible(False)  # turn off the axis spine.

    if save_fig:
        plt.savefig(save_fig + '.png', dpi=600)
    plt.show()


def assemble_filename(image_number, base_name, ending='.mrc'):
    return base_name + str(image_number).zfill(5) + ending


def create_directory(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            print("Creation of the directory %s failed." % path)
    else:
        print("Successfully created the directory %s." % path)


def filename_folder_initialization(image_number, base_name, snippet='', ending='mrc'):
    # Create directory for output of process
    dir_name = (base_name[:-1] + '/' + str(image_number).zfill(5)) + '/' + snippet
    create_directory(dir_name)

    # Keep track of output folder name for future work
    output_folder = dir_name + '/'

    # Assemble filename of raw HRTEM image
    hrtem_filename = assemble_filename(image_number, base_name, ending)

    return hrtem_filename, output_folder