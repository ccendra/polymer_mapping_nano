import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import auxiliary_functions as aux


class DirectorField:
    # Class developed by Luke Balhorn and edited by Camila Cendra
    def __init__(self, row, col, theta, intensity):
        # Theta must be given in degrees
        self.index = (row, col)
        self.start_point = (row + 0.5, col + 0.5)  # Stay in coordinates of matrix.
        self.x, self.y = self.start_point
        self.theta = theta
        self.intensity = intensity
        self.edges = self.find_first_edges()
        # To seed lines moving in two different directions
        self.lines = [[self.edges[0], self.start_point], [self.start_point, self.edges[1]]]

    def length(self):
        return len(self.lines[0]) + len(self.lines[1]) - 2

    def find_first_edges(self):
        # First check special cases
        if self.theta == 0 or self.theta == 180:
            return (np.floor(self.x), self.y), (np.ceil(self.x), self.y)
        elif self.theta == 90 or self.theta == 270:
            return (self.x, np.floor(self.y)), (self.x, np.ceil(self.y))
        else:
            # There are four possible intersections - one with each side of the box
            theta_rad = self.theta * np.pi / 180
            index = self.index
            center = self.start_point
            possible_points = [
                (index[0], center[1] + (index[0] - center[0]) * np.tan(theta_rad)),
                (index[0] + 1, center[1] + (index[0] + 1 - center[0]) * np.tan(theta_rad)),
                (center[0] + (index[1] - center[1]) / np.tan(theta_rad), index[1]),
                (center[0] + (index[1] + 1 - center[1]) / np.tan(theta_rad), index[1] + 1)
            ]
            # Due to symmetry, the nearest two points are the edge of the box.  45 degree angle crosses corner, which
            # results in a tie for closest which is fine as long as we get two different points.
            nearest_points = sorted(possible_points, key=lambda x: distance_2d(x, center))
            if self.theta == 45 or self.theta == 135:  # Remove Identical Points as needed
                while distance_2d(nearest_points[0], nearest_points[1]) < distance_2d(nearest_points[0], center):
                    nearest_points.remove(nearest_points[1])
                return nearest_points[:2]
            else:
                return nearest_points[:2]


def distance_2d(p1, p2):
    """Returns distance between two 2D vectors."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def seed_lines(peaks_matrix, angles):
    # Set up variables
    n_rows, n_cols, n_angles = peaks_matrix.shape
    line_seeds = []

    # Generate peaks matrix
    for x in range(n_rows):
        for y in range(n_cols):
            for i in range(n_angles):
                if peaks_matrix[x, y, i] == 1:
                    # Have to flip coordinates since rows are 'y' and columns are 'x'. Similarly,
                    # given that we want to plot vs. real coordinates, need to rotate image.
                    # We can do this by flipping coordinates as done below.
                    new = DirectorField(y, n_rows - 1 - x, angles[i], 1)  # Angle in Degrees
                    line_seeds.append(new)

    return line_seeds


def plot_director_field(peaks_matrix, angles, x_length_nm, y_length_nm, perpendicular=True,
                        step_nm=10, size=8, linewidth=0.5, colored_lines=False, save_fig='', show_plot=False):

    if perpendicular:
        angles = angles + 90

    seeds_list = seed_lines(peaks_matrix, angles)

    lines_to_print = []
    colors_to_print = []
    for seed in seeds_list:
        for line in seed.lines:
            lines_to_print.append(line)
            colors_to_print.append(aux.color_by_angle(seed.theta))

    x_ticks_nm = np.arange(0, x_length_nm, step=step_nm)
    x_ticks_px = x_ticks_nm / (x_length_nm / peaks_matrix.shape[1])

    y_ticks_nm = np.arange(0, y_length_nm, step=step_nm)
    y_ticks_px = y_ticks_nm / (y_length_nm / peaks_matrix.shape[1])

    fig, ax = plt.subplots(figsize=(size, size))
    if colored_lines:
        line_plot = mc.LineCollection(lines_to_print, linewidth=linewidth, colors=colors_to_print)
    else:
        line_plot = mc.LineCollection(lines_to_print, linewidth=linewidth, colors='black')

    ax.add_collection(line_plot)
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.xticks(x_ticks_px, x_ticks_nm.astype(int))
    plt.yticks(y_ticks_px, y_ticks_nm.astype(int))
    plt.xlabel('distance / nm')
    plt.ylabel('distance / nm')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    plt.savefig(save_fig + '.png', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)
