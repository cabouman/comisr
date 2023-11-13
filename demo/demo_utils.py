import numpy as np
import matplotlib.pyplot as plt


def imshow_hi_res(array, title='', vmin=None, vmax=None, cmap='viridis', show=False):
    """ Display an array using a figure size large enough to map one array element to one pixel
    The scaling isn't exactly correct, so some pixels are slightly larger than others, partially depending on the
    labels on the axes and on the colorbar.  It's difficult to get this exactly correct.
    Args:
        array: The array to display
        title: The title for the plot
        vmin: Minimum of the intensity window - same as vmin in imshow
        vmax: Maximum of the intensity window - same as vmax in imshow
        cmap: The color map as in imshow - same as cmap in imshow
        show: If true, then plt.show() is called to display immediately, otherwise call
              fig.show() on the object returned from this function to show the plot.

    Returns:
        The pyplot figure object
    """
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    # Scale for one pixel per element and account for the border
    fig_height = 3.7 * array.shape[0] * px
    fig_width = 4.6 * array.shape[1] * px
    fontsize = max([fig_height, 6.0])
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    plt.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()
    return fig


def have_same_shape(array1, array2):
    """
    Check if two arrays have the same shape - return True if so, false if not.
    """
    return np.allclose(np.array(array1.shape), np.array(array2.shape))
