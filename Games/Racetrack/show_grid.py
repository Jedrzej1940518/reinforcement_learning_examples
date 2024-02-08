import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import exported_grid

def show_grid(grid):
    cmap = ListedColormap(exported_grid.colors)
    image = grid

    fig, ax = plt.subplots()
    mat = ax.matshow(image, cmap=cmap, vmin=0, vmax=len(exported_grid.colors))
    ax.set_xticks(range(exported_grid.ncols))
    ax.set_yticks(range(exported_grid.nrows))
    ax.set_xticks(np.arange(-.5, exported_grid.ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, exported_grid.nrows, 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='both', axis='both', length=0) # don't show tick marks

    plt.show()
