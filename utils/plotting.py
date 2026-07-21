import json
import math

from matplotlib import pyplot as plt
import numpy as np


def plot_3d_array_grid(
    array_3d, images_per_row=4, cmap="gray", title_prefix="Frame"
):
    """Plots a 3D NumPy array of images into a multi-row grid layout.

    Parameters:
    - array_3d (np.ndarray): 3D array of shape (num_images, height, width).
    - images_per_row (int): Number of images to display per row (default is 4).
    - cmap (str): Matplotlib colormap.
    - title_prefix (str): Label used above each individual image.
    """
    if not isinstance(array_3d, np.ndarray) or array_3d.ndim != 3:
        raise ValueError(
            "Input must be a 3D NumPy array with shape (num_images, height, width)."
        )

    num_images, height, width = array_3d.shape

    # Calculate the required number of rows dynamically
    num_rows = math.ceil(num_images / images_per_row)

    # Create the figure with a size scaled to the grid dimensions
    fig, axes = plt.subplots(
        num_rows,
        images_per_row,
        figsize=(images_per_row * 3, num_rows * 3),
        squeeze=False,  # Ensures axes is always a 2D array even for 1 row
    )

    # Loop through all grid slots
    for i in range(num_rows * images_per_row):
        row = i // images_per_row
        col = i % images_per_row
        ax = axes[row, col]

        if i < num_images:
            # Display the frame image
            ax.imshow(array_3d[i], cmap=cmap)
            ax.set_title(f"{title_prefix} {i}")
            ax.axis("off")
        else:
            # Hide empty axes if the last row isn't fully filled (e.g., 6 images total)
            ax.axis("off")

    plt.tight_layout()
    plt.show()