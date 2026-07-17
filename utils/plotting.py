import json

from matplotlib import pyplot as plt
import numpy as np


def plot_images_from_json(json_path, cmap="gray"):
    """Loads a 3D array from a JSON file and plots the 2D slices side by side.

    Parameters:
    - json_path (str): File path to the JSON file.
    - key_name (str): The dictionary key inside the JSON where the array is stored.
    - cmap (str): Matplotlib colormap (default is 'gray' for 2D image slices).
    """
    # 1. Open and parse the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. Extract the list and convert it into a 3D NumPy array
    # Expected shape: (num_images, height, width)
    matrix_3d = np.array(data, dtype=np.uint8)

    if matrix_3d.ndim != 3:
        raise ValueError(
            f"Expected a 3D array, but got an array with {matrix_3d.ndim} dimensions."
        )

    num_images, height, width = matrix_3d.shape
    print(
        f"Loaded {num_images} images. Each image dimension: {height}x{width}"
    )

    # 3. Create a side-by-side subplot layout dynamically
    # figsize scales width based on how many images you have
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    # If there's only 1 image, matplotlib doesn't return an array of axes,
    # so we wrap it in a list to make it iterable.
    if num_images == 1:
        axes = [axes]

    # 4. Loop through each 2D slice and plot it
    for i in range(num_images):
        axes[i].imshow(matrix_3d[i], cmap=cmap)
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")  # Hide the pixel grid coordinates for a cleaner look

    plt.tight_layout()  # Adjust spacing cleanly
    plt.show()