from pathlib import Path

import numpy as np
import torch


def preprocess_frame(
    frame: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess a raw Atari frame.

    Steps:
        1. Remove irrelevant image regions.
        2. Convert from HWC to CHW.
        3. Convert to float32.
        4. Optionally normalize to [0, 1].

    Args:
        frame:
            Raw Gymnasium observation with shape (H, W, C).

        normalize:
            Whether to scale pixel values into [0, 1].

    Returns:
        Tensor with shape (C, H, W).
    """

    frame = np.concatenate((frame[:165], frame[175:190]), axis=0)
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame.astype(np.float32)

    if normalize:
        frame /= 255.0

    return frame


def ensure_directory(path: str | Path) -> Path:
    """
    Creates a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Path object pointing to the directory.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path