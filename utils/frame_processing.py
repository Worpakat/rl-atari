import numpy as np


def cut_and_transpose_frame(frame: np.ndarray) -> np.ndarray:
    """
    Remove irrelevant image regions.

    Args:
        frame:
            Raw Gymnasium observation with shape (H, W) or (H, W, C).

    Returns:
        Numpy array with shape (H, W) or (C, H, W).
    """
    frame = np.concatenate((frame[:161], frame[175:190]), axis=0)

    if frame.ndim == 3: # Transpose from HWC to CHW if frame is RGB.
        frame = np.transpose(frame, (2, 0, 1))
    
    return frame


def convert_and_norm_sequence(
    frames: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess grayscale Atari frame sequences.

    Supports:
        (T, H, W)
        (B, T, H, W)

    Steps:
        1. Remove irrelevant image regions.
        2. Convert to float32.
        3. Optionally normalize to [0, 1].

    Args:
        frames:
            Frame sequence of shape (T, H, W) or
            batch of sequences (B, T, H, W).

        normalize:
            Whether to scale pixel values into [0, 1].

    Returns:
        Preprocessed array with the same leading dimensions.
    """

    frames = frames.astype(np.float32)

    if normalize:
        frames /= 255.0

    return frames




def preprocess_frame(
    frame: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess a raw Atari frame.
    
    !! OBSOLETED !!

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

    frame = np.concatenate((frame[:161], frame[175:190]), axis=0)
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame.astype(np.float32)

    if normalize:
        frame /= 255.0

    return frame

