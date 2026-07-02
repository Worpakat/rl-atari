from collections import deque
import random

import numpy as np
import torch


class FrameSequenceBuffer:
    """
    Fixed-length buffer for constructing consecutive frame sequences.

    This class is intentionally a thin wrapper around ``collections.deque``.
    Although its current functionality is simple, it provides a meaningful
    abstraction throughout the project and allows future extensions without
    changing the training code.
    """

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self._buffer = deque(maxlen=sequence_length)

    def append(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)

    def clear(self) -> None:
        self._buffer.clear()

    def is_ready(self) -> bool:
        return len(self._buffer) == self.sequence_length

    def get_sequence(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Numpy array of shape (sequence_length, C, H, W).
        """
        if not self.is_ready():
            raise RuntimeError("Frame sequence is not yet complete.")

        return np.stack(tuple(self._buffer), axis=0)


class SequenceReplayBuffer:
    """
    Replay buffer storing frame sequences.

    The current implementation stores only frame sequences for Phase 0 DSAE
    training. It is intentionally designed to be extended later with actions,
    rewards, next states and terminal flags for RL training.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._memory)

    def append(self, frame_sequence: np.ndarray) -> None:
        """
        Stores one frame sequence.

        Parameters
        ----------
        frame_sequence:
            Numpy array of shape (sequence_length, C, H, W).
        """
        # Prevent later modifications by storing an independent tensor.
        self._memory.append(frame_sequence.copy())

    def can_sample(self, batch_size: int) -> bool:
        return len(self) >= batch_size

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Randomly samples a mini-batch.

        Returns
        -------
        dict
            {
                "frames": Tensor of shape
                          (batch_size, sequence_length, C, H, W)
            }
        """
        batch = random.sample(self._memory, batch_size)

        return {
            "frames": torch.from_numpy(np.stack(batch, axis=0))
        }

    def clear(self) -> None:
        self._memory.clear()

    def state_dict(self):
        """
        Returns the replay buffer state.
        """

        return {
            "buffer": list(self._memory),
        }


    def load_state_dict(self, state_dict):
        """
        Restores the replay buffer state.
        """

        self.buffer.clear()
        self.buffer.extend(state_dict["buffer"])