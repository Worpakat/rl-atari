from dataclasses import dataclass
from collections import deque

import numpy as np
import torch

from utils.data_buffers import BaseBuffer


@dataclass(slots=True)
class Transition:
    """
    One environment transition collected during trajectory generation.
    """
    state: np.ndarray
    action: int
    reward: float
    representation: torch.Tensor | None = None
    is_exploration_action: bool = False


class TransitionQueue(BaseBuffer):

    def __getitem__(self, index: int) -> Transition:
        return self._memory[index]

    def is_full(self) -> bool:
        return len(self) == self.capacity

    def append(
        self,
        transition: Transition,
    ) -> None:

        self._memory.append(transition)


class TransitionQueueManager:
    """
    Manages multiple transition queues.

    Each TransitionQueue represents one uninterrupted trajectory
    (e.g. between two deaths).

    The manager keeps track of the total number of stored transitions
    across all queues so that optimization can be triggered efficiently.
    """

    def __init__(self, capacity: int) -> None:

        self.capacity = capacity

        self._queues: deque[TransitionQueue] = deque()
        self._current_queue = TransitionQueue(self.capacity)

        self._total_size = 0

    @property
    def total_size(self) -> int:
        """
        Total number of stored transitions.
        """
        return self._total_size

    def is_full(self) -> bool:
        """
        Returns whether the manager reached its total capacity.
        """
        return self._total_size >= self.capacity

    def append(self, transition: Transition) -> None:
        """
        Appends a transition to the current trajectory.
        """

        self._current_queue.append(transition)
        self._total_size += 1

    def end_trajectory(self) -> None:
        """
        Finishes the current trajectory and starts a new one.

        Empty trajectories are ignored.
        """

        if len(self._current_queue) == 0:
            return

        self._queues.append(self._current_queue)

        self._current_queue = TransitionQueue(self.capacity)

    def trajectories(self) -> list[TransitionQueue]:
        """
        Returns all completed trajectories.

        If the current trajectory is non-empty, it is also included.
        """

        trajectories = list(self._queues)

        if len(self._current_queue) > 0:
            trajectories.append(self._current_queue)

        return trajectories
    
    def get_last_transition(self) -> Transition:
        return self._current_queue[-1]

    def clear(self) -> None:
        """
        Removes all stored trajectories.
        """
        self._queues.clear()
        self._current_queue.clear()
        self._total_size = 0

    def __len__(self) -> int:
        return self._total_size

    def __iter__(self):
        """
        Iterates over all trajectories.
        """

        yield from self.trajectories()


class TransitionDelayBuffer(BaseBuffer):
    """
    Delays transitions by a fixed number of steps.

    New transitions are appended immediately but are not released until
    they become older than the configured delay.

    This allows delayed environment signals (e.g. life loss) to modify
    recent transitions before they are committed to replay memory or
    trajectory buffers.
    """

    def __init__(self, delay: int):
        self.delay = delay
        self._buffer = deque()

    def append(self, transition: Transition) -> Transition | None:
        """
        Appends a transition.

        Returns
        -------
        Transition | None

            The oldest transition if it is ready to be released.
            Otherwise None.
        """

        self._buffer.append(transition)

        if len(self._buffer) > self.delay:
            return self._buffer.popleft()

        return None

    def pop_oldest(self) -> Transition:
        return self._buffer.popleft()


    def pop_all(self) -> list[Transition]:
        """
        Returns all remaining delayed transitions.
        """
        remaining = list(self._buffer)
        
        self._buffer.clear()

        return remaining
    
    def discard_newest(self, count: int) -> None:
        """
        Discards the newest `count` transitions.
        """
        count = min(count, len(self._buffer))

        for _ in range(count):
            self._buffer.pop()


    def discard_all(self):
        """This and `clear` are equivalent. 
        Both exist due to naming convention."""
        self._buffer.clear()

    def clear(self):
        self._buffer.clear()
    

    def __len__(self):
        return len(self._buffer)