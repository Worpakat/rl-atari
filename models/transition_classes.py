from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from collections import deque
import random

import numpy as np
import torch



# ---------------------------------------------------------

# !! WE HAVE TO KEEP THIS CLASS IN THIS FILE TO PREVENT CYCLIC IMPORTS.
class BaseBuffer(ABC):

    def __init__(
        self,
        capacity: int,
    ):
        self.capacity = capacity
        self._memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._memory)

    def can_sample(
        self,
        batch_size: int,
    ) -> bool:
        return len(self) >= batch_size

    def sample(
        self,
        batch_size: int,
    ):
        return random.sample(self._memory, batch_size)

    def clear(self) -> None:
        self._memory.clear()

    def state_dict(self) -> dict:
        return {
            "memory": list(self._memory),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._memory = deque(
            state_dict["memory"],
            maxlen=self._memory.maxlen,
        )

    @abstractmethod
    def append(self, *args, **kwargs):
        pass


# ---------------------------------------------------------

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
    death_transition: bool = False


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
    
    def get_last(self) -> Transition:
        return self._memory[-1]
    
    def remove_first(self, count: int) -> None:
        """
        Removes the first `count` transitions.
        """
        count = min(count, len(self._memory))
        for _ in range(count):
            self._memory.popleft()


    def remove_last(self, count: int) -> None:
        """
        Removes the last `count` transitions.
        """
        count = min(count, len(self._memory))
        for _ in range(count):
            self._memory.pop()


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
    
    def get_current_trajectory(self) -> TransitionQueue:
        return self._current_queue

    def get_last_transition(self) -> Transition:
        return self._current_queue.get_last()

    def clear(self) -> None:
        """
        Removes all stored trajectories.
        """
        self._queues.clear()
        self._current_queue.clear()
        self._total_size = 0

    def __len__(self) -> int:
        actual_length = 0

        for queue in self._queues:
            actual_length += len(queue)

        return actual_length
    

    def __iter__(self):
        """
        Iterates over all trajectories.
        """

        yield from self.trajectories()


class TrajectoryType(Enum):
    FIRST = "first"
    INTERMEDIATE = "intermediate"
    LAST = "last"


class RiverRaidStaticSequenceHandler:
    """
    Removes non-interactive static animation transitions from River Raid
    trajectories.

    Parameters
    ----------
    initial_static_frames:
        Number of static transitions at the beginning of the episode.

    death_static_frames:
        Number of static transitions after each death.

    terminal_static_frames:
        Number of static transitions at the end of the episode.
    """

    def __init__(
        self,
        sequence_length: int,
        initial_static_frames: int,
        intermediate_static_frames: int,
        terminal_static_frames: int,
    ):
        # 'tbr': to be removed
        self.initial_transitions_tbr = initial_static_frames - (sequence_length - 1)
        # First valid sequence is When sequence has the first non-static frame; like "|st|st|st|nst|".
        # Thus we need to remove (sequence_length - 1) transitions. This is not a whole explanation of removing logic.
        # Look: ... To Be Added ...

        self.death_transitions_tbr = intermediate_static_frames - 1
        # Look for explanation: ... To Be Added ...

        self.terminal_transitions_tbr = terminal_static_frames - 2
        # Look for explanation: ... To Be Added ...

    def process(
        self,
        transition_queue: TransitionQueue,
        trajectory_type: TrajectoryType,
        penalty: float | None = None,
    ) -> None:
        """
        Removes static transitions from the given trajectory in-place.
        A penalty can be applied to the last transition (death transition) if provided.
        """
        # Remove beginning only for the first trajectory.
        if trajectory_type is TrajectoryType.FIRST:
            transition_queue.remove_first(self.initial_transitions_tbr)

        # Remove ending.
        if trajectory_type is TrajectoryType.LAST:           
            transition_queue.remove_last(self.terminal_transitions_tbr)
        else:
            transition_queue.remove_last(self.death_transitions_tbr)

        if penalty is not None:
            last_transition = transition_queue.get_last()

            last_transition.reward = penalty
            last_transition.death_transition = True

