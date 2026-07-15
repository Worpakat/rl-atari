from dataclasses import dataclass
from abc import ABC, abstractmethod
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



##----------NEC_Specials----------##

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

@dataclass(slots=True)
class ReplayMemoryUnit:
    """
    One replay memory sample used for network optimization.
    """

    state: np.ndarray
    action: int
    q_target: torch.Tensor

# ------------------------------------


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

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self._memory.clear()
        self._memory.extend(state_dict["memory"])

    @abstractmethod
    def append(self, *args, **kwargs):
        pass


#-----Used_for_DSAE--------

class SequenceReplayBuffer(BaseBuffer):
    """
    Replay buffer storing frame sequences.

    The current implementation stores only frame sequences for Phase 0 DSAE
    training. It is intentionally designed to be extended later with actions,
    rewards, next states and terminal flags for RL training.
    """

    def append(
        self,
        frame_sequence: np.ndarray,
    ) -> None:

        self._memory.append(frame_sequence.copy())

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:

        batch = super().sample(batch_size)

        return {
            "frames": torch.from_numpy(np.stack(batch, axis=0))
        }
    

#-----Used_for_NEC--------

class ReplayMemory(BaseBuffer):
    def append(self, state: np.ndarray, action: int, q_target: float) -> None:
        self._memory.append(
            ReplayMemoryUnit(
                state=state.copy(),
                action=action,
                q_target=q_target,
            )
        )
    def extract_batch(self, batch: list[ReplayMemoryUnit]) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """
        Helper method. Extracts, converts, and returns batch of states, actions and Q-targets.
        """

        states = torch.from_numpy(np.stack([transition.state for transition in batch]))
        actions = [transition.action for transition in batch]
        q_targets = torch.stack([transition.q_target for transition in batch])
        
        return states, actions, q_targets

    def get_states_total_size(self) -> int:
        """Returns the total size of the states in MB."""
        return np.sum([transition.state.nbytes for transition in self._memory]) / 1024**2

class TransitionQueue(BaseBuffer):

    def __getitem__(
        self,
        index: int,
    ) -> Transition:

        return self._memory[index]

    def is_full(self) -> bool:
        return len(self) == self.capacity

    def append(
        self,
        transition: Transition,
    ) -> None:

        self._memory.append(transition)