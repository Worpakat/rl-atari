from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque

import random

import numpy as np
import torch

from utils.frame_processing import convert_and_norm_sequence



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

    def get_raw_sequence(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Numpy array of shape (sequence_length, H, W) or (sequence_length, C, H, W).
        """
        if not self.is_ready():
            raise RuntimeError("Frame sequence is not yet complete.")
        
        return np.stack(tuple(self._buffer), axis=0)
    

    def get_sequence(self) -> torch.Tensor:
        """
        Returns
        -------
        np.ndarray
            Preprocessed numpy array of shape (sequence_length, C, H, W).
        """
        return convert_and_norm_sequence(self.get_raw_sequence())



##----------NEC_Specials----------##

@dataclass(slots=True)
class ReplayMemoryUnit:
    """
    One replay memory sample used for network optimization.
    """

    state: np.ndarray
    action: int
    q_target: torch.Tensor
    
    # Priority used by prioritized replay.
    # Ignored when uniform replay is used.
    priority: float = 1.0

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

    def load_state_dict(self, state_dict: dict) -> None:
        self._memory = deque(
            state_dict["memory"],
            maxlen=self._memory.maxlen,
        )

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

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        priority_alpha: float = 0.6,
        priority_epsilon: float = 1e-5,
    ):
        super().__init__(capacity)

        self.prioritized = prioritized
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon


    def append(
        self,
        state: np.ndarray,
        action: int,
        q_target: torch.Tensor,
    ) -> None:

        # New transitions should always be sampled at least once.
        if self.prioritized and self.__len__() > 0:
            initial_priority = max(unit.priority for unit in self._memory)
        else:
            initial_priority = 1.0

        self._memory.append(
            ReplayMemoryUnit(
                state=state.copy(),
                action=action,
                q_target=q_target,
                priority=initial_priority,
            )
        )


    def sample(
        self,
        batch_size: int,
    ) -> tuple[list[int], list[ReplayMemoryUnit]]:

        if not self.prioritized:
            indices = random.sample(range(self.__len__()), batch_size)
            batch = [self._memory[i] for i in indices]
            return indices, batch

        priorities = np.asarray(
            [unit.priority for unit in self._memory],
            dtype=np.float64,
        )

        probabilities = priorities / priorities.sum()

        indices = np.random.choice(
            len(self._memory),
            size=batch_size,
            replace=False,
            p=probabilities,
        )
        
        batch = [self._memory[i] for i in indices]

        return indices.tolist(), batch
    

    def extract_batch(
            self, 
            batch: list[ReplayMemoryUnit],
            device: torch.device = torch.device("cpu"),
            ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """
        Helper method. Extracts, converts, and returns batch of states, actions and Q-targets.
        """

        states = (
            torch.from_numpy(
            convert_and_norm_sequence(np.stack([transition.state for transition in batch]))
            ).unsqueeze(2)
            .to(device)
            ) # For Grayscale
        
        actions = [transition.action for transition in batch]
        q_targets = torch.stack([transition.q_target for transition in batch]).unsqueeze(1).to(device)
        
        return states, actions, q_targets
    

    def update_priorities(
        self,
        indices: list[int],
        td_errors: torch.Tensor,
    ) -> None:

        td_errors = td_errors.detach().abs().cpu().tolist()

        for index, error in zip(indices, td_errors):
            self._memory[index].priority = (error + self.priority_epsilon) ** self.priority_alpha 


    def get_states_total_size(self) -> int:
        """Returns the total size of the states in MB."""
        return np.sum([transition.state.nbytes for transition in self._memory]) / 1024**2







##==============OLD_IMPLEMENTATION=====================

# class ReplayMemory(BaseBuffer):
#     def append(self, state: np.ndarray, action: int, q_target: float) -> None:
#         self._memory.append(
#             ReplayMemoryUnit(
#                 state=state.copy(),
#                 action=action,
#                 q_target=q_target,
#             )
#         )
#     def extract_batch(
#             self, 
#             batch: list[ReplayMemoryUnit],
#             device: torch.device = torch.device("cpu"),
#             ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
#         """
#         Helper method. Extracts, converts, and returns batch of states, actions and Q-targets.
#         """

#         states = (
#             torch.from_numpy(
#             convert_and_norm_sequence(np.stack([transition.state for transition in batch]))
#             ).unsqueeze(2)
#             .to(device)
#             ) # For Grayscale
        
#         actions = [transition.action for transition in batch]
#         q_targets = torch.stack([transition.q_target for transition in batch]).to(device)
        
#         return states, actions, q_targets

#     def get_states_total_size(self) -> int:
#         """Returns the total size of the states in MB."""
#         return np.sum([transition.state.nbytes for transition in self._memory]) / 1024**2


