from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto

import random

import numpy as np
import torch

from models.transition_classes import BaseBuffer, Transition

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



# ------------------------------------



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
class ReplayBucketType(Enum):
    WARMUP = auto()
    NEW = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    DEATH = auto()

@dataclass(slots=True)
class ReplayMemoryUnit:
    """
    One replay memory sample used for network optimization.
    """
    state: np.ndarray
    action: int
    q_target: torch.Tensor
    
    priority: float = 1.0
    # Priority used by prioritized replay.
    # Ignored when uniform replay or stratified replay is used.

    bucket: ReplayBucketType | None
    # Used only for stratified replay. 
    # Indicates which bucket the transition is currently in.

    death_transition: bool = False

    insert_id: int = 0
    # Used only for stratified replay to track the order of insertion
    # and remove the oldest transitions when buckets are full.

 
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


    def append(self, transition: Transition, q_target: torch.Tensor, warmup: bool = False) -> None:

        # New transitions should always be sampled at least once.
        if self.prioritized and self.__len__() > 0:
            initial_priority = max(unit.priority for unit in self._memory)
        else:
            initial_priority = 1.0

        self._memory.append(
            ReplayMemoryUnit(
                state=transition.state.copy(),
                action=transition.action,
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
        q_targets = torch.stack([transition.q_target for transition in batch]).to(device)
        
        return states, actions, q_targets
    

    def update_priorities(
        self,
        indices: list[int],
        td_errors_abs: torch.Tensor,
    ) -> None:

        for index, error in zip(indices, td_errors_abs):
            # self._memory[index].priority = (error + self.priority_epsilon) ** self.priority_alpha 
            
            self._memory[index].priority = (error + 1) ** self.priority_alpha # Experimental priority


    def get_states_total_size(self) -> int:
        """Returns the total size of the states in MB."""
        return np.sum([transition.state.nbytes for transition in self._memory]) / 1024**2


class StratifiedReplayMemory(BaseBuffer):
    """
    Stratified replay memory.

    Buckets
    -------
    NEW
        Newly inserted transitions. Every transition is trained at least once.

    LOW
        Low TD-error transitions.

    MEDIUM
        Medium TD-error transitions.

    HIGH
        High TD-error transitions.

    DEATH
        Death and near-death transitions.
    """

    def __init__(
        self,
        bucket_capacities: list[int],
        bucket_rates: dict[ReplayBucketType, float],
        td_statistics_beta: float,
        td_std_multiplier: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__()

        # Buckets
        self.bucket_capacities = bucket_capacities
        # [LOW, MEDIUM, HIGH, DEATH]
        
        self._warmup_bucket = deque()
        self._new_bucket = deque()
        self._low_bucket = deque()
        self._medium_bucket = deque()
        self._high_bucket = deque()
        self._death_bucket = deque()

        self.buckets = { 
            ReplayBucketType.WARMUP: self._warmup_bucket,
            ReplayBucketType.NEW: self._new_bucket,
            ReplayBucketType.LOW: self._low_bucket,
            ReplayBucketType.MEDIUM: self._medium_bucket,
            ReplayBucketType.HIGH: self._high_bucket,
            ReplayBucketType.DEATH: self._death_bucket,
        }
        # Dictionary makes accessing and operations easier sometimes.

        # Sampling
        self._new_index = 0 # Used to track _new_bucket sampling progress.
        self.bucket_rates = bucket_rates
        # [LOW, MEDIUM, HIGH, DEATH]

        # TD statistics
        self.td_statistics_beta = td_statistics_beta
        self.td_std_multiplier = td_std_multiplier

        self.td_errors = []
        self.td_mean = None
        self.td_std = None

        # Utils
        self.first_turn = True # Used for first turn of fresh training and loaded checkpoints.
        self.next_insert_id = np.array(0, dtype=np.long)
        # Used to track the order of insertion and remove the oldest transitions when buckets are full.

        self.verbose = verbose # For reporting


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def append(self, transition: Transition, q_target: torch.Tensor, warmup: bool = False) -> None:
        """
        Converts a Transition into a ReplayMemoryUnit and stores it in the
        temporary NEW bucket.
        """
        if warmup:
            self._warmup_bucket.append(
                ReplayMemoryUnit(
                    state=transition.state.copy(),
                    action=transition.action,
                    q_target=q_target,
                    death_transition=transition.death_transition,
                    bucket=ReplayBucketType.WARMUP,
                    insert_id=self.next_insert_id
                )
            )

        else:                
            self._new_bucket.append(
                ReplayMemoryUnit(
                    state=transition.state.copy(),
                    action=transition.action,
                    q_target=q_target,
                    death_transition=transition.death_transition,
                    bucket=ReplayBucketType.NEW,
                    insert_id=self.next_insert_id
                )
            )

        self.next_insert_id += 1


    def mark_death_windows(self) -> None:
        """
        Marks death and near-death transitions inside the warmup and new
        buckets to later be moved to the death bucket.
        """
        for source_bucket in (self._warmup_bucket, self._new_bucket):

            if len(source_bucket) == 0:
                continue

            protect_until = -1

            for index in range(len(source_bucket) - 1, -1, -1):

                transition = source_bucket[index]

                if transition.death_transition:
                    protect_until = max(-1, index - self.death_window)

                if transition.death_transition or index > protect_until:
                    transition.death_transition = True 
                    

    def sample(
        self,
        batch_size: int,
        network_optimization_period: int,
    ) -> list[ReplayMemoryUnit]:
        """
        Samples one optimization batch.

        Strategy
        --------
        1. Always sample current-turn transitions from the NEW bucket.
        2. If there are still warmup transitions, sample remaining
        instances from the WARMUP bucket.
        3. Otherwise sample from replay buckets in priority order:
            Death -> High -> Low -> Medium.
        Missing quota is transferred to the next bucket.
        """

        batch = []
        last_td_index = 0

        # ==========================================================
        # NEW transitions
        # ==========================================================

        new_count = min(network_optimization_period, len(self._new_bucket) - self._new_index)

        for i in range(self._new_index, self._new_index + new_count):
            batch.append(self._new_bucket[i])

        self._new_index += new_count

        remaining = batch_size - new_count

        last_td_index = batch_size - remaining  # = new_count 
        # 'new_count' is the one plus of last index of new transitions,
        # transitions to be used for TD stats calculation.

        if remaining <= 0:
            return last_td_index, batch 
        
        # ==========================================================
        # WARMUP transitions
        # ==========================================================

        if len(self._warmup_bucket) > 0:

            count = min(remaining, len(self._warmup_bucket))

            for _ in range(count):
                batch.append(self._warmup_bucket.popleft())

            remaining -= count

            last_td_index = batch_size - remaining 
            # This one is most likely not required, but we keep it just in case.
            # Why is not it required?: 
            # -> It is very unlikely that there will be any trnasition in the warmup bucket after the first turn. 

            # If replay buckets are not initialized yet, simply return.
            if remaining == 0:
                return last_td_index, batch

        # ==========================================================
        # Classified replay buckets
        # ==========================================================

        bucket_chain = [
            (
                self._death_bucket,
                round(remaining * self.bucket_rates[ReplayBucketType.DEATH]),
            ),
            (
                self._high_bucket,
                round(remaining * self.bucket_rates[ReplayBucketType.HIGH]),
            ),
            (
                self._low_bucket,
                round(remaining * self.bucket_rates[ReplayBucketType.LOW]),
            ),
            (
                self._medium_bucket,
                round(remaining * self.bucket_rates[ReplayBucketType.MEDIUM]),
            ),
        ]

        carry = 0

        for bucket, requested in bucket_chain:

            requested += carry

            take = min(requested, len(bucket))

            if take > 0:
                batch.extend(random.sample(bucket, take))

            carry = requested - take

        # ==========================================================
        # Final fallback
        # ==========================================================

        # We fill remaining quota with any available transitions from all buckets.
        if carry > 0:

            available = (
                list(self._death_bucket)
                + list(self._high_bucket)
                + list(self._low_bucket)
                + list(self._medium_bucket)
                + list(self._new_bucket)
            )

            available = [item for item in available if item not in batch]

            if len(available) > 0:
                batch.extend(
                    random.sample(
                        available,
                        min(carry, len(available))
                    )
                )

        return last_td_index, batch 
        # We return indices and batch from same name function of ReplayMemory.
        # To not break the training code, we return None for indices here since they are not used in stratified replay.

    def move_between_buckets(
        self,
        transitions: list[ReplayMemoryUnit],
        td_errors_abs: torch.Tensor,
    ) -> None:
        """
        Assigns sampled transitions to their corresponding replay buckets
        according to their latest TD errors.

        Death transitions are always kept inside the death bucket and are
        never classified by TD error.
        """

        low_boundary = self.td_mean - self.td_std * self.td_std_multiplier
        high_boundary = self.td_mean + self.td_std * self.td_std_multiplier

        for transition, td_error in zip(transitions, td_errors_abs):

            # ------------------------------------------------------
            # Determine destination bucket
            # ------------------------------------------------------

            if transition.death_transition:
                destination_bucket = ReplayBucketType.DEATH

            elif td_error < low_boundary:
                destination_bucket = ReplayBucketType.LOW

            elif td_error > high_boundary:
                destination_bucket = ReplayBucketType.HIGH

            else:
                destination_bucket = ReplayBucketType.MEDIUM

            # ------------------------------------------------------
            # Move transition
            # ------------------------------------------------------
        
            try:
                self.buckets[transition.bucket].remove(transition)
                # We do this, because we pop from warmup bucket during sampling.
                # Unless we incluede remove line into try block, we would get an error.
            except ValueError:
                pass

            transition.bucket = destination_bucket

            self.buckets[destination_bucket].append(transition)

        # Remove oldest transitions if any bucket exceeds its capacity.
        self._clip_buckets()

    def register_td_errors(self, td_errors: list[float]) -> None:
        self.td_errors.extend(td_errors)

    def update_td_statistics(self) -> None:
        """
        Updates exponentially moving TD-error statistics.

        Only TD errors originating from Warmup/New transitions should be
        passed to this function.
        """

        if not self.td_errors:
            return

        td_errors = torch.tensor(self.td_errors)

        batch_mean = td_errors.mean().item()
        batch_std = td_errors.std(unbiased=False).item()

        if self.first_turn:
            self.td_mean = batch_mean
            self.td_std = batch_std
            return

        beta = self.td_statistics_beta

        self.td_mean = (
            beta * self.td_mean
            + (1.0 - beta) * batch_mean
        )

        self.td_std = (
            beta * self.td_std
            + (1.0 - beta) * batch_std
        )

        self.td_errors.clear()

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
        q_targets = torch.stack([transition.q_target for transition in batch]).to(device)
        
        return states, actions, q_targets

    def report(self):
        if not self.verbose:
            return

        total_len = len(self._low_bucket) + len(self._medium_bucket) + len(self._high_bucket) + len(self._death_bucket) 

        print(f"Replay Buffer | Mean TD-error: {self.td_mean:.4f} | Std TD-error: {self.td_std:.4f}")
        print("Bucket Sizes and Rates:" + "\n"
            + f"Low: {len(self._low_bucket)} | {len(self._low_bucket) / total_len * 100:.2f}% "
            + f"Medium: {len(self._medium_bucket)} | {len(self._medium_bucket) / total_len * 100:.2f}% "
            + f"High: {len(self._high_bucket)} | {len(self._high_bucket) / total_len * 100:.2f}% "
            + f"Death: {len(self._death_bucket)} | {len(self._death_bucket) / total_len * 100:.2f}% ")

    def state_dict(self) -> dict:
        """Serialization."""

    def load_state_dict(self, state: dict) -> None:
        """Deserialization."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clip_buckets(self) -> None:
        """
        Clips replay buckets to their maximum capacities by removing the
        oldest transitions (smallest insert_id).
        """

        managed_buckets = (
            self._low_bucket,
            self._medium_bucket,
            self._high_bucket,
            self._death_bucket,
        )

        for bucket, bucket_capacity in zip(managed_buckets, self.bucket_capacities):

            excess = len(bucket) - bucket_capacity

            if excess <= 0:
                continue

            # Oldest transitions to remove.
            remove_ids = [
                transition.insert_id
                for transition in sorted(
                    bucket,
                    key=lambda transition: transition.insert_id,
                )[:excess]
            ]

            # Rebuild bucket without removed transitions.
            filtered_bucket = deque(
                (
                    transition
                    for transition in bucket
                    if transition.insert_id not in remove_ids
                )
            )

            bucket.clear()
            bucket.extend(filtered_bucket)