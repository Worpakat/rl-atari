from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto

from pathlib import Path
import random

import numpy as np
import torch

from models.transition_classes import BaseBuffer, Transition

from utils.frame_processing import convert_and_norm_sequence
from utils.misc import ensure_directory



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
class FrameUnit:
    """
    Stores a frame and counts the number of transitions that use it. 
    """
    frame_id: int
    frame: np.ndarray
    transition_count: int = 0


@dataclass(slots=True, eq=False)
class ReplayMemoryUnit:
    """
    One replay memory sample used for network optimization.
    """
    start_frame_id: int
    action: int
    q_target: np.ndarray
    # q_target: torch.Tensor # Keeping old just in case.
    
    priority: float = 1.0
    # Priority used by prioritized replay.
    # Ignored when uniform replay or stratified replay is used.

    bucket: ReplayBucketType | None = None
    # Used only for stratified replay. 
    # Indicates which bucket the transition is currently in.

    death_transition: bool = False

    insert_id: int = 0
    # Used only for stratified replay to track the order of insertion
    # and remove the oldest transitions when buckets are full.

    state: np.ndarray | None = None # This is kept for old save files. 


class ReplayMemory(BaseBuffer):

    def __init__(
        self,
        capacity: int,
        prioritized: bool = False,
        priority_alpha: float = 0.6,
        priority_epsilon: float = 1e-5,
        verbose: bool = False
    ):
        super().__init__(capacity)

        self.prioritized = prioritized
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon

        self.verbose = verbose


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
                q_target=q_target.to("cpu").detach().numpy(),
                priority=initial_priority,                
            )
        )


    def sample(
        self,
        batch_size: int,
        network_optimization_period: int,
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
        q_targets = torch.from_numpy(np.stack([transition.q_target for transition in batch])).to(device)
        # q_targets = torch.stack([transition.q_target for transition in batch]).to(device)
        
        return states, actions, q_targets
    
    def update_priorities(
        self,
        indices: list[int],
        td_errors_abs: torch.Tensor,
    ) -> None:

        for index, error in zip(indices, td_errors_abs):
            # self._memory[index].priority = (error + self.priority_epsilon) ** self.priority_alpha 
            
            self._memory[index].priority = (error + 1) ** self.priority_alpha # Experimental priority

    def report(self):
        if not self.verbose:
            return

        print(f"Replay Memory Size: {self.__len__()}")
        

    def get_states_total_size(self) -> int:
        """Returns the total size of the states in MB."""
        return np.sum([transition.state.nbytes for transition in self._memory]) / 1024**2

    def save(
        self,
        save_directory: str | Path,
        chunk_size: int = 5000,
    ) -> None:
        """
        Saves the stratified replay memory into a directory.

        Directory structure
        -------------------
        replay_memory/
            metadata.pt
            replay_000.pt
            replay_001.pt
            ...
        """
        save_directory = ensure_directory(save_directory)

        # Metadata
        metadata = {
            "capacity": self.capacity,
            "prioritized": self.prioritized,
            "priority_alpha": self.priority_alpha,
            "priority_epsilon":self.priority_epsilon,
        }
        torch.save(metadata, (save_directory / "metadata.pt"))

        # Memory
        memory = list(self._memory)

        for chunk_index, start in enumerate(range(0, len(memory), chunk_size)):
            chunk = memory[start : start + chunk_size]

            torch.save(
                chunk,
                save_directory / f"{"replay"}_{chunk_index:03d}.pt",
            )

    def load(
        self,
        save_directory: str | Path,
        use_checkpoint_config: bool = True,
    ) -> None:
        """
        Loads a previously saved stratified replay memory.
        """
        save_directory = Path(save_directory)

        if not save_directory.exists():
            raise FileNotFoundError(
                f"Replay memory directory does not exist: {save_directory}"
            )

        # Metadata
        metadata = torch.load(
            save_directory / "metadata.pt",
            map_location="cpu",
            weights_only=False,
        )

        if use_checkpoint_config: # Use the saved metadata configurations from the checkpoint.
            self.capacity=metadata["capacity"] 
            self.prioritized=metadata["prioritized"]
            self.priority_alpha=metadata["priority_alpha"]
            self.priority_epsilon=metadata["priority_epsilon"]

        # Clear existing replay
        self._memory.clear()

        # Load memory chunks
        files = sorted(save_directory.glob("replay_*.pt"))

        for file in files:
            try:
                chunk = torch.load(
                    file,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as e:
                print(f"Error loading {file}: {e}")

            self._memory.extend(chunk)

        print(f"Loaded replay memory:" 
              f"\n  size: {len(self._memory)} transitions" 
              f"\n  ({self.get_states_total_size():.2f} MB)" 
              f"\n  capacity: {self.capacity}" 
              f"\n  prioritized: {self.prioritized}"
              f"\n  priority_alpha: {self.priority_alpha}"
              f"\n  priority_epsilon: {self.priority_epsilon}"
              ) 
        


class StratifiedReplayMemory():
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
        death_window: int = 15,
        add_new_times: int = 1,
        new_incluede_period: int = 1,
        sequence_length: int = 4,
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
        self.death_window = death_window # Used to mark death and near-death transitions.
        # [LOW, MEDIUM, HIGH, DEATH]

        self.bucket_rates = {
            ReplayBucketType.LOW: bucket_rates[0],
            ReplayBucketType.MEDIUM: bucket_rates[1],
            ReplayBucketType.HIGH: bucket_rates[2],
            ReplayBucketType.DEATH: bucket_rates[3],
        }
        

        # TD statistics
        self.td_statistics_beta = td_statistics_beta
        self.td_std_multiplier = td_std_multiplier

        self.td_errors = []
        self.td_mean = None
        self.td_std = None

        self.low_boundary = None
        self.high_boundary = None

        # --------------------------------------------
        # Utils
        # --------------------------------------------
        
        # Frame Management
        self._frames: dict[int, FrameUnit] = {} # Dictionary mapping frame IDs to frames
        self._sequence_length = sequence_length # Transition sequence length
        self.next_frame_id = 0 # Used to track unique frame IDs
        self._last_frame_ids = [] # Used to track sequential overlapping frames during ``append()``.
        self.next_insert_id = 0 # Used to track unique insert IDs of 'ReplayMemoryUnit's.


        self.first_turn = True # Used for first turn of fresh training and loaded checkpoints.
        # Used to track the order of insertion and remove the oldest transitions when buckets are full.
        
        self.add_new_times = add_new_times # How many times to add new transitions to the batch during sampling. 
        self.new_incluede_period = new_incluede_period # How many every optimization steps to include new transitions in the batch.

        self._new_incluede_counter = 1 # Used to track new_incluede_period progress. 
        # ! Start from 1 to include new transitions in the first optimization step.
        self._new_index = 0 # Used to track _new_bucket sampling progress.
        
        self.verbose = verbose # For reporting


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        transition: Transition,
        q_target: torch.Tensor,
        warmup: bool = False,
    ) -> None:
        """
        Converts a Transition into a ReplayMemoryUnit while storing each
        frame only once.

        Consecutive transitions reuse overlapping frames. A death transition
        terminates the current frame sequence, so the next transition starts
        a new sequence.
        """

        # ------------------------------------------------------------
        # 1. Determine which bucket receives the transition
        # ------------------------------------------------------------
        bucket = ReplayBucketType.WARMUP if warmup else ReplayBucketType.NEW
        target_bucket = self._warmup_bucket if warmup else self._new_bucket

        # A transition state consists of consecutive frames.
        state_frames = transition.state

        # ------------------------------------------------------------
        # 2. First transition of a sequence
        # ------------------------------------------------------------
        if len(self._last_frame_ids) == 0:

            start_frame_id = self.next_frame_id

            for frame in state_frames:
                frame_unit = FrameUnit(
                    frame_id=self.next_frame_id,
                    frame=frame.copy(),
                    transition_count=1,
                )

                self._frames[self.next_frame_id] = frame_unit
                self._last_frame_ids.append(self.next_frame_id)

                self.next_frame_id += 1

        # ------------------------------------------------------------
        # 3. Consecutive transition
        # ------------------------------------------------------------
        else:

            # The first N-1 frames are already present because the
            # states overlap.
            #
            # Example for 4-frame states:
            #
            # previous: [f0, f1, f2, f3]
            # current:  [f1, f2, f3, f4]
            #
            # Reuse f1, f2, f3 and create only f4.

            overlap = min(
                len(self._last_frame_ids),
                self._sequence_length - 1,
            )

            # The state starts at the corresponding overlapping frame.
            start_frame_id = self._last_frame_ids[-overlap]

            # Increase reference counts for the frames belonging to
            # this new transition.
            for frame_id in self._last_frame_ids[-overlap:]:
                self._frames[frame_id].transition_count += 1

            # Add newly appearing frames.
            new_frame_start = overlap

            for frame in state_frames[new_frame_start:]:
                frame_unit = FrameUnit(
                    frame_id=self.next_frame_id,
                    frame=frame.copy(),
                    transition_count=1,
                )

                self._frames[self.next_frame_id] = frame_unit
                self._last_frame_ids.append(self.next_frame_id)

                self.next_frame_id += 1

            # Keep only the frame IDs necessary to construct the next
            # overlapping state.
            self._last_frame_ids = self._last_frame_ids[-self._sequence_length:]

        # ------------------------------------------------------------
        # 4. Create ReplayMemoryUnit
        # ------------------------------------------------------------
        replay_unit = ReplayMemoryUnit(
            start_frame_id=start_frame_id,
            action=transition.action,
            q_target=q_target.to("cpu").detach().numpy(),
            death_transition=transition.death_transition,
            bucket=bucket,
            insert_id=self.next_insert_id,
        )

        target_bucket.append(replay_unit)

        self.next_insert_id += 1

        # ------------------------------------------------------------
        # 5. Death transition terminates the sequence
        # ------------------------------------------------------------
        if transition.death_transition:
            self._last_frame_ids = []
        


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

    def can_sample(self, batch_size: int) -> bool:
        """
        This function is not needed for this Replay Buffer.
        It exist for compatibility with other Replay Buffers.
        """
        return True

    def update_optimization_steps( # ! EXPERIMENTAL: "Wait and Opt"
        self, steps: int, 
        network_optimization_period: int
        ) -> int:
        """
        Determines how many optimization steps to perform in the current turn
        with respect to the "wait and Opt" strategy. Uses `new_incluede_period`.
        """
        if (
            self.new_incluede_period > 1 and 
            ((self._new_incluede_counter - 1) % self.new_incluede_period == 0)
            ): 
            # If it is new bucket transition including turn, we optimize until new bucket transitions is fnished.
            return int(len(self._new_bucket) / network_optimization_period) + 1

        # Otherwise we optimize as many as we do originally.
        return steps

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
        td_index_border = (0, 0)
        remaining = batch_size

        # ==========================================================
        # NEW transitions
        # ==========================================================
        if ((self._new_incluede_counter - 1) % self.new_incluede_period )== 0: # ! EXPERIMENTAL: "Wait and Opt"

            new_count = min(network_optimization_period, (len(self._new_bucket) - self._new_index))

            for _ in range(self.add_new_times): # Add new transitions multiple times to the batch.
                for i in range(self._new_index, self._new_index + new_count):
                    batch.append(self._new_bucket[i])


            if self.first_turn:        
                self._new_index += new_count

            remaining = batch_size - new_count * self.add_new_times

            td_index_border = (new_count * (self.add_new_times - 1), new_count * self.add_new_times)  
            # This is the index border for TD stats calculation.

            if remaining <= 0:
                # print("new return batch size:", len(batch))
                return td_index_border, batch 

        
        # ==========================================================
        # WARMUP transitions
        # ==========================================================

        if len(self._warmup_bucket) > 0:

            count = min(remaining, len(self._warmup_bucket))

            for _ in range(count):
                batch.append(self._warmup_bucket.popleft())

            remaining -= count

            td_index_border = (td_index_border[0], (batch_size - remaining))
            # This one is most likely not required, but we keep it just in case.
            # Why is not it required?: 
            # -> It is very unlikely that there will be any trnasition in the warmup bucket after the first turn. 

            # If replay buckets are not initialized yet, simply return.
            if remaining == 0:
                return td_index_border, batch


        # ==========================================================
        # Classified replay buckets
        # ==========================================================

        death_quota = round(remaining * self.bucket_rates[ReplayBucketType.DEATH])
        high_quota  = round(remaining * self.bucket_rates[ReplayBucketType.HIGH])
        low_quota   = round(remaining * self.bucket_rates[ReplayBucketType.LOW])

        # --------------------------
        # Death
        # --------------------------

        take = min(death_quota, len(self._death_bucket), remaining)

        if take > 0:
            batch.extend(random.sample(self._death_bucket, take))

        remaining -= take

        death_missing = death_quota - take

        # --------------------------
        # High
        # --------------------------

        requested = high_quota + death_missing
        # We take the missing death transitions from the high bucket.

        take = min(requested, len(self._high_bucket), remaining)

        if take > 0:
            batch.extend(random.sample(self._high_bucket, take))

        remaining -= take

        # high_missing = requested - take

        # --------------------------
        # Low
        # --------------------------

        requested = low_quota

        take = min(requested, len(self._low_bucket), remaining)

        if take > 0:
            batch.extend(random.sample(self._low_bucket, take))

        remaining -= take

        # low_missing = requested - take

        # --------------------------
        # Medium
        # --------------------------

        # We take what's left from the medium bucket. It contains medium_quota, low_missing and high_missing. 
        requested = remaining
        take = min(requested, len(self._medium_bucket))

        if take > 0:
            batch.extend(random.sample(self._medium_bucket, take))

        remaining -= take

        # ==========================================================
        # Final fallback
        # ==========================================================

        # We fill remaining quota with any available transitions from all buckets.
        
        if remaining > 0:

            # print("DROPPPED TO RANDOM SAMPLING !!!")
            
            available = (
                list(self._death_bucket)
                + list(self._high_bucket)
                + list(self._low_bucket)
                + list(self._medium_bucket)
                + list(self._new_bucket)
            )

            batch_ids = {id(transition) for transition in batch}

            available = [
                transition
                for transition in available
                if id(transition) not in batch_ids
            ]

            take = min(remaining, len(available))
            batch.extend(random.sample(available, take))

            # print("Drop, remaining:", remaining, " | available:", len(available), " | batch size:", len(batch))
                
        # print("Last return Batch size:", len(batch))

        return td_index_border, batch 
        # We return indices and batch from same name function of ReplayMemory.
        # To not break the training code, we return None for indices here since they are not used in stratified replay.

    def move_between_buckets(
        self,
        transitions: list[ReplayMemoryUnit],
        td_errors_abs: list[float],
    ) -> None:
        """
        Assigns sampled transitions to their corresponding replay buckets
        according to their latest TD errors.

        Death transitions are always kept inside the death bucket and are
        never classified by TD error.
        """
        new_transitions_count = 0
        # Determine how many is there new transitions at the beginning
        for transition in transitions:
            if transition.bucket == ReplayBucketType.NEW:
                new_transitions_count += 1
            else:
                break

        # We only want to move the last added new transitions to their corresponding buckets.
        start_index = new_transitions_count / self.add_new_times 
        start_index = int((self.add_new_times-1) * start_index) 
        
        # start_index = int((self.add_new_times-1) * new_transitions_count) # Bugged version. 

        transitions = transitions[start_index:]
        td_errors_abs = td_errors_abs[start_index:]

        for transition, td_error in zip(transitions, td_errors_abs):

            # ------------------------------------------------------
            # Determine destination bucket
            # ------------------------------------------------------

            if transition.death_transition:
                destination_bucket = ReplayBucketType.DEATH

            elif td_error < self.low_boundary:
                destination_bucket = ReplayBucketType.LOW

            elif td_error > self.high_boundary:
                destination_bucket = ReplayBucketType.HIGH

            else:
                destination_bucket = ReplayBucketType.MEDIUM

            # ------------------------------------------------------
            # Move transition
            # ------------------------------------------------------

            if transition.bucket == destination_bucket:
                continue

            # Remove transition from its current bucket.
            if transition.bucket != ReplayBucketType.WARMUP: # WARMUP transitions are already removed from buckets during sampling.
                self.buckets[transition.bucket].remove(transition)

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

            self.td_errors.clear()

            # Update boundries
            self.low_boundary = self.td_mean - self.td_std * self.td_std_multiplier
            self.high_boundary = self.td_mean + self.td_std * self.td_std_multiplier

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

        # Update boundries
        self.low_boundary = self.td_mean - self.td_std * self.td_std_multiplier
        self.high_boundary = self.td_mean + self.td_std * self.td_std_multiplier

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
                convert_and_norm_sequence(
                    np.stack([ # Stacking the batch
                        np.stack([ # Stacking the transition frames
                            self._frames[unit.start_frame_id + i].frame
                            for i in range(self._sequence_length)
                        ])
                        for unit in batch
                    ])
                )
            )
            .unsqueeze(2) # For Grayscale
            .to(device)
        ) 

        actions = [transition.action for transition in batch]
        q_targets = torch.from_numpy(np.stack([transition.q_target for transition in batch])).to(device)
        
        return states, actions, q_targets

    def reset_new_bucket(self) -> None:
        if ((self._new_incluede_counter - 1) % self.new_incluede_period )== 0: # ! EXPERIMENTAL: "Wait and Opt"
            self._new_bucket.clear()

        self._new_index = 0

        self._new_incluede_counter += 1  

    def report(self):
        if not self.verbose:
            return

        total_len = len(self._low_bucket) + len(self._medium_bucket) + len(self._high_bucket) + len(self._death_bucket) 

        print(f"Replay Buffer | Mean TD-error: {self.td_mean:.4f} | Std TD-error: {self.td_std:.4f}")
        print("Bucket Sizes and Rates:" + "\n"
            + f"Low: {len(self._low_bucket)}, {len(self._low_bucket) / total_len * 100:.2f}% | "
            + f"Medium: {len(self._medium_bucket)}, {len(self._medium_bucket) / total_len * 100:.2f}% | "
            + f"High: {len(self._high_bucket)}, {len(self._high_bucket) / total_len * 100:.2f}% | "
            + f"Death: {len(self._death_bucket)}, {len(self._death_bucket) / total_len * 100:.2f}% | "
            + f"New: {len(self._new_bucket)}, |"
            + f"Warmup: {len(self._warmup_bucket)}, |"
            + f"Total: {total_len}, |"
            + f"Frames: {len(self._frames)} / {self.get_states_total_size()} MB")


    def save(
        self,
        save_directory: str | Path,
        chunk_size: int = 20_000,
    ) -> None:
        """
        Saves the stratified replay memory and shared frame store.

        Directory structure
        -------------------
        replay_memory/
            metadata.pt
            low.pt
            medium.pt
            high.pt
            death.pt
            frames_000.pt
            frames_001.pt
            ...
        """
        save_directory = ensure_directory(save_directory)

        metadata = {
            "td_mean": self.td_mean,
            "td_std": self.td_std,
            "next_insert_id": self.next_insert_id,
            "next_frame_id": self.next_frame_id,
            "bucket_capacities": self.bucket_capacities,
        }

        torch.save(metadata, save_directory / "metadata.pt")

        # Save each replay bucket as a single file.
        torch.save(list(self._low_bucket), save_directory / "low.pt")
        torch.save(list(self._medium_bucket), save_directory / "medium.pt",)
        torch.save(list(self._high_bucket), save_directory / "high.pt")
        torch.save(list(self._death_bucket), save_directory / "death.pt")

        # Save shared frames in chunks.
        frames = list(self._frames.values())

        for chunk_index, start in enumerate(
            range(0, len(frames), chunk_size)
        ):
            chunk = frames[start:start + chunk_size]

            torch.save(
                chunk,
                (save_directory / f"frames_{chunk_index:03d}.pt"),
            )


    def load(
        self, 
        save_directory: str | Path,
        use_checkpoint_config: bool = True,
        ) -> None:
        """
        Loads a stratified replay memory saved by save().

        Reconstructs:
            - replay buckets
            - frame dictionary
            - replay metadata
        """
        save_directory = Path(save_directory)

        if not save_directory.exists():
            raise FileNotFoundError(
                f"Replay memory directory does not exist: {save_directory}"
            )

        # ---------------------------------------------------------
        # 1. Load metadata
        # ---------------------------------------------------------
        metadata = torch.load(
            save_directory / "metadata.pt",
            map_location="cpu",
            weights_only=False,
        )

        self.td_mean = metadata["td_mean"]
        self.td_std = metadata["td_std"]
        self.next_insert_id = metadata["next_insert_id"]
        self.next_frame_id = metadata["next_frame_id"]

        if use_checkpoint_config: # Use the saved bucket capacities from the checkpoint.
            self.bucket_capacities = metadata["bucket_capacities"]

        # Calculate last TD boundries.
        self.low_boundary = self.td_mean - self.td_std * self.td_std_multiplier
        self.high_boundary = self.td_mean + self.td_std * self.td_std_multiplier

        # It is not actually a first turn
        self.first_turn = False

        # ---------------------------------------------------------
        # 2. Load buckets
        # ---------------------------------------------------------
        self._low_bucket = deque(
            torch.load(
                save_directory / "low.pt",
                map_location="cpu",
                weights_only=False,
            )
        )

        self._medium_bucket = deque(
            torch.load(
                save_directory / "medium.pt",
                map_location="cpu",
                weights_only=False,
            )
        )

        self._high_bucket = deque(
            torch.load(
                save_directory / "high.pt",
                map_location="cpu",
                weights_only=False,
            )
        )

        self._death_bucket = deque(
            torch.load(
                save_directory / "death.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        # Sanity check.
        self._make_sure_transition_buckets(self._low_bucket, "low")
        self._make_sure_transition_buckets(self._medium_bucket, "medium")
        self._make_sure_transition_buckets(self._high_bucket, "high")
        self._make_sure_transition_buckets(self._death_bucket, "death")


        # ---------------------------------------------------------
        # 3. Reconstruct frame dictionary
        # ---------------------------------------------------------
        self._frames = {}

        frame_files = sorted(save_directory.glob("frames_*.pt"))

        for frame_file in frame_files:
            frame_chunk = torch.load(
                frame_file,
                map_location="cpu",
                weights_only=False,
            )

            for frame in frame_chunk:
                self._frames[frame.frame_id] = frame

        print(
            f"Replay memory loaded from {save_directory}:"
            f"\n  Total Frames: {len(self._frames)}"
            f"\n  Low:    {len(self._low_bucket)}"
            f"\n  Medium: {len(self._medium_bucket)}"
            f"\n  High:   {len(self._high_bucket)}"
            f"\n  Death:  {len(self._death_bucket)}"
        )


    def get_states_total_size(self) -> int:
        """Returns the total size of the states in MB."""
        sum_mbs = 0

        
        sum_mbs = sum([frame_unit.frame.nbytes for frame_unit in self._frames.values()]) / 1024**2

        return sum_mbs
        
    def _clip_buckets(self) -> None:
        """
        Clips replay buckets to their maximum capacities by removing the
        oldest transitions (smallest insert_id).

        When a transition is removed, the reference count of every frame
        belonging to that transition is decremented. Frames with no remaining
        transitions are removed from the frame store.
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
            transitions_to_remove = sorted(
                bucket,
                key=lambda transition: transition.insert_id,
            )[:excess]

            remove_ids = {transition.insert_id for transition in transitions_to_remove}

            # Update frame reference counts.
            for transition in transitions_to_remove:
                start_id = transition.start_frame_id

                for frame_id in range(
                    start_id,
                    start_id + self._sequence_length
                ):
                    frame = self._frames[frame_id]

                    frame.transition_count -= 1

                    if frame.transition_count == 0: # Frame is no longer used, remove it.
                        del self._frames[frame_id]

            # Rebuild bucket without removed transitions.
            filtered_bucket = deque(
                transition
                for transition in bucket
                if transition.insert_id not in remove_ids
            )

            bucket.clear()
            bucket.extend(filtered_bucket)
  
    def _make_sure_transition_buckets(self, bucket, prefix):
        """ 
        This is a sanity check to ensure that the loaded transitions have the correct bucket type.
        """
        print("make sure", prefix)
        for transition in bucket: 
            if transition.bucket != ReplayBucketType[prefix.upper()]:
                print(f"Transition {transition.insert_id} has incorrect bucket type: {transition.bucket}")                
                transition.bucket = ReplayBucketType[prefix.upper()]


#============================================================
#============================================================
#-----------------------OLD_CODE-----------------------------

# ----------OLD_VERSION----------

# @dataclass(slots=True, eq=False)
# class ReplayMemoryUnit:
#     """
#     One replay memory sample used for network optimization.
#     """
#     state: np.ndarray
#     action: int
#     q_target: np.ndarray
#     # q_target: torch.Tensor # Keeping old just in case.
    
#     priority: float = 1.0
#     # Priority used by prioritized replay.
#     # Ignored when uniform replay or stratified replay is used.

#     bucket: ReplayBucketType | None = None
#     # Used only for stratified replay. 
#     # Indicates which bucket the transition is currently in.

#     death_transition: bool = False

#     insert_id: int = 0
#     # Used only for stratified replay to track the order of insertion
#     # and remove the oldest transitions when buckets are full.

# -------------------------------------------------------


    # def append(self, transition: Transition, q_target: torch.Tensor, warmup: bool = False) -> None:
    #     """
    #     Converts a Transition into a ReplayMemoryUnit and stores it in the
    #     temporary NEW bucket.
    #     """
    #     if warmup:
    #         self._warmup_bucket.append(
    #             ReplayMemoryUnit(
    #                 state=transition.state.copy(),
    #                 action=transition.action,
    #                 q_target=q_target.to("cpu").detach().numpy(),
    #                 death_transition=transition.death_transition,
    #                 bucket=ReplayBucketType.WARMUP,
    #                 insert_id=self.next_insert_id
    #             )
    #         )

    #     else:                
    #         self._new_bucket.append(
    #             ReplayMemoryUnit(
    #                 state=transition.state.copy(),
    #                 action=transition.action,
    #                 q_target=q_target.to("cpu").detach().numpy(),
    #                 death_transition=transition.death_transition,
    #                 bucket=ReplayBucketType.NEW,
    #                 insert_id=self.next_insert_id
    #             )
    #         )

    #     self.next_insert_id += 1
    

    # def extract_batch(
    #         self, 
    #         batch: list[ReplayMemoryUnit],
    #         device: torch.device = torch.device("cpu"),
    #         ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    #     """
    #     Helper method. Extracts, converts, and returns batch of states, actions and Q-targets.
    #     """

    #     states = (
    #         torch.from_numpy(
    #         convert_and_norm_sequence(np.stack([transition.state for transition in batch]))
    #         ).unsqueeze(2)
    #         .to(device)
    #     ) # For Grayscale

    #     actions = [transition.action for transition in batch]
    #     q_targets = torch.from_numpy(np.stack([transition.q_target for transition in batch])).to(device)
        
    #     return states, actions, 
    

    # def _clip_buckets(self) -> None:
    #     """
    #     Clips replay buckets to their maximum capacities by removing the
    #     oldest transitions (smallest insert_id).
    #     """

    #     managed_buckets = (
    #         self._low_bucket,
    #         self._medium_bucket,
    #         self._high_bucket,
    #         self._death_bucket,
    #     )

    #     for bucket, bucket_capacity in zip(managed_buckets, self.bucket_capacities):

    #         excess = len(bucket) - bucket_capacity

    #         if excess <= 0:
    #             continue

    #         # Oldest transitions to remove.
    #         remove_ids = [
    #             transition.insert_id
    #             for transition in sorted(
    #                 bucket,
    #                 key=lambda transition: transition.insert_id,
    #             )[:excess]
    #         ]

    #         # Rebuild bucket without removed transitions.
    #         filtered_bucket = deque(
    #             (
    #                 transition
    #                 for transition in bucket
    #                 if transition.insert_id not in remove_ids
    #             )
    #         )

    #         bucket.clear()
    #         bucket.extend(filtered_bucket)


    # def save(
    #     self,
    #     save_directory: str | Path,
    #     chunk_size: int = 5000,
    # ) -> None:
    #     """
    #     Saves the stratified replay memory into a directory.

    #     Directory structure
    #     -------------------
    #     replay_memory/
    #         metadata.pt
    #         low_000.pt
    #         low_001.pt
    #         ...
    #         medium_000.pt
    #         ...
    #         high_000.pt
    #         death_000.pt
    #     """
    #     save_directory = ensure_directory(save_directory)

    #     metadata = {
    #         "td_mean": self.td_mean,
    #         "td_std": self.td_std,
    #         "next_insert_id": self.next_insert_id,
    #         "bucket_capacities": self.bucket_capacities,
    #     }

    #     torch.save(metadata, (save_directory / "metadata.pt"))

    #     self._save_bucket(
    #         self._low_bucket,
    #         "low",
    #         save_directory,
    #         chunk_size,
    #     )

    #     self._save_bucket(
    #         self._medium_bucket,
    #         "medium",
    #         save_directory,
    #         chunk_size,
    #     )

    #     self._save_bucket(
    #         self._high_bucket,
    #         "high",
    #         save_directory,
    #         chunk_size,
    #     )

    #     self._save_bucket(
    #         self._death_bucket,
    #         "death",
    #         save_directory,
    #         chunk_size,
    #     )


    # def load(
    #     self,
    #     save_directory: str | Path,
    #     use_checkpoint_config: bool = True,
    # ) -> None:
    #     """
    #     Loads a previously saved stratified replay memory.
    #     """

    #     save_directory = Path(save_directory)

    #     if not save_directory.exists():
    #         raise FileNotFoundError(
    #             f"Replay memory directory does not exist: {save_directory}"
    #         )

    #     # Metadata
    #     metadata = torch.load(
    #         save_directory / "metadata.pt",
    #         map_location="cpu",
    #         weights_only=False,
    #     )

    #     self.td_mean = metadata["td_mean"]
    #     self.td_std = metadata["td_std"]
    #     self.next_insert_id = metadata["next_insert_id"]

    #     if use_checkpoint_config: # Use the saved bucket capacities from the checkpoint.
    #         self.bucket_capacities = metadata["bucket_capacities"]

    #     # Calculate last TD boundries.
    #     self.low_boundary = self.td_mean - self.td_std * self.td_std_multiplier
    #     self.high_boundary = self.td_mean + self.td_std * self.td_std_multiplier

    #     # It is not actually a first turn
    #     self.first_turn = False

    #     # Clear existing replay
    #     self._low_bucket.clear()
    #     self._medium_bucket.clear()
    #     self._high_bucket.clear()
    #     self._death_bucket.clear()

    #     self._new_bucket.clear()
    #     self._warmup_bucket.clear()

    #     # Load buckets
    #     self._load_bucket(self._low_bucket, "low", save_directory)
    #     self._load_bucket(self._medium_bucket, "medium", save_directory)
    #     self._load_bucket(self._high_bucket, "high", save_directory)
    #     self._load_bucket(self._death_bucket, "death", save_directory)

    #     print(
    #         f"Loaded replay memory:"
    #         f"\n  Low:    {len(self._low_bucket)}"
    #         f"\n  Medium: {len(self._medium_bucket)}"
    #         f"\n  High:   {len(self._high_bucket)}"
    #         f"\n  Death:  {len(self._death_bucket)}"
    #     )


    # def _save_bucket(
    #     self,
    #     bucket: deque[ReplayMemoryUnit],
    #     bucket_name: str,
    #     save_directory: Path,
    #     chunk_size: int = 5000,
    #     ) -> None:
    #     """
    #     Saves one replay bucket into multiple chunk files.

    #     Example:
    #         low_000.pt
    #         low_001.pt
    #         ...
    #     """

    #     bucket = list(bucket)

    #     for chunk_index, start in enumerate(range(0, len(bucket), chunk_size)):

    #         chunk = bucket[start : start + chunk_size]

    #         torch.save(
    #             chunk,
    #             save_directory / f"{bucket_name}_{chunk_index:03d}.pt",
    #         )

    # def _load_bucket(
    #     self, bucket: deque, 
    #     prefix: str, 
    #     save_directory: str | Path
    #     ):
    #     """
    #     Rebuilds a bucket by loading and merging multiple chunks.
    #     """
    #     files = sorted(save_directory.glob(f"{prefix}_*.pt"))

    #     for file in files:
    #         try:
    #             chunk = torch.load(
    #                 file,
    #                 map_location="cpu",
    #                 weights_only=False,
    #             )
    #         except Exception as e:
    #             print(f"Error loading {file}: {e}")
            
    #         # This is a sanity check to ensure that the loaded transitions have the correct bucket type.
    #         for transition in chunk: 
    #             if transition.bucket != ReplayBucketType[prefix.upper()]:                
    #                 # print(f"Warning: Transition bucket mismatch in {file}. Expected {prefix.upper()}, got {transition.bucket}.")
    #                 transition.bucket = ReplayBucketType[prefix.upper()]

    #         bucket.extend(chunk)
