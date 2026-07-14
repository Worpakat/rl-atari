import random

import torch
import torch.nn as nn

from utils.misc import discount
from utils.data_buffers import Transition, TransitionQueue
from models.dnd import DND, LookupResult
from models.seq_enc import NECEncoder, EncoderOutput
from models.memory_update import (MemoryUpdateRequest,
                                  MemoryUpdateStrategy, 
                                  OriginalNECUpdateStrategy)

class NECAgent(nn.Module):
    """
    Neural Episodic Control agent with sequential state encoding.

    Owns the trainable encoder together with the differentiable neural
    dictionary (DND) and the memory update strategy. The trainer
    interacts only with this class during reinforcement learning.
    """

    def __init__(
        self,
        encoder: NECEncoder,
        dnds: list[DND],
        update_strategy: MemoryUpdateStrategy,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float
    ):
        super().__init__()

        self.encoder = encoder
        self.dnds = dnds

        self.update_strategy = update_strategy
        self.original_update_strategy = OriginalNECUpdateStrategy(self.update_strategy.learning_rate)
        # ! This one used for 'warmup' phase update and inserts.

        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Buffer used due to save as model state. 
        self.register_buffer("current_epsilon", torch.tensor(epsilon_start, dtype=torch.float32)),
    
    
    def encode(
        self,
        frames: torch.Tensor,
        random_sampling: bool = True,
    ) -> EncoderOutput:
        """
        Encodes a sequence of frames into its state representation.

        Returns the state representation together with its optional
        auxiliary representation.
        """

        return self.encoder(frames, random_sampling=random_sampling)
    
    def lookup_to_dnd(
        self,
        action: int,
        key: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
        return_indices: bool = False,
        return_similarities: bool = False,
        return_neighbors: bool = False,
        track_key_updates: bool = False,
    ) -> LookupResult:
        """
        
        """

        return self.dnds[action].lookup(
            key=key,
            auxiliary=auxiliary,
            return_indices=return_indices,
            return_similarities=return_similarities,
            return_neighbors=return_neighbors,
            track_key_updates=track_key_updates
        )
    
    def lookup(
        self,
        key: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
        return_indices: bool = False,
        return_similarities: bool = False,
        return_neighbors: bool = False,
    ) -> list[LookupResult]:
        """
        Looks up the estimated action values for a state representation.

        Performs one lookup in each action-specific DND. Optional lookup
        information is returned only when explicitly requested in order to
        avoid unnecessary memory allocations during training.

        Args:
            key:
                State representation produced by the encoder.

            auxiliary:
                Optional auxiliary representation used by the selected
                similarity function.

            return_indices:
                Whether to return the indices of retrieved neighbors.

            return_similarities:
                Whether to return similarity scores of retrieved neighbors.

            return_neighbors:
                Whether to return neighbor representations and values.

        Returns:
            Lookup results for every action.
        """

        return [
            dnd.lookup(
                key=key,
                auxiliary=auxiliary,
                return_indices=return_indices,
                return_similarities=return_similarities,
                return_neighbors=return_neighbors,
            )
            for dnd in self.dnds
        ]
    

    def decay_epsilon(self) -> float:
        """
        Decays current epsilon value by a given factor.
        """
        if self.current_epsilon > self.epsilon_end:
            self.current_epsilon = self.current_epsilon * self.epsilon_decay


    def choose_action(
        self,
        encoder_output: EncoderOutput,
        exploration: bool = True
    ) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        Parameters
        ----------
        encoder_output:
            Encoded representation of the current state.

        epsilon:
            Exploration probability.

        Returns
        -------
        (int, bool)
            Selected action and whether it was an exploration action.
        """

        # Exploration.
        if exploration and random.random() < self.current_epsilon.item():
            return (random.randrange(len(self.dnds)), True)

        # Exploitation.
        results = self.lookup(
            key=encoder_output.representation,
            auxiliary=encoder_output.auxiliary,
        )
        q_values = torch.stack([result.q_value for result in results])
        
        return (int(torch.argmax(q_values).item()), False)

    def compute_q_targets(
        self,
        transition_queue: TransitionQueue,
        gamma: float,
        n_step: int,
        warmup: bool = False,
    ) -> torch.Tensor:
        """
        Computes N-step Q targets for every transition in the trajectory.
        """

        transition_count = len(transition_queue)
        rewards = torch.tensor(
            [transition.reward for transition in transition_queue],
            dtype=torch.float32,
        )

        # Discounted returns beginning from every timestep.
        discounted_returns = torch.from_numpy(discount(rewards.numpy(), gamma)).to(torch.float32)
        # !! Dtype is converted to float64 in discount(). We need to convert it back.

        q_targets = torch.empty_like(discounted_returns)

        for transition_index in range(transition_count):
            # Warmup or insufficient future transitions.
            
            if (warmup or transition_index + n_step >= transition_count):
                q_targets[transition_index] = discounted_returns[transition_index]

                continue

            # N-step discounted reward.
            discounted_reward = (
                discounted_returns[transition_index]
                - (gamma ** n_step) * discounted_returns[transition_index + n_step]
            )

            # Bootstrap estimate.
            bootstrap_transition = transition_queue[transition_index + n_step]
            
            if bootstrap_transition.representation is None:
                bootstrap_transition.representation = self.encode(
                    frames=torch.from_numpy(bootstrap_transition.state).unsqueeze(0).to(self.encoder.device),
                    random_sampling=False, # We use 'posterior_mean's as representations for stability
                ).representation.detach().cpu()
            
            lookup_results = self.lookup(key=bootstrap_transition.representation)

            bootstrap_value = torch.stack(
                [result.value for result in lookup_results]
            ).max()

            q_targets[transition_index] = (discounted_reward + (gamma ** n_step) * bootstrap_value)

        return q_targets

    def contains(self ,key: torch.Tensor, action: int) -> bool:
        """
        Returns whether the given key already exists in committed memory.
        """
        return self.dnds[action].contains(key)

    def create_memory_update_request(
        self,
        transition: Transition,
        q_target: torch.Tensor,
        lookup_result: LookupResult|None,
        update_or_insert: str = 'insert',
        warmup: bool = False,
        exploration_update: bool = False,
        ) -> MemoryUpdateRequest | list[MemoryUpdateRequest]:
        """
        Makes self.update_strategy calculate update values for given transition and returns 
        MemoryUpdateRequest or list of MemoryUpdateRequest objects.
        """
        
        if lookup_result is None: ### !!! CARRY THIS PART to UPDATE STRATEGY CLASS TOO !!! ###
            dnd = self.dnds[transition.action]
            
            #----------------Warmup_Phase_Insert----------------------------
            if warmup and update_or_insert == 'insert': 

                return self.original_update_strategy.calculate_memory_update_request(
                                                dnd=dnd,
                                                transition=transition,
                                                q_target=q_target,)

            #-----------------------------------------------------------------
            
            # No insert required, update given keys value with target by original bellman equation.
            index = dnd.get_index(transition.representation)
            current_value = dnd.get_value(index)
            
            update_value = self.update_strategy.calculate_bellman_update_change(current_value, q_target)
            
            return MemoryUpdateRequest(
                update_or_insert='update',
                action=transition.action,
                index=index,
                key=transition.representation,
                is_change=True,
                update_value=update_value,
            )

        else:
            # Insert required, update given keys value with target by original bellman equation.
            dnd = self.dnds[transition.action]
            
            return self.update_strategy.calculate_memory_update_request(
                                            dnd=dnd,
                                            transition=transition,
                                            q_target=q_target,
                                            lookup_result=lookup_result,
                                            exploration_update=exploration_update,)
                        

    def apply_memory_updates(
        self,
        update_requests: list[MemoryUpdateRequest],
    ) -> None:
        """
        Applies memory update requests using the configured update strategy.

        The update strategy is responsible for interpreting each request and
        modifying the corresponding action-specific DND.

        Args:
            update_requests:
                Collection of memory update requests generated by the trainer.
        """

        self.update_strategy.apply(dnds=self.dnds, update_requests=update_requests)

    ## ==========Batch_Update_Methods===========

    def lookup_batch(
        self,
        representations: torch.Tensor,
        actions: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
        track_key_updates: bool = False,
    ) -> torch.Tensor:
        """
        Performs DND lookups for a mini-batch.

        Optionally prepares the DNDs for trainable key optimization and
        records the memory entries participating in the lookups.

        Returns:
            Tensor of predicted Q-values with shape (batch_size,).
        """

        if track_key_updates:

            for dnd in self.dnds:
                dnd.initialize_key_optimizer()

        predictions = []

        for batch_index, (representation, action) in enumerate(zip(representations, actions)):

            lookup_result = self.lookup_to_dnd(
                action=action,
                key=representation,
                auxiliary=None if auxiliary is None else auxiliary[batch_index],
                track_key_updates=track_key_updates,
            )

            predictions.append(lookup_result.value) 

        return torch.stack(predictions)

    def zero_key_gradients(self) -> None:
        """
        Clears gradients of all trainable DND key optimizers.
        """

        for dnd in self.dnds:

            if dnd.key_optimizer is not None:
                dnd.key_optimizer.zero_grad()

    def step_key_optimizers(self) -> None:
        """
        Updates all trainable DND keys.
        """

        for dnd in self.dnds:

            if dnd.key_optimizer is not None:
                dnd.key_optimizer.step()
                dnd.build_index()
    
    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the NEC agent.
        """
        return {
            "encoder": self.encoder.state_dict(),
            "dnds": [dnd.state_dict() for dnd in self.dnds],
        }


