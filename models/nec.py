import random

import torch
import torch.nn as nn

from models.dnd import DND, LookupResult
from models.seq_enc import NECEncoder, EncoderOutput
from models.memory_update import MemoryUpdateRequest, MemoryUpdateStrategy
from utils.data_buffers import Transition

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
    ):
        super().__init__()

        self.encoder = encoder
        self.dnds = dnds
        self.update_strategy = update_strategy

    
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
        representation: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
        return_indices: bool = False,
        return_similarities: bool = False,
        return_neighbors: bool = False,
    ):
        """
        
        """

        return self.dnds[action].lookup(
            key=representation,
            auxiliary=auxiliary,
            return_indices=return_indices,
            return_similarities=return_similarities,
            return_neighbors=return_neighbors,
        )
    
    def lookup(
        self,
        representation: torch.Tensor,
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
            representation:
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
                key=representation,
                auxiliary=auxiliary,
                return_indices=return_indices,
                return_similarities=return_similarities,
                return_neighbors=return_neighbors,
            )
            for dnd in self.dnds
        ]

    
    def choose_action(
        self,
        encoder_output: EncoderOutput,
        epsilon: float,
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
        int
            Selected action.
        """

        # Exploration.
        if random.random() < epsilon:
            return random.randrange(len(self.dnds))

        # Exploitation.
        results = self.lookup(
            representation=encoder_output.representation,
            auxiliary=encoder_output.auxiliary,
        )
        q_values = torch.stack([result.q_value for result in results])
        
        return int(torch.argmax(q_values).item())
    

    def create_memory_update_request(
        self,
        transition: Transition,
        q_target: torch.Tensor,
        lookup_result: LookupResult|None,
        exploration_update: bool = False,
        ) -> MemoryUpdateRequest | list[MemoryUpdateRequest]:
        """
        Makes self.update_strategy calculate update values for given transition and returns 
        MemoryUpdateRequest or list of MemoryUpdateRequest objects.
        """

        if lookup_result is None: 
            # No insert required, update given keys value with target by original bellman equation.
            dnd = self.dnds[transition.action]
            
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







            
    def get_dnd(
        self,
        action: int,
    ) -> DND:
        """
        Returns the DND corresponding to the given action.

        Args:
            action:
                Discrete action index.

        Returns:
            Action-specific differentiable neural dictionary.
        """

        return self.dnds[action]


