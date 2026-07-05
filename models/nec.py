import random

import torch
import torch.nn as nn

from models.dnd import DND, LookupResult
from models.seq_enc import NECEncoder, EncoderOutput
from models.memory_update import MemoryUpdateRequest, MemoryUpdateStrategy

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

        #
        # Exploration.
        #

        if random.random() < epsilon:
            return random.randrange(len(self.dnds))

        #
        # Exploitation.
        #

        results = self.lookup(
            representation=encoder_output.representation,
            auxiliary=encoder_output.auxiliary,
        )
        q_values = torch.stack([result.q_value for result in results])
        
        return int(torch.argmax(q_values).item())
    
    def forward(
        self,
        frames: torch.Tensor,
        random_sampling: bool = True,
        return_indices: bool = False,
        return_similarities: bool = False,
        return_neighbors: bool = False,
    ) -> tuple[list[LookupResult], EncoderOutput]:
        """
        Encodes a sequence of frames and estimates the action values.

        This is a convenience wrapper combining the encoder and the DND
        lookups. Besides the action value estimates, the complete encoder
        outputs are returned so that the trainer can compute any required
        auxiliary losses.

        Args:
            frames:
                Batch of frame sequences.

            random_sampling:
                Whether to sample from the posterior distributions during
                encoding.

            return_indices:
                Whether DND lookups should return neighbor indices.

            return_similarities:
                Whether DND lookups should return neighbor similarity scores.

            return_neighbors:
                Whether DND lookups should return neighbor information.

        Returns:
            lookup_results:
                Lookup result for each action.

            encoder_output:
                Complete encoder outputs.
        """

        encoder_output = self.encode(frames=frames, random_sampling=random_sampling)

        lookup_results = self.lookup(
            representation=encoder_output.representation,
            auxiliary=encoder_output.auxiliary,
            return_indices=return_indices,
            return_similarities=return_similarities,
            return_neighbors=return_neighbors,
        )

        return lookup_results, encoder_output
    
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

        self.update_strategy.apply(
            dnds=self.dnds,
            update_requests=update_requests,
        )