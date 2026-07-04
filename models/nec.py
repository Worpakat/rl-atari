import torch
import torch.nn as nn

from models.dnd import DND
from models.seq_enc import NECEncoder, EncoderOutput
# from models.memory_update import MemoryUpdateStrategy

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

        return self.encoder(
            frames,
            random_sampling=random_sampling,
        )