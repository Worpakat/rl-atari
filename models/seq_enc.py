from dataclasses import dataclass

import torch
import torch.nn as nn

from models.dsae import (LinearBlock, reparameterize,
                         ConvBlock,
                         FrameEncoder,
                         DynamicsPrior)

@dataclass
class EncoderOutput:
    representation: torch.Tensor
    posterior_mean: torch.Tensor | None = None
    posterior_logvar: torch.Tensor | None = None
    prior_mean: torch.Tensor | None = None
    prior_logvar: torch.Tensor | None = None
    frame_features: torch.Tensor | None = None

class SequenceEncoder(nn.Module):
    """
    Estimates the posterior distribution

        q(z_1, ..., z_T | x_1, ..., x_T)

    from a sequence of frame feature vectors.

    Input:
        frame_features:
            (batch_size, sequence_length, conv_dim)

    Output:
        mean:
            (batch_size, sequence_length, latent_dim)

        logvar:
            (batch_size, sequence_length, latent_dim)

        latent:
            (batch_size, sequence_length, latent_dim)
    """

    def __init__(
        self,
        conv_dim: int,
        hidden_dim: int,
        latent_dim: int,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.bi_lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.mean_head = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(
        self,
        frame_features: torch.Tensor,
        random_sampling: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes a sequence of frame features into a sequence of
        stochastic latent representations.
        """

        features, _ = self.bi_lstm(frame_features)

        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        latent = reparameterize(mean, logvar, random_sampling)

        return mean, logvar, latent
    

class SequencePrior(nn.Module):
    """
    Autoregressive Gaussian prior over the latent sequence.

    This module is a thin wrapper around the DSAE DynamicsPrior
    implementation. The underlying computation is identical, but the
    naming is adapted to the Sequential NEC architecture.

    Keeping this wrapper provides two advantages:

    1. The architecture remains self-contained and independent of
       DSAE-specific terminology.

    2. The prior implementation can be replaced or extended in the
       future without changing the remaining encoder code.
    """

    def __init__(
        self,
        sequence_length: int,
        latent_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.prior = DynamicsPrior(
            sequence_length=sequence_length,
            dynamics_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        batch_size: int,
        device: torch.device,
        random_sampling: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.prior(
            batch_size=batch_size,
            device=device,
            random_sampling=random_sampling,
        )
    

class Adapter(nn.Module):
    """
    Optional linear adapter layer to transform the flattened sequential encoder output 
    to a suitable representation size. To use as DND keys.

    latent_dim: Latent dimension
    representation_dim: Final representation dimension, key dimension.
    """
    def __init__(
            self, 
            latent_dim: int,
            sequence_length: int, 
            representation_dim: int):
        super().__init__()

        in_features = sequence_length * latent_dim

        self.linear = nn.Sequential(
            LinearBlock(in_features=in_features, 
                        out_features=int(in_features/2),
                        use_norm=False),

            LinearBlock(in_features=int(in_features/2), 
                        out_features=representation_dim,
                        use_norm=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Adapter input shape", x.shape)
        return self.linear(x)


class SequentialEncoder(nn.Module):
    """
    Sequential latent encoder used by the Sequential NEC architecture.

    The model consists of

        FrameEncoder
            ↓
        SequenceEncoder
            ↓
        SequencePrior

    Depending on ``flatten_output``, the posterior latent sequence is
    either returned as a sequence or flattened into a single feature
    vector suitable for downstream modules such as a projection head or
    DND.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        input_height: int,
        input_width: int,
        latent_dim: int,
        conv_channels: int = 256,
        conv_block: int = 4,
        conv_dim: int = 2048,
        hidden_dim: int = 512,
        lstm_layers: int = 1,
        flatten_output: bool = False,
        adapter: bool = False,
        representation_dim: int | None = None,
    ):
        super().__init__()

        self.flatten_output = flatten_output

        self.frame_encoder = FrameEncoder(
            sequence_length=sequence_length,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            conv_channels=conv_channels,
            conv_block=conv_block,
            conv_dim=conv_dim,
        )

        self.sequence_encoder = SequenceEncoder(
            conv_dim=conv_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lstm_layers=lstm_layers,
        )

        self.sequence_prior = SequencePrior(
            sequence_length=sequence_length,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        if adapter:
            self.adapter = Adapter(latent_dim=latent_dim, representation_dim=representation_dim)
        else:
            self.adapter = nn.Identity()

    def forward(self, frames: torch.Tensor, random_sampling: bool = True) -> EncoderOutput:
        """
        Encodes a frame sequence into posterior and prior latent
        distributions.
        """

        frame_features = self.frame_encoder(frames)

        posterior_mean, posterior_logvar, posterior_latents = (
            self.sequence_encoder(
                frame_features,
                random_sampling=random_sampling,
            )
        )

        prior_mean, prior_logvar, _ = self.sequence_prior(
            batch_size=frames.size(0),
            device=frames.device,
            random_sampling=random_sampling,
        )

        representation = posterior_latents

        print("Representation shape before", representation.shape)

        if self.flatten_output:
            representation = representation.flatten(start_dim=1)

            print("Representation shape after", representation.shape)


        representation = self.adapter(representation)

        return EncoderOutput(
            representation=representation,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )
    

class NECEncoder(nn.Module):
    """
    This class exists for naming convenience.
    """
    def __init__(self, encoder: SequentialEncoder, **kwargs):
        super().__init__()
        self.encoder = encoder

    def forward(self, frames: torch.Tensor, random_sampling: bool = True) -> EncoderOutput:
        return self.encoder(frames, random_sampling=random_sampling)