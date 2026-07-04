import torch
import torch.nn as nn


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor, random_sampling: bool) -> torch.Tensor:
    """
    Reparameterization trick.

    During training:
        sample from N(mean, var)

    During evaluation:
        return mean
    """

    if not random_sampling:
        return mean

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mean + eps * std

class ConvBlock(nn.Module):
    """
    Conv2D -> Norm (optional) -> Activation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_norm: bool = True,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]

        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    ConvTranspose2D -> Norm (optional) -> Activation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        use_norm: bool = True,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)

        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        ]

        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LinearBlock(nn.Module):
    """
    Linear -> Norm (optional) -> Activation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_norm: bool = True,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)

        layers = [
            nn.Linear(in_features, out_features)
        ]

        if use_norm:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrameEncoder(nn.Module):
    """
    Encodes each frame independently into a compact feature vector.

    Input:
        (batch_size, sequence_length, channels, height, width)

    Output:
        (batch_size, sequence_length, conv_dim)
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        input_height: int,
        input_width: int,
        conv_channels: int = 256,
        conv_block: int = 4,
        conv_dim: int = 2048,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.conv_channels = conv_channels
        self.conv_dim = conv_dim

        self.conv_layers = nn.Sequential(
            ConvBlock(input_channels, conv_channels, kernel_size=5, stride=1, padding=2),
        )
            
        for _ in range(conv_block - 1):   
            self.conv_layers.append(ConvBlock(conv_channels, conv_channels, kernel_size=5, stride=2, padding=2))

        with torch.no_grad(): # We run this to get the `flattened_dim`.
            dummy = torch.zeros(1, input_channels, input_height, input_width)

            dummy = self.conv_layers(dummy)

            self.feature_height = dummy.shape[-2]
            self.feature_width = dummy.shape[-1]

            flattened_dim = (conv_channels * self.feature_height * self.feature_width)

        self.projection = nn.Sequential(
            LinearBlock(flattened_dim, conv_dim * 2),
            LinearBlock(conv_dim * 2, conv_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.shape

        if seq_len != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {seq_len}."
            )

        x = x.reshape(batch_size * seq_len, channels, height, width)

        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.projection(x)
        x = x.reshape(batch_size, seq_len, self.conv_dim)

        return x
    
class ContentEncoder(nn.Module):
    """
    Estimates the posterior distribution q(f | x_1, ..., x_T).

    Input:
        frame_features:
            (batch_size, sequence_length, conv_dim)

    Output:
        mean:
            (batch_size, content_dim)

        logvar:
            (batch_size, content_dim)

        content:
            (batch_size, content_dim)
    """

    def __init__(self, conv_dim: int, hidden_dim: int, content_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.content_dim = content_dim

        self.bi_lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.mean_head = nn.Linear(hidden_dim * 2, content_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, content_dim)

    def forward(self, frame_features: torch.Tensor, random_sampling: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lstm_out, _ = self.bi_lstm(frame_features)

        forward_final = lstm_out[:, -1, :self.hidden_dim]
        backward_final = lstm_out[:, 0, self.hidden_dim:]

        sequence_summary = torch.cat([forward_final, backward_final], dim=1)

        mean = self.mean_head(sequence_summary)
        logvar = self.logvar_head(sequence_summary)

        content = reparameterize(mean, logvar, random_sampling)

        return mean, logvar, content
    
class DynamicsEncoder(nn.Module):
    """
    Estimates the posterior distribution q(z_1, ..., z_T | x_1, ..., x_T, f).

    Input:
        frame_features:
            (batch_size, sequence_length, conv_dim)

        content:
            (batch_size, content_dim)

    Output:
        mean:
            (batch_size, sequence_length, dynamics_dim)

        logvar:
            (batch_size, sequence_length, dynamics_dim)

        dynamics:
            (batch_size, sequence_length, dynamics_dim)
    """

    def __init__(
        self,
        sequence_length: int,
        conv_dim: int,
        content_dim: int,
        hidden_dim: int,
        dynamics_dim: int,
        lstm_layers: int = 1,
        rnn_layers: int = 1,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.content_dim = content_dim

        self.bi_lstm = nn.LSTM(
            input_size=conv_dim + content_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.rnn = nn.RNN(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.mean_head = nn.Linear(hidden_dim, dynamics_dim)
        self.logvar_head = nn.Linear(hidden_dim, dynamics_dim)

    def forward(
        self,
        frame_features: torch.Tensor,
        content: torch.Tensor,
        random_sampling: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        content = content.unsqueeze(1).expand(-1, self.sequence_length, -1)

        features = torch.cat([frame_features, content], dim=-1)
        features, _ = self.bi_lstm(features)
        features, _ = self.rnn(features)

        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        dynamics = reparameterize(mean, logvar, random_sampling)

        return mean, logvar, dynamics

class DynamicsPrior(nn.Module):
    """
    Autoregressive Gaussian prior over the dynamics latent sequence.

    Generates the prior distributions:

        p(z_1), ..., p(z_T)

    using an LSTMCell.

    Output:
        mean:
            (batch_size, sequence_length, dynamics_dim)

        logvar:
            (batch_size, sequence_length, dynamics_dim)

        samples:
            (batch_size, sequence_length, dynamics_dim)
    """

    def __init__(self, sequence_length: int, dynamics_dim: int, hidden_dim: int):
        super().__init__()

        self.sequence_length = sequence_length
        self.dynamics_dim = dynamics_dim
        self.hidden_dim = hidden_dim

        self.lstm_cell = nn.LSTMCell(input_size=dynamics_dim, hidden_size=hidden_dim)

        self.mean_head = nn.Linear(hidden_dim, dynamics_dim)
        self.logvar_head = nn.Linear(hidden_dim, dynamics_dim)

    def forward(self, batch_size: int, device: torch.device, random_sampling: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z = torch.zeros(batch_size, self.dynamics_dim, device=device)
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        means = []
        logvars = []
        samples = []

        for _ in range(self.sequence_length):

            h, c = self.lstm_cell(z, (h, c))

            mean = self.mean_head(h)
            logvar = self.logvar_head(h)

            z = reparameterize(mean, logvar, random_sampling)

            means.append(mean)
            logvars.append(logvar)
            samples.append(z)

        means = torch.stack(means, dim=1)
        logvars = torch.stack(logvars, dim=1)
        samples = torch.stack(samples, dim=1)

        return means, logvars, samples

class FrameDecoder(nn.Module):
    """
    Reconstructs an image sequence from the content and dynamics latents.

    Input:
        content:
            (batch_size, content_dim)

        dynamics:
            (batch_size, sequence_length, dynamics_dim)

    Output:
        reconstruction:
            (batch_size, sequence_length,
             input_channels, input_height, input_width)
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        input_height: int,
        input_width: int,
        content_dim: int,
        dynamics_dim: int,
        feature_height: int,
        feature_width: int,
        conv_channels: int = 256,
        conv_dim: int = 2048,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.conv_channels = conv_channels
        self.feature_height = feature_height
        self.feature_width = feature_width

        flattened_dim = (conv_channels * self.feature_height * self.feature_width)

        self.projection = nn.Sequential(
            LinearBlock(content_dim + dynamics_dim, conv_dim * 2, use_norm=False),
            LinearBlock(conv_dim * 2, flattened_dim, use_norm=False),
        )

        self.deconv_layers = nn.Sequential(

            DeconvBlock(
                conv_channels,
                conv_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),

            DeconvBlock(
                conv_channels,
                conv_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),

            DeconvBlock(
                conv_channels,
                conv_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),

            DeconvBlock(
                conv_channels,
                input_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                use_norm=False,
                activation=nn.Tanh(),
            ),
        )

    def forward(
        self,
        content: torch.Tensor,
        dynamics: torch.Tensor,
    ) -> torch.Tensor:

        content = content.unsqueeze(1).expand(-1, self.sequence_length, -1)
        
        latent = torch.cat([content, dynamics], dim=-1)
        batch_size = latent.shape[0]

        latent = latent.reshape(batch_size * self.sequence_length, -1)

        x = self.projection(latent)
    
        x = x.reshape(
            batch_size * self.sequence_length,
            self.conv_channels,
            self.feature_height,
            self.feature_width,
        )

        x = self.deconv_layers(x)

        x = x.reshape(
            batch_size,
            self.sequence_length,
            self.input_channels,
            self.input_height,
            self.input_width,
        )

        return x
    

class DisentangledVAE(nn.Module):
    """
    Disentangled Sequential Variational Autoencoder.

    Produces:
        - content latent
        - dynamics latent sequence
        - dynamics prior
        - reconstructed frame sequence
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        input_height: int,
        input_width: int,
        content_dim: int = 256,
        dynamics_dim: int = 32,
        conv_channels: int = 256,
        conv_dim: int = 2048,
        hidden_dim: int = 512,
        lstm_layers: int = 1,
        rnn_layers: int = 1,
    ):
        super().__init__()

        self.frame_encoder = FrameEncoder(
            sequence_length=sequence_length,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            conv_channels=conv_channels,
            conv_dim=conv_dim,
        )

        self.content_encoder = ContentEncoder(
            conv_dim=conv_dim,
            hidden_dim=hidden_dim,
            content_dim=content_dim,
        )

        self.dynamics_encoder = DynamicsEncoder(
            sequence_length=sequence_length,
            conv_dim=conv_dim,
            content_dim=content_dim,
            hidden_dim=hidden_dim,
            dynamics_dim=dynamics_dim,
            lstm_layers=lstm_layers,
            rnn_layers=rnn_layers,
        )

        self.dynamics_prior = DynamicsPrior(
            sequence_length=sequence_length,
            dynamics_dim=dynamics_dim,
            hidden_dim=hidden_dim,
        )

        self.frame_decoder = FrameDecoder(
            sequence_length=sequence_length,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            content_dim=content_dim,
            dynamics_dim=dynamics_dim,
            feature_height=self.frame_encoder.feature_height,
            feature_width=self.frame_encoder.feature_width,
            conv_channels=conv_channels,
            conv_dim=conv_dim,
        )

    def forward(self, frames: torch.Tensor, random_sampling: bool = True):

        batch_size = frames.size(0)

        prior_mean, prior_logvar, _ = self.dynamics_prior(
            batch_size=batch_size,
            device=frames.device,
            random_sampling=random_sampling,
        )

        frame_features = self.frame_encoder(frames)

        content_mean, content_logvar, content = self.content_encoder(
            frame_features,
            random_sampling=random_sampling,
        )

        dynamics_mean, dynamics_logvar, dynamics = self.dynamics_encoder(
            frame_features,
            content,
            random_sampling=random_sampling,
        )

        reconstruction = self.frame_decoder(
            content,
            dynamics
        )
    
        return {
            "content_mean": content_mean,
            "content_logvar": content_logvar,
            "content": content,
            "dynamics_mean": dynamics_mean,
            "dynamics_logvar": dynamics_logvar,
            "dynamics": dynamics,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "reconstruction": reconstruction,
        }
    
    