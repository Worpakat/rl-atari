from pathlib import Path
import argparse

import torch

from utils.training_config import TrainingConfig
from models.dsae import DisentangledVAE
from training.dsae_trainer import DSAETrainer



def train(config: TrainingConfig):
    """
    Runs one DSAE warm-up experiment.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DisentangledVAE(
        sequence_length=config.sequence_length,
        input_channels=config.input_channels,
        input_height=config.input_height,
        input_width=config.input_width,
        content_dim=config.content_dim,
        dynamics_dim=config.dynamics_dim,
        conv_channels=config.conv_channels,
        conv_dim=config.conv_dim,
        hidden_dim=config.hidden_dim,
        lstm_layers=config.lstm_layers,
        rnn_layers=config.rnn_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        weight_decay=config.weight_decay,
    )

    trainer = DSAETrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        experiment_dir=config.experiment_dir,
        device=device,
    )

    return trainer.train()


def main():
    """
    Entry point for terminal execution.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training configuration JSON.",
    )

    args = parser.parse_args()

    config = TrainingConfig.load(args.config)

    train(config)


if __name__ == "__main__":
    main()


#===EXAMPLE-FOR-TERMINAL-SCRIPT-EXECUTION===
# >>> python scripts/train_phase_0.py --config configs/pong.json