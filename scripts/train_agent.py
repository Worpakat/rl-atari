from pathlib import Path
import argparse

import torch

from models.dnd import DND
from models.memory_update import Option1UpdateStrategy, Option2UpdateStrategy, OriginalNECUpdateStrategy
from models.seq_enc import NECEncoder, SequentialEncoder
from models.similarity_functions import exponential_inverse, inverse_distance
from training.nec_trainer import NECTrainer
from utils.neighbor_index import FaissIndex
from utils.training_config import TrainingConfig
from models.nec import NECAgent

def train(config: TrainingConfig):
    """
    Runs one DSAE warm-up experiment.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize NEC Encoder
    nec_encoder = NECEncoder(
        encoder = SequentialEncoder(
            sequence_length = config.sequence_length,
            input_channels = config.input_channels,
            input_height = config.input_height,
            input_width = config.input_width,
            latent_dim = config.content_dim,
            conv_channels = config.conv_channels,
            conv_block = config.conv_block,
            conv_dim = config.conv_dim,
            hidden_dim = config.hidden_dim,
            lstm_layers = config.lstm_layers,
            flatten_output = config.flatten_output,
        )
    )

    # Initialize Similarity Function
    similarity_function = None
    if config.similarity_function == "exponential_inverse":
        similarity_function = exponential_inverse
    
    elif config.similarity_function == "inverse_distance":
        similarity_function = inverse_distance

    else:
        raise ValueError(f"Unknown similarity function: {config.similarity_function}")

    # Initialize DNDs
    dnds = [
            DND(
                representation_dim = config.representation_dim,
                value_dim = config.dnd_value_dim if config.dnd_value_dim is not None else 1, 
                use_auxiliary= config.use_auxiliary if config.use_auxiliary is not None else False,
                auxiliary_dim = config.auxiliary_dim if config.auxiliary_dim is not None else 1, 
                max_memory = config.dnd_max_memory,
                num_neighbors = config.dnd_num_neighbors,
                neighbor_index = FaissIndex(config.representation_dim, config.dnd_duplicate_threshold),
                similarity_function = similarity_function,
                learning_rate = config.dnd_learning_rate,
                device = device
            ) 
            for _ in range(config.num_actions)]

    # Initialize Update Strategy
    if config.update_strategy == "nec_original":
        update_strategy = OriginalNECUpdateStrategy(learning_rate = config.dnd_q_learning_rate)
    
    elif config.update_strategy == "option_1":
        update_strategy = Option1UpdateStrategy(
                                learning_rate = config.dnd_q_learning_rate, 
                                exploration_lr = config.dnd_exploration_q_lr
                                )
    
    elif config.update_strategy == "option_2":
        update_strategy = Option2UpdateStrategy(
                                learning_rate = config.dnd_q_learning_rate, 
                                exploration_lr = config.dnd_exploration_q_lr,
                                neighbor_shrink = config.dnd_neighbor_shrink
                                )
        
    else:
        raise ValueError(f"Unknown update strategy: {config.update_strategy}")

    # Initialize NEC Agent
    agent = NECAgent(
        encoder = nec_encoder,
        dnds = dnds,
        update_strategy = update_strategy,
    ).to(device)

    # Initialize NEC Encoder Optimizer
    encoder_optimizer = torch.optim.Adam(
        nec_encoder.parameters(),
        lr = config.encoder_learning_rate,
    )

    # Initialize NEC Trainer
    trainer = NECTrainer(
        agent = agent,
        encoder_optimizer = encoder_optimizer,
        config = config,
        experiment_dir = config.experiment_dir,
        device = device
    )


    return trainer


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

    trainer = train(config)

    if config.warmup_steps > 0:
        trainer.warmup()
        
    trainer.train()


if __name__ == "__main__":
    main()


#===EXAMPLE-FOR-TERMINAL-SCRIPT-EXECUTION===
# >>> python scripts/train_phase_0.py --config configs/pong.json