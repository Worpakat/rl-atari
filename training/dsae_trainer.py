from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
import ale_py
gym.register_envs(ale_py) # Explicitly register the Atari games to gym

from utils.training_config import TrainingConfig
from utils.metrics_logger import MetricsLogger
from utils.reconstruction_recorder import ReconstructionRecorder
from utils.checkpoint import CheckpointManager
from utils.data_buffers import FrameSequenceBuffer
from utils.data_buffers import SequenceReplayBuffer
from utils.misc import ensure_directory, preprocess_frame

from losses.dsae import reconstruction_loss, content_kl_loss, dynamics_kl_loss

class DSAETrainer:
    """
    Trainer for the DSAE warm-up phase.

    During this phase the representation model is trained solely with the
    reconstruction and KL losses. No reinforcement learning components are
    involved.
    """

    def __init__(
        self,
        model,
        optimizer,
        config: TrainingConfig,
        experiment_dir: str | Path,
        device: str | torch.device = "cpu",
        trial=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.trial = trial

        self.device = torch.device(device)
        self.model.to(self.device)

        self.experiment_dir = ensure_directory(experiment_dir)

        self.logger = MetricsLogger(self.experiment_dir)
        self.checkpoint_manager = CheckpointManager(self.experiment_dir)
        self.recorder = ReconstructionRecorder(self.experiment_dir, save_every=config.reconstruction_period)
        
        self.sequence_buffer = FrameSequenceBuffer(sequence_length=config.sequence_length)
        self.batch_buffer = SequenceReplayBuffer(capacity=config.batch_size)

        self.environment = gym.make(config.environment_name, render_mode=None)

        self.global_step = 0
        self.episode = 0
        self.optimization_step = 0

    def _setup(self):
        """
        Initializes the environment and frame sequence buffer.
        """

        observation, _ = self.environment.reset()

        observation = preprocess_frame(observation)

        self.sequence_buffer.clear()

        for _ in range(self.config.sequence_length):
            self.sequence_buffer.append(observation)

    def _finished(self) -> bool:
        """
        Returns whether DSAE training has finished.
        """

        if self.config.max_optimization_steps is not None:
            return self.optimization_step >= self.config.max_optimization_steps

        if self.config.max_environment_steps is not None:
            return self.global_step >= self.config.max_environment_steps

        if self.config.max_episodes is not None:
            return self.episode >= self.config.max_episodes

        raise ValueError(
            "At least one stopping criterion must be specified."
        )

    def _environment_step(self):
        """
        Executes one environment interaction and stores completed
        frame sequences in the replay buffer.
        """

        action = self.environment.action_space.sample()

        observation, _, terminated, truncated, _ = self.environment.step(action)

        observation = preprocess_frame(observation)

        self.sequence_buffer.append(observation)

        if self.sequence_buffer.is_ready():
            self.batch_buffer.append(self.sequence_buffer.get_sequence())

        self.global_step += 1

        if terminated or truncated:

            self.episode += 1

            observation, _ = self.environment.reset()

            observation = preprocess_frame(observation)

            self.sequence_buffer.clear()

            for _ in range(self.config.sequence_length):
                self.sequence_buffer.append(observation)

    def _optimization_step(self):
        """
        Performs one DSAE optimization step if enough sequences have
        been collected.
        """

        if self.batch_buffer.can_sample(self.config.batch_size) == False:
            return

        print(f"Optimization step {self.optimization_step}: Optimizing...")

        batch = self.batch_buffer.sample(self.config.batch_size)

        frames = batch["frames"].to(self.device, non_blocking=True)

        (
            content_mean,
            content_logvar,
            _,
            dynamics_mean,
            dynamics_logvar,
            _,
            dynamics_prior_mean,
            dynamics_prior_logvar,
            reconstruction,
        ) = self.model(frames)

        print("Reconstruction", reconstruction.shape)
        print("Frames", frames.shape)

        loss_reconstruction = reconstruction_loss(input_frames=frames, reconstructed_frames=reconstruction)

        loss_content_kl = content_kl_loss(content_mean, content_logvar)

        loss_dynamics_kl = dynamics_kl_loss(
            dynamics_mean,
            dynamics_logvar,
            dynamics_prior_mean,
            dynamics_prior_logvar,
        )

        total_loss = (
            self.config.reconstruction_weight * loss_reconstruction
            + self.config.content_kl_weight * loss_content_kl
            + self.config.dynamics_kl_weight * loss_dynamics_kl
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        
        self.optimization_step += 1
        self.batch_buffer.clear()
        
        return {
            "total_loss": total_loss.item(),
            "reconstruction_loss": loss_reconstruction.item(),
            "content_kl_loss": loss_content_kl.item(),
            "dynamics_kl_loss": loss_dynamics_kl.item(),
            "frames": frames,
            "reconstruction": reconstruction,
        }

    def _logging_step(self, logs: dict | None):
        """
        Records training metrics.
        """

        self.logger.log(
            optimization_step=self.optimization_step,
            environment_step=self.global_step,
            episode=self.episode,
            total_loss=logs["total_loss"],
            reconstruction_loss=logs["reconstruction_loss"],
            content_kl_loss=logs["content_kl_loss"],
            dynamics_kl_loss=logs["dynamics_kl_loss"],
        )

    def _checkpoint_step(self, logs: dict):
        """
        Saves a training checkpoint periodically.
        """

        if self.optimization_step % self.config.checkpoint_period != 0:
            return

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_state": {
                "optimization_step": self.optimization_step,
                "environment_step": self.global_step,
                "episode": self.episode,
            },
            "metrics": {
                "total_loss": logs["total_loss"],
                "reconstruction_loss": logs["reconstruction_loss"],
                "content_kl_loss": logs["content_kl_loss"],
                "dynamics_kl_loss": logs["dynamics_kl_loss"],
            },
            "batch_buffer": self.batch_buffer.state_dict(),
            "config": self.config.to_dict(),
        }

        self.checkpoint_manager.save(
            checkpoint,
            checkpoint_name=f"step_{self.optimization_step}"
        )

    def _recording_step(self, logs: dict):
        """
        Records reconstruction examples periodically.
        """

        if self.recorder.should_record(self.optimization_step) == False:
            return
        
        index = torch.randint(0, logs["frames"].size(0), (1,)).item()

        record_path = self.recorder.save(
            original=logs["frames"][index : index+1],
            reconstruction=logs["reconstruction"][index : index+1],
            name=f"opt_step_{self.optimization_step}"
        )
        print(f"Environment step {self.global_step}; Episode {self.episode}; Optimization step {self.optimization_step}, : Reconstruction saved at {record_path}")

    def _optuna_step(self, logs: dict):
        """
        Reports intermediate results to Optuna and checks whether the
        current trial should be pruned.
        """

        if self.trial is None:
            return

        self.trial.report(
            logs["total_loss"],
            self.optimization_step,
        )

        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def train(self):
        """
        Starts DSAE training.
        """

        self._setup()

        print("Starting training...")

        try:
            while not self._finished():

                self._environment_step()

                logs =self._optimization_step()

                if logs is None:
                    continue

                print(f"Optimization step {self.optimization_step}: \n {logs}")

                self._logging_step(logs)

                self._checkpoint_step(logs)

                self._recording_step(logs)

                self._optuna_step(logs)

            return self.logger.last()
        
        finally:
            self.environment.close()
            