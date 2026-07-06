
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
import ale_py
gym.register_envs(ale_py) # Explicitly register the Atari games to gym

from models.nec import NECAgent

from utils.data_buffers import (FrameSequenceBuffer, 
                                TransitionQueue, 
                                ReplayMemory,
                                Transition,
                                ReplayMemoryUnit)

from utils.training_config import TrainingConfig
from utils.metrics_logger import MetricsLogger
from utils.checkpoint import CheckpointManager

from utils.misc import ensure_directory, preprocess_frame

from losses.dsae import reconstruction_loss, content_kl_loss, dynamics_kl_loss


class NECTrainer:
    """
    
    """
    
    def __init__(
        self,
        agent: NECAgent,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        experiment_dir: str | Path,
        device: str | torch.device = "cpu",
        trial=None,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.config = config
        self.trial = trial

        self.device = torch.device(device)
        self.agent.to(self.device)

        self.experiment_dir = ensure_directory(experiment_dir)

        self.logger = MetricsLogger(self.experiment_dir)
        self.checkpoint_manager = CheckpointManager(self.experiment_dir)
        self.recorder = VideoRecorder(self.experiment_dir, save_every=config.video_period)

        self.environment = gym.make(config.environment_name, render_mode=None)

        # Data buffers
        self.sequence_buffer = FrameSequenceBuffer(sequence_length=config.sequence_length)
        self.transition_queue = TransitionQueue(capacity=config.trajectory_length)
        self.replay_memory = ReplayMemory(capacity=config.replay_capacity)

        # Training progress
        self.global_step = 0
        self.episode = 0
        self.optimization_step = 0
    
    
    def _setup(self):
        """
        Initializes the environment and training buffers.
        """

        observation, _ = self.environment.reset()

        observation = preprocess_frame(observation)

        self.sequence_buffer.clear()
        self.transition_queue.clear()

        for _ in range(self.config.sequence_length):
            self.sequence_buffer.append(observation)
  
    
    def _finished(self) -> bool:
        """
        Returns whether training has finished.
        """


    def _environment_step(self):
        """
        Executes one environment interaction and stores the resulting
        transition in the trajectory buffer.
        """

        # Encode current state.
        state = self.sequence_buffer.get_sequence()

        encoder_output = self.agent.encode(
            frames=torch.from_numpy(state).unsqueeze(0).to(self.device),
            random_sampling=True,
        )

        # Select action.
        action = self.agent.choose_action(encoder_output, self.config.epsilon)

        # Environment interaction.
        observation, reward, terminated, truncated, _ = self.environment.step(action)
        observation = preprocess_frame(observation)
        self.sequence_buffer.append(observation)

        
        # Store transition.
        self.transition_queue.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                representation = (
                    encoder_output.representation.detach().cpu()
                    if self.config.cache_representations
                    else None
                )
            )
        )

        self.global_step += 1

        if terminated or truncated:

            self.episode += 1

            observation, _ = self.environment.reset()

            observation = preprocess_frame(observation)

            self.sequence_buffer.clear()

            for _ in range(self.config.sequence_length):
                self.sequence_buffer.append(observation)


    def _memory_optimization_step(self):
        """
        Computes N-step targets for the collected trajectory, updates the
        episodic memories and stores processed transitions into the replay
        memory.
        """

        updates_to_be_applied = []

        lookup_requirements = self.agent.update_strategy.lookup_requirements

        for transition_index, transition in enumerate(self.transition_queue):

            # Compute N-step target.
            q_target = self._compute_q_target(transition_index)

            representation = transition.representation

            # If representation stored in the transition is `None`, compute it.
            if representation is None: 
                encoder_output = self.agent.encode(
                    frames=torch.from_numpy(transition.state).unsqueeze(0).to(self.device),
                    random_sampling=True,
                )
                transition.representation = encoder_output.representation
                

            # Determine whether the state already exists in memory.
            contains = self.agent.contains(transition.representation, transition.action)

            if contains: # State exists in memory.

                # Update memory with original Bellman update
                update_request = self.agent.create_memory_update_request(
                    transition=transition,
                    q_target=q_target,
                    lookup_result=None
                )
                updates_to_be_applied.append(update_request)


            else: # State does not exist in memory. 

                # Lookup action-specific DND.
                lookup_result = self.agent.lookup_to_dnd(
                    action=transition.action,
                    representation=transition.representation,
                    auxiliary=None,  # Placeholder for future optional auxiliary.
                    return_indices=lookup_requirements.return_indices,
                    return_similarities=lookup_requirements.return_similarities,
                    return_neighbors=lookup_requirements.return_neighbors,
                )

                # Create memory update request.
                update_request = self.agent.create_memory_update_request(
                    transition=transition,
                    q_target=q_target,
                    lookup_result=lookup_result,
                    exploration_update=self.config.exploration_update,
                )
                updates_to_be_applied.extend(update_request)


            # Store transition for network optimization.
            transition.representation = None
            self.replay_memory.append(state=transition.state, action=transition.action, q_target=q_target)

        # Apply all memory updates simultaneously.
        self.agent.apply_memory_updates(updates_to_be_applied)


    




    def _optimization_step(self):
        """
        Performs one reinforcement learning optimization step if enough
        transitions have been collected.
        """



    def _network_optimization_step(self):
        """
        Optimizes the encoder (and optionally the DND keys) using
        mini-batches sampled from replay memory.
        """

    def _logging_step(self, logs):
        ...

    def _checkpoint_step(self, logs):
        ...

    def _recording_step(self):
        """
        Records gameplay videos periodically.
        """

    def _optuna_step(self, logs):
        ...

    def train(self):
        """
        Starts NEC training.
        """

        self._setup()

        if self.config.use_warmup:
            self._warmup()

        print("Starting training...")

        try:
            while not self._finished():

                self._environment_step()

                logs = self._optimization_step()

                if logs is None:
                    continue

                self._logging_step(logs)

                self._checkpoint_step(logs)

                self._recording_step()

                self._optuna_step(logs)

            return self.logger.last()

        finally:
            self.environment.close()