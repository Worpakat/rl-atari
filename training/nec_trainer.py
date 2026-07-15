
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
import ale_py

from utils.gym_wrappers import RestrictedActionWrapper, RewardWrapper
from gymnasium.wrappers import GrayscaleObservation
gym.register_envs(ale_py) # Explicitly register the Atari games to gym


from training.evaluator import Evaluator

from models.nec import NECAgent

from utils.data_buffers import (FrameSequenceBuffer, 
                                TransitionQueue, 
                                ReplayMemory,
                                Transition)

from utils.training_config import TrainingConfig
from utils.metrics_logger import MetricsLogger
from utils.checkpoint import CheckpointManager

from utils.misc import cut_and_transpose_frame, ensure_directory, convert_and_norm_sequence

from losses.nec_loss import compute_network_loss


class NECTrainer:
    """
    
    """
    
    def __init__(
        self,
        agent: NECAgent,
        encoder_optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        experiment_dir: str | Path,
        device: str | torch.device = "cpu",
    ):
        self.agent = agent
        self.encoder_optimizer = encoder_optimizer
        self.config = config

        self.device = torch.device(device)
        self.agent.to(self.device)

        self.experiment_dir = ensure_directory(experiment_dir)

        self.logger = MetricsLogger(self.experiment_dir)
        self.checkpoint_manager = CheckpointManager(self.experiment_dir)
        self.evaluator = (
                Evaluator(
                    self.agent, 
                    self.config, 
                    self.experiment_dir, 
                    self.device)
                )
        self.episode_reward = 0

        self.environment = self._init_environment()
    
        # Data buffers
        self.sequence_buffer = FrameSequenceBuffer(sequence_length=config.sequence_length)
        self.transition_queue = TransitionQueue(capacity=config.transition_queue_size)
        self.replay_memory = ReplayMemory(capacity=config.replay_memory_size)

        # Training progress
        self.global_step = 0
        self.episode = 0
        self.optimization_step = 0
        self.checkpoint_start = config.checkpoint_start
    
    
    def _init_environment(self):
        """
        Creates the training environment.
        """
        environment = gym.make(self.config.environment_name, render_mode=None)

        environment = RewardWrapper(environment,
                                    strategy=self.config.reward_strategy,
                                    parameters=self.config.reward_parameters)

        if self.config.action_mapping: # In case of mapping is changed.
            environment = RestrictedActionWrapper(
                environment,
                action_mapping=self.config.action_mapping,
            )

        if self.config.grayscale:
            environment = GrayscaleObservation(environment)

        return environment

    def _setup(self):
        """
        Initializes the environment and training buffers.
        """
        observation, _ = self.environment.reset()

        observation = cut_and_transpose_frame(observation)

        self.sequence_buffer.clear()
        self.transition_queue.clear()

        for _ in range(self.config.sequence_length):
            self.sequence_buffer.append(observation)
  
    def _finished(self) -> bool:
        """
        Returns whether training has finished.
        """
        return self.episode >= self.config.max_episodes

    def _warmup_finished(self) -> bool:
        """
        Returns whether warmup has finished.
        """
        return self.global_step >= self.config.warmup_steps

    def warmup(self):
        """
        Populates the replay memory and DNDs using a random policy before
        reinforcement learning begins.
        """
        print("Starting warmup...")

        while not self._warmup_finished():

            self._setup()
            
            counter = 0
            
            while True:
                counter += 1

                action = self.environment.action_space.sample()

                observation, reward, terminated, truncated, _ = (
                    self.environment.step(action)
                )

                observation = cut_and_transpose_frame(observation)

                self.sequence_buffer.append(observation)

                if not self.sequence_buffer.is_ready():
                    continue

                raw_state = self.sequence_buffer.get_raw_sequence()
                state = convert_and_norm_sequence(raw_state)
                
                frames = torch.from_numpy(state).unsqueeze(0).unsqueeze(2).to(self.device)
                print(f"Frames shape: {frames.shape}")

                encoder_output = self.agent.encode(
                    frames=(
                        torch.from_numpy(state)
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .to(self.device)),
                    random_sampling=False,
                )

                self.transition_queue.append(
                    Transition(
                        state=raw_state,
                        action=action,
                        reward=reward,
                        representation = (
                            encoder_output.representation.detach().cpu()
                            if self.config.cache_representations
                            else None
                            ),
                        is_exploration_action=True
                    )
                )

                self.global_step += 1

                if terminated or truncated or self.transition_queue.is_full():
                    print(f"Transition Queue has taken {counter} transitions; Transition Queue size: {len(self.transition_queue)}")
                    break

            q_targets = self.agent.compute_q_targets(
                transition_queue=self.transition_queue,
                gamma=self.config.gamma,
                n_step=self.config.n_step,
                warmup=True,
            )

            updates_to_be_applied = []

            for transition, q_target in zip(self.transition_queue, q_targets):

                # Determine whether the state already exists in memory.
                index = self.agent.get_memory_index(transition.representation, transition.action)

                if index: # State exists in memory.

                    # Update memory with original Bellman update
                    update_request = self.agent.create_memory_update_request(
                        transition=transition,
                        q_target=q_target,
                        lookup_result=None,
                        update_or_insert="update",
                        index=index,
                        warmup=True,
                    )
                    updates_to_be_applied.append(update_request)

                else: # State does not exist in memory. 

                    # Create memory update request: Original NEC Q-target insert
                    update_request = self.agent.create_memory_update_request(
                        transition=transition,
                        q_target=q_target,
                        lookup_result=None,
                        warmup=True,
                        update_or_insert="insert",
                        exploration_update=False
                    )
                    updates_to_be_applied.extend(update_request)

                # Store transition for network optimization.
                transition.representation = None
                self.replay_memory.append(state=transition.state, action=transition.action, q_target=q_target)

            self.agent.apply_memory_updates(updates_to_be_applied)


        print("Warmup completed.")

    def _environment_step(self):
        """
        Executes one environment interaction and stores the resulting
        transition in the trajectory buffer.
        """
        terminated = False
        truncated = False

        while not (terminated or truncated) and not self.transition_queue.is_full():

            # Encode current state.
            raw_state = self.sequence_buffer.get_raw_sequence()
            state = convert_and_norm_sequence(raw_state)

            encoder_output = self.agent.encode(
                frames=(
                    torch.from_numpy(state)
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .to(self.device)),
                random_sampling=False,
            )

            # Select action.
            action, is_exploration = self.agent.choose_action(encoder_output)

            # Environment interaction.
            observation, reward, terminated, truncated, _ = self.environment.step(action)
            self.episode_reward += reward

            observation = cut_and_transpose_frame(observation)
            self.sequence_buffer.append(observation)
            
            # Store transition.
            self.transition_queue.append(
                Transition(
                    state=raw_state,
                    action=action,
                    reward=reward,
                    representation = (
                        encoder_output.representation.detach().cpu()
                        if self.config.cache_representations
                        else None
                    ),
                    is_exploration_action=is_exploration
                )
            )

            self.global_step += 1

            if terminated or truncated or self.transition_queue.is_full():
                # Either episode ends or trajectory is full, in which case we stop.
                self.episode += 1

                observation, _ = self.environment.reset()

                observation = convert_and_norm_sequence(observation)

                self.sequence_buffer.clear()

                for _ in range(self.config.sequence_length):
                    self.sequence_buffer.append(observation)
                
                break
        
        # Decay epsilon after each episode.
        self.agent.decay_epsilon()

    def _memory_optimization_step(self):
        """
        Computes N-step targets for the collected trajectory, updates the
        episodic memories and stores processed transitions into the replay
        memory.
        """
        updates_to_be_applied = []

        lookup_requirements = self.agent.update_strategy.lookup_requirements   
        
        # Compute N-step targets at once.
        q_targets = self.agent.compute_q_targets(
            transition_queue=self.transition_queue,
            gamma=self.config.gamma,
            n_step=self.config.n_step,
        )

        for transition_index, transition in enumerate(self.transition_queue):

            q_target = q_targets[transition_index]
            representation = transition.representation

            # If representation stored in the transition is `None`, compute it.
            if representation is None: 
                state = convert_and_norm_sequence(transition.state)

                encoder_output = self.agent.encode(
                    frames=(
                        torch.from_numpy(state)
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .to(self.device)),
                    random_sampling=False,
                )
                transition.representation = encoder_output.representation
                

            # Determine whether the state already exists in memory.
            index = self.agent.get_memory_index(transition.representation, transition.action)

            if index: # State exists in memory.

                # Update memory with original Bellman update
                update_request = self.agent.create_memory_update_request(
                    transition=transition,
                    q_target=q_target,
                    lookup_result=None,
                    update_or_insert="update",
                    index=index,
                    warmup=False,
                )
                updates_to_be_applied.append(update_request)


            else: # State does not exist in memory. 

                # Lookup action-specific DND.
                lookup_result = self.agent.lookup_to_dnd(
                    action=transition.action,
                    key=transition.representation,
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

    def _network_optimization_step(self):
        """
        Optimizes the encoder (and optionally the DND keys) using
        mini-batches sampled from replay memory.
        """

        if not self.replay_memory.can_sample(self.config.batch_size):
            return

        losses = []

        # Network will be optimized every 'network_optimization_period' transitions.
        steps = int(len(self.transition_queue) / self.config.network_optimization_period)
        
        # Last place it is used in an episode, we clear the transition queue to gain space.
        self.transition_queue.clear() 

        for _ in range(steps):

            # Sample a mini-batch.
            batch = self.replay_memory.sample(self.config.batch_size)
            states, actions, q_targets = self.replay_memory.extract_batch(batch)

            # Encode state sequences.
            encoder_output = self.agent.encode(states, random_sampling=False) # We use 'posterior_mean's as representations for stability

            # Estimate Q-values from the episodic memories.
            predicted_q_values = self.agent.lookup_batch(
                representations=encoder_output.representation,
                actions=actions,
                track_key_updates=self.config.key_updates,
            )

            # Compute optimization loss.
            loss = compute_network_loss(
                predicted_q_values=predicted_q_values,
                q_targets=q_targets,
                encoder_output=encoder_output
            )

            # Optimize encoder.
            self.encoder_optimizer.zero_grad()
            self.agent.zero_key_gradients()

            loss['total_loss'].backward()

            self.encoder_optimizer.step()

            # Optionally optimize DND keys.
            if self.config.key_updates:
                self.agent.step_key_optimizers()

            self.optimization_step += 1

            # Store loss to be logged.
            loss['optimization_step'] = self.optimization_step
            losses.append(loss)


        return losses

    
    ##=========LOGGING_AND_EVALUATION===========
    
    def _should_checkpoint(self):
        return self.episode % self.config.checkpoint_period == 0
    
    def _should_evaluate(self):
        return self.episode % self.config.evaluation_period == 0
    
    def _logging_step(self, logs: dict | None):
        """
        Records training metrics.
        """
        # Multiple optimization steps
        for l in logs:
            self.logger.log(
                optimization_step=l['optimization_step'],
                global_step=self.global_step,
                episode=self.episode,
                total_reward=self.episode_reward,
                total_loss=l["total_loss"].item(),
                td_loss=l["td_loss"].item(),
                kl_loss=l["kl_loss"].item(),
            )
        
        print(f"Episode {self.episode}, optimization step {self.optimization_step}, is fnished.")
        print(self.logger.last())
        
        self.episode_reward = 0

        if self._should_checkpoint(): # Saving logs and checkpoint simultaneously.
            self.logger.save(
                start_step=self.checkpoint_start,
                end_step=self.optimization_step,
                step_name="opt_step",
                clear=True
            )

    def _checkpoint_step(self, logs: dict):
        """
        Saves a training checkpoint periodically.
        """
        
        print("Saving model...")
        
        model_checkpoint = {
            "model": self.agent.state_dict(),
            "optimizer": self.encoder_optimizer.state_dict(),
            "training_state": {
                "optimization_step": self.optimization_step,
                "environment_step": self.global_step,
                "episode": self.episode,
            },
            # "metrics": {
            #     "total_loss": logs["total_loss"],
            #     "reconstruction_loss": logs["reconstruction_loss"],
            #     "content_kl_loss": logs["content_kl_loss"],
            #     "dynamics_kl_loss": logs["dynamics_kl_loss"],
            # },
        }
    
        dnd_sizes = []
        for i, dnd in enumerate(self.agent.dnds):
            dnd_sizes.append(dnd.keys.numel() * dnd.keys.element_size() / 1024**2)

        print(f"DND sizes: {dnd_sizes} | total: {np.sum(dnd_sizes)} MB")

        # print("---------------------------------------")
        # print("AGENT STATE DICTIONARY:")
        # print(self.agent.state_dict())

        self.checkpoint_manager.save(
            model_checkpoint,
            filename=f"model_ep_{self.episode}_step_{self.optimization_step}",
            colab_execution=self.config.colab_execution
        )

        if self.config.save_replay_memory:
            print(f"Replay memory length: {len(self.replay_memory)}")
            print(f"Replay memory states total size: {self.replay_memory.get_states_total_size()}")

            print("Saving replay memory...")

            replay_memory_checkpoint = {
                "replay_memory": self.replay_memory.state_dict(),
                "training_state": {
                    "optimization_step": self.optimization_step,
                    "environment_step": self.global_step,
                    "episode": self.episode,
                }
            }
      
            self.checkpoint_manager.save(
                replay_memory_checkpoint,
                filename=f"rep_memo_ep_{self.episode}_step_{self.optimization_step}",
                colab_execution=self.config.colab_execution
            )

        
        print("Checkpoint check: Checkpoint saved.")

        self.checkpoint_start = self.optimization_step



    def _optuna_step(self, logs):
        ...

    def train(self):
        """
        Starts NEC training.
        """
        self._setup()

        print("Starting training...")

        try:

            while not self._finished():
                
                self._environment_step()

                self._memory_optimization_step()

                logs = self._network_optimization_step()

                self._logging_step(logs)

                if self._should_checkpoint():
                    self._checkpoint_step(logs)

                if self._should_evaluate():
                    print("Evaluating...")
                    evaluation_summary = self.evaluator.evaluate(
                        render_mode='rgb_array',
                        log_file_custom=f"ep_{self.episode}",
                        video_file_custom=f"ep_{self.episode}"
                    )

                    self._optuna_step(evaluation_summary)

            return self.logger.last()

        finally:

            self.environment.close()