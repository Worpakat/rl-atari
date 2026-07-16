
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

from utils.data_buffers import (FrameSequenceBuffer, ReplayMemory)
                                

from utils.transition_classes import (
    Transition,
    TransitionDelayBuffer,
    TransitionQueue,        
    TransitionQueueManager,
)

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
        

        self.episode_reward = 0 # Current episode reward
        self.environment = self._init_environment()
        self.death_penalty = self.config.death_penalty
        self.current_lives = 0
    
        # Data buffers
        self.sequence_buffer = FrameSequenceBuffer(sequence_length=config.sequence_length)
        self.transition_queue_manager = TransitionQueueManager(capacity=config.transition_queue_size)
        self.transition_delay_buffer = TransitionDelayBuffer(delay=self.config.transition_buffer["transition_delay"])
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
        observation, info = self.environment.reset()

        observation = cut_and_transpose_frame(observation)

        self.current_lives = info["lives"]
        self.episode_reward = 0

        self.sequence_buffer.clear()
        self.transition_queue_manager.clear()
        self.transition_delay_buffer.clear()

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

        self._setup()

        while True:

            # Gather one complete episode.
            while True:

                action = self.environment.action_space.sample()

                observation, reward, terminated, truncated, info = self.environment.step(action)
                
                observation = cut_and_transpose_frame(observation)
                self.sequence_buffer.append(observation)

                raw_state = self.sequence_buffer.get_raw_sequence()
                state = convert_and_norm_sequence(raw_state)

                encoder_output = self.agent.encode(
                    frames=(
                        torch.from_numpy(state)
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .to(self.device)
                    ),
                    random_sampling=False,
                )

                # Detect life loss.
                death = (
                    self.death_penalty is not None
                    and info["lives"] < self.current_lives
                )

                if death:
                    reward = self.death_penalty

                # Store transition.
                transition = Transition(
                    state=raw_state,
                    action=action,
                    reward=reward,
                    representation=(
                        encoder_output.representation.detach().cpu()
                        if self.config.cache_representations
                        else None
                    ),
                    is_exploration_action=True,
                )

                released_transition = self.transition_delay_buffer.append(transition)

                if released_transition is not None:
                    self.transition_queue_manager.append(released_transition)


                self.global_step += 1

                # Finish trajectory on death.
                if death:
                    # Retrieve actual termination transition. (Actual transition that death is occurred.)
                    terminal_transition = self.transition_delay_buffer.pop_oldest()

                    terminal_transition.reward = self.death_penalty
                    
                    self.transition_queue_manager.append(terminal_transition)
                    
                    self.transition_queue_manager.end_trajectory()
                    
                    # Remaining static states are discarded. We don't incluede those neiher in replay memory or training.
                    self.transition_delay_buffer.discard_all()

                    self.sequence_buffer.clear()

                    for _ in range(self.config.sequence_length):
                        self.sequence_buffer.append(observation)

                self.current_lives = info["lives"]

                # Episode finished.
                if terminated or truncated:
                    # Discard terminal animation frames.
                    self.transition_delay_buffer.discard_newest(self.config.transition_buffer["terminal_static_frames"])

                    # Release remaining transitions.
                    while len(self.transition_delay_buffer) > 0:
                        self.transition_queue_manager.append(
                            self.transition_delay_buffer.pop_oldest()
                        )

                    if self.death_penalty is not None: # Apply death penalty to last transition if it is used.
                        last_transition = self.transition_queue_manager.get_last_transition()
                        last_transition.reward = self.death_penalty
                    
                    self.transition_queue_manager.end_trajectory()

                    break

            # ----------MEMORY_UPDATE----------
            updates_to_be_applied = []
            
            for transition_queue in self.transition_queue_manager:

                q_targets = self.agent.compute_q_targets(
                    transition_queue=transition_queue,
                    gamma=self.config.gamma,
                    n_step=self.config.n_step,
                    warmup=True,
                )

                for transition, q_target in zip(transition_queue, q_targets):
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

            # Apply memory updates.
            self.agent.apply_memory_updates(updates_to_be_applied)

            # Clear transition queue in case of warmup and episode is not over.
            self.transition_queue_manager.clear()
            self.transition_delay_buffer.clear()

            # Warmup stopping condition.
            if self._warmup_finished():
                break

            # Start next episode.
            observation, info = self.environment.reset()

            observation = cut_and_transpose_frame(observation)

            self.sequence_buffer.clear()

            for _ in range(self.config.sequence_length):
                self.sequence_buffer.append(observation)

            self.current_lives = info["lives"]

        print("Warmup complete.")

    
    def _environment_step(self):
        """
        Executes environment interactions until either

        - the current episode ends, or
        - the transition queue manager reaches capacity.
        """

        while True:

            # Encode current state.
            raw_state = self.sequence_buffer.get_raw_sequence()
            state = convert_and_norm_sequence(raw_state)

            encoder_output = self.agent.encode(
                frames=(
                    torch.from_numpy(state)
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .to(self.device)
                ),
                random_sampling=False,
            )

            # Action selection.
            action, is_exploration = self.agent.choose_action(encoder_output)

            # Environment interaction.
            observation, reward, terminated, truncated, info = self.environment.step(action)
            
            observation = cut_and_transpose_frame(observation)
            self.sequence_buffer.append(observation)

            # Detect life loss.
            death = (self.death_penalty is not None and info["lives"] < self.current_lives)

            if death:
                reward = self.death_penalty

            self.episode_reward += reward

            # Store transition.
            transition = Transition(
                state=raw_state,
                action=action,
                reward=reward,
                representation=(
                    encoder_output.representation.detach().cpu()
                    if self.config.cache_representations
                    else None
                ),
                is_exploration_action=is_exploration,
            )

            released_transition = self.transition_delay_buffer.append(transition)

            if released_transition is not None:
                self.transition_queue_manager.append(released_transition)

            self.global_step += 1

            # Finish current trajectory if a life was lost.
            if death:
                # Retrieve actual termination transition. (Actual transition that death is occurred.)
                terminal_transition = self.transition_delay_buffer.pop_oldest()

                terminal_transition.reward = self.death_penalty

                self.transition_queue_manager.end_trajectory()
                
                # Remaining static states are discarded. We don't incluede those neiher in replay memory or training.
                self.transition_delay_buffer.discard_all()

                self.sequence_buffer.clear()

                for _ in range(self.config.sequence_length):
                    self.sequence_buffer.append(observation)


            self.current_lives = info["lives"]

            # Episode finished.
            if terminated or truncated:
                # Discard terminal animation frames.
                self.transition_delay_buffer.discard_newest(self.config.transition_buffer["terminal_static_frames"])

                # Release remaining transitions.
                while len(self.transition_delay_buffer) > 0:
                    self.transition_queue_manager.append(
                        self.transition_delay_buffer.pop_oldest()
                    )

                if self.death_penalty is not None: # Apply death penalty to last transition if it is used.
                    last_transition = self.transition_queue_manager.get_last_transition()
                    last_transition.reward = self.death_penalty
                
                self.transition_queue_manager.end_trajectory()

                break

            # Gathering complete.
            if self.transition_queue_manager.is_full():
                break

        # Decay epsilon
        self.agent.decay_epsilon()


    def _memory_optimization_step(self):
        """
        Computes N-step targets for the collected trajectory, updates the
        episodic memories and stores processed transitions into the replay
        memory.
        """
        updates_to_be_applied = []
        lookup_requirements = self.agent.update_strategy.lookup_requirements   
        
        for transition_queue in self.transition_queue_manager:

            # Compute N-step targets at once.
            q_targets = self.agent.compute_q_targets(
                transition_queue=transition_queue,
                gamma=self.config.gamma,
                n_step=self.config.n_step,
            )

            for transition_index, transition in enumerate(transition_queue):

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
        steps = int(len(self.transition_queue_manager) / self.config.network_optimization_period)
        
        # Last place it is used in an episode, we clear the transition queue to gain space.
        self.transition_queue_manager.clear() 
        self.transition_delay_buffer.clear()

        for _ in range(steps):

            # Sample a mini-batch.
            batch = self.replay_memory.sample(self.config.batch_size)
            states, actions, q_targets = self.replay_memory.extract_batch(batch, device=self.device)

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

            # Store loss to be logged.
            loss['optimization_step'] = self.optimization_step
            losses.append(loss)
            self.optimization_step += 1


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
            }
        }
    
        dnd_sizes = []
        for i, dnd in enumerate(self.agent.dnds):
            dnd_sizes.append(dnd.keys.numel() * dnd.keys.element_size() / 1024**2)

        # print(f"DND sizes: {dnd_sizes} | total: {np.sum(dnd_sizes)} MB")

        self.checkpoint_manager.save(
            model_checkpoint,
            filename=f"model_ep_{self.episode}_step_{self.optimization_step}",
            colab_execution=self.config.colab_execution
        )

        if self.config.save_replay_memory:
            print(f"Replay memory length: {len(self.replay_memory)}")
            print(f"Replay memory states total size: {self.replay_memory.get_states_total_size()} MB")

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
        self.episode += 1


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

                self._update_episode_step()

            return self.logger.last()

        finally:

            self.environment.close()




###===================================================================
###===================================================================

###===========================OLD_FUNCTIONS===========================

    # def warmup(self):     
    #     """
    #     Populates the replay memory and DNDs using a random policy before
    #     reinforcement learning begins.
    #     """
    #     print("Starting warmup...")
        
    #     setup_flag = True # Used to whether to reset the environment or not.
    #     # Reset the environment if episode is over. Do not reset if episode is not over but only death occurred.
        
    #     while not self._warmup_finished():
            
    #         if setup_flag:
    #             info = self._setup()
    #             setup_flag = False
            
    #         counter = 0
    #         self.current_lives = self.environment.unwrapped.ale.lives()

    #         while True:
    #             counter += 1

    #             action = self.environment.action_space.sample()

    #             observation, reward, terminated, truncated, info = self.environment.step(action)

    #             observation = cut_and_transpose_frame(observation)

    #             self.sequence_buffer.append(observation)

    #             raw_state = self.sequence_buffer.get_raw_sequence()
    #             state = convert_and_norm_sequence(raw_state)

    #             encoder_output = self.agent.encode(
    #                 frames=(
    #                     torch.from_numpy(state)
    #                     .unsqueeze(0)
    #                     .unsqueeze(2)
    #                     .to(self.device)),
    #                 random_sampling=False,
    #             )

    #             # Check for death, if so, apply given death penalty.
    #             death_flag = self.death_penalty and info["lives"] < self.current_lives
    #             if death_flag: 
    #                 reward = self.death_penalty

    #             self.transition_queue.append(
    #                 Transition(
    #                     state=raw_state,
    #                     action=action,
    #                     reward=reward,
    #                     representation = (
    #                         encoder_output.representation.detach().cpu()
    #                         if self.config.cache_representations
    #                         else None
    #                         ),
    #                     is_exploration_action=True
    #                 )
    #             )

    #             self.global_step += 1

    #             ## !! REFACTOR this functions death episode ending conditioning and resetting. 
    #             # Do it at least same as we do in _environment_step function.

    #             if terminated or truncated: # Episode is over.
    #                 print(f"Transition Queue has taken {counter} transitions; Transition Queue size: {len(self.transition_queue)}")
    #                 setup_flag = True # Reset the environment
    #                 break

    #             if death_flag or self.transition_queue.is_full():
    #                 print (f"Remaining lives: {info['lives']}")
    #                 print(f"Transition Queue has taken {counter} transitions; Transition Queue size: {len(self.transition_queue)}")
    #                 # Do not reset the environment, since the episode is not over

    #                 # ! BUT: Fill sequence buffer with last state.
    #                 self.sequence_buffer.clear()
    #                 for _ in range(self.config.sequence_length):
    #                     self.sequence_buffer.append(observation)
                    
    #                 break 
    #                 ## ! We break anyway, to not connect death previous and after episodes while calculating Q-targets.

    #         q_targets = self.agent.compute_q_targets(
    #             transition_queue=self.transition_queue,
    #             gamma=self.config.gamma,
    #             n_step=self.config.n_step,
    #             warmup=True,
    #         )

    #         updates_to_be_applied = []

    #         for transition, q_target in zip(self.transition_queue, q_targets):

    #             # Determine whether the state already exists in memory.
    #             index = self.agent.get_memory_index(transition.representation, transition.action)

    #             if index: # State exists in memory.

    #                 # Update memory with original Bellman update
    #                 update_request = self.agent.create_memory_update_request(
    #                     transition=transition,
    #                     q_target=q_target,
    #                     lookup_result=None,
    #                     update_or_insert="update",
    #                     index=index,
    #                     warmup=True,
    #                 )
    #                 updates_to_be_applied.append(update_request)

    #             else: # State does not exist in memory. 

    #                 # Create memory update request: Original NEC Q-target insert
    #                 update_request = self.agent.create_memory_update_request(
    #                     transition=transition,
    #                     q_target=q_target,
    #                     lookup_result=None,
    #                     warmup=True,
    #                     update_or_insert="insert",
    #                     exploration_update=False
    #                 )
    #                 updates_to_be_applied.extend(update_request)

    #             # Store transition for network optimization.
    #             transition.representation = None
    #             self.replay_memory.append(state=transition.state, action=transition.action, q_target=q_target)

    #         self.agent.apply_memory_updates(updates_to_be_applied)

    #         # Clear transition queue in case of warmup and episode is not over.
    #         self.transition_queue.clear()


    #     print("Warmup completed.")




    # def _environment_step(self):
    #         """
    #         Executes one environment interaction and stores the resulting
    #         transition in the trajectory buffer.
    #         """

    #         while True:

    #             # Encode current state.
    #             raw_state = self.sequence_buffer.get_raw_sequence()
    #             state = convert_and_norm_sequence(raw_state)

    #             encoder_output = self.agent.encode(
    #                 frames=(
    #                     torch.from_numpy(state)
    #                     .unsqueeze(0)
    #                     .unsqueeze(2)
    #                     .to(self.device)),
    #                 random_sampling=False,
    #             )

    #             # Select action.
    #             action, is_exploration = self.agent.choose_action(encoder_output)

    #             # Environment interaction.
    #             observation, reward, terminated, truncated, info = self.environment.step(action)

    #             observation = cut_and_transpose_frame(observation)
    #             self.sequence_buffer.append(observation)
                
    #             # Check for death, if so, apply given death penalty.
    #             death_flag = self.death_penalty and info["lives"] < self.current_lives
    #             if death_flag: 
    #                 reward = self.death_penalty
                
    #             self.episode_reward += reward
                
    #             # Store transition.
    #             self.transition_queue.append(
    #                 Transition(
    #                     state=raw_state,
    #                     action=action,
    #                     reward=reward,
    #                     representation = (
    #                         encoder_output.representation.detach().cpu()
    #                         if self.config.cache_representations
    #                         else None
    #                     ),
    #                     is_exploration_action=is_exploration
    #                 )
    #             )

    #             self.global_step += 1

    #             if (
    #                 terminated 
    #                 or truncated 
    #                 or self.transition_queue.is_full() 
    #                 or death_flag
    #             ):
    #                 # Either episode ends or trajectory is full or death penalty is applied, in which case we stop.
                    
    #                 if terminated or truncated:
    #                     # Episode is over, reset environment.
    #                     self.episode_ended = True
    #                     observation, info = self.environment.reset()

    #                 self.current_lives = info["lives"] # We assign current lives in any case.
    #                 # If episode is over, reset lives. 
    #                 # If death penalty is applied, update lives.

    #                 observation = cut_and_transpose_frame(observation)

    #                 self.sequence_buffer.clear()

    #                 for _ in range(self.config.sequence_length):
    #                     self.sequence_buffer.append(observation)
                    
    #                 break
            
    #         # Decay epsilon after each episode.
    #         self.agent.decay_epsilon()