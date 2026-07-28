
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

from models.data_buffers import (FrameSequenceBuffer, ReplayMemory, StratifiedReplayMemory)
                                

from models.transition_classes import (
    Transition,
    TransitionQueueManager,
    RiverRaidStaticSequenceHandler,
    TrajectoryType
)

from losses.nec_loss import compute_network_loss

from utils.training_config import TrainingConfig
from utils.metrics_logger import MetricsLogger
from utils.checkpoint import CheckpointManager

from utils.misc import ensure_directory, print_and_save_death_transitions
from utils.frame_processing import cut_and_transpose_frame, convert_and_norm_sequence

####
def print_gpu_usage(where):
    # Returns (free_bytes, total_bytes) on the GPU
    total_bytes = torch.cuda.memory_reserved()

    # free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)

    # print(f"Free physical VRAM {where}:  {free_gb:.2f} GB / {total_gb:.2f} GB")
    
    print(f"VRAM allocated {where}:  {total_gb:.2f} GB")


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
        
        self.batch_index_logger = None
        if self.config.get("log_batch_index", False):
            self.batch_index_logger = MetricsLogger(self.experiment_dir)


        self.episode_reward = 0 # Current episode reward
        self.environment = self._init_environment()
        self.death_penalty = self.config.death_penalty
        self.current_lives = 0
    
        # Data buffers
        self.sequence_buffer = FrameSequenceBuffer(sequence_length=config.sequence_length)
        self.transition_queue_manager = TransitionQueueManager(capacity=config.transition_queue_size)
        self.static_sequence_handler = RiverRaidStaticSequenceHandler(
            initial_static_frames=config["static_sequence_handler"]["initial_static_frames"], 
            intermediate_static_frames=config["static_sequence_handler"]["death_static_frames"],
            terminal_static_frames=config["static_sequence_handler"]["terminal_static_frames"],
            sequence_length=config.sequence_length
        )

        # Replay memory
        self.replay_memory = None
        if self.config.replay_memory_type == "normal":
            self.replay_memory = ReplayMemory(
                **self.config.normal_memory_kwargs
                # capacity=config.replay_memory_size,
                # prioritized=config.prioritized_replay,
                # priority_alpha=config.priority_alpha,
                # priority_epsilon=config.priority_epsilon,
            )

        elif self.config.replay_memory_type == "stratified":
            self.replay_memory = StratifiedReplayMemory(
                **self.config.stratified_memory_kwargs
            )

        else:
            raise ValueError(f"Invalid replay memory type: {self.config.replay_memory_type}. Use 'normal' or 'stratified'.")
        

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
            
            first_flag = True # Flag for the first trajectory. Used with the StaticSequenceHandler.

            # Gather one complete episode.
            while True:

                action = self.environment.action_space.sample()

                observation, reward, terminated, truncated, info = self.environment.step(action)
                
                observation = cut_and_transpose_frame(observation)
                self.sequence_buffer.append(observation)

                raw_state = self.sequence_buffer.get_raw_sequence()
                state = convert_and_norm_sequence(raw_state)

                self.agent.encoder.eval() # ! Look at end of the file, why we use eval() here.
                with torch.no_grad(): # We don't need gradients during warmup.
                    
                    encoder_output = self.agent.encode(
                        frames=(
                            torch.from_numpy(state)
                            .unsqueeze(0)
                            .unsqueeze(2)
                            .to(self.device)
                        ),
                        random_sampling=False,
                    )

                self.agent.encoder.train()

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

                self.transition_queue_manager.append(transition)

                self.global_step += 1

                # Finish trajectory on death.
                if death and not (terminated or truncated): 
                    # ! Each episode end transition also a death transition.
                    # We don't handle episode end last transition here.

                    current_queue = self.transition_queue_manager.get_current_trajectory()

                    trajectory_type = None

                    if first_flag:
                        trajectory_type = TrajectoryType.FIRST
                        first_flag = False
                    else:
                        trajectory_type = TrajectoryType.INTERMEDIATE
                    
                    # Handle static transitions.
                    self.static_sequence_handler.process(current_queue, trajectory_type, self.death_penalty)

                    self.transition_queue_manager.end_trajectory()
                

                    # Prepare next trajectory.
                    self.sequence_buffer.clear()

                    for _ in range(self.config.sequence_length):
                        self.sequence_buffer.append(observation)

                self.current_lives = info["lives"]

                # Episode finished.
                if terminated or truncated:

                    current_queue = self.transition_queue_manager.get_current_trajectory()

                    trajectory_type = TrajectoryType.LAST
                    
                    # Handle static transitions.
                    self.static_sequence_handler.process(current_queue, trajectory_type, self.death_penalty)        
                    
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
                    self.replay_memory.append(transition, q_target=q_target, warmup=True)

            # Apply memory updates.
            self.agent.apply_memory_updates(updates_to_be_applied)

            # Clear transition queue in case of warmup and episode is not over.
            self.transition_queue_manager.clear()

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

        first_flag = True # Flag for the first trajectory. Used with the StaticSequenceHandler.

        while True:

            # Encode current state.
            raw_state = self.sequence_buffer.get_raw_sequence()
            state = convert_and_norm_sequence(raw_state)

            self.agent.encoder.eval() # ! Look at end of the file, why we use eval() here.
            with torch.no_grad(): # We don't need gradients during warmup.
                
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
                # This contains lookup, which means has to be in no_grad context.
            
            # Getting out of no_grad context.
            self.agent.encoder.train() 

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

            self.transition_queue_manager.append(transition)    


            self.global_step += 1

            # Finish current trajectory if a life was lost.
            if death and not (terminated or truncated): 
                # ! Each episode end transition also a death transition.
                # We don't handle episode end transition here.

                current_queue = self.transition_queue_manager.get_current_trajectory()

                trajectory_type = None

                if first_flag:
                    trajectory_type = TrajectoryType.FIRST
                    first_flag = False
                else:
                    trajectory_type = TrajectoryType.INTERMEDIATE
                
                # Handle static transitions.
                self.static_sequence_handler.process(current_queue, trajectory_type, self.death_penalty)

                self.transition_queue_manager.end_trajectory()

                # Prepare next trajectory.
                self.sequence_buffer.clear()

                for _ in range(self.config.sequence_length):
                    self.sequence_buffer.append(observation)


            self.current_lives = info["lives"]

            # Capacity reached, end transition gathering.
            if self.transition_queue_manager.is_full():                
                
                # Reset environment.
                observation, info = self.environment.reset()
                self.current_lives = info["lives"] 

                observation = cut_and_transpose_frame(observation)
                self.sequence_buffer.clear()

                for _ in range(self.config.sequence_length):
                    self.sequence_buffer.append(observation)
                
                # ?! Why we check 'transition_queue_manager.is_full()' before than that if episode is ended?:
                # * Capacity could be full and episode could be ended because 
                # "terminated" or "truncated" flags or proceeds before the appended transitions 
                # due to delayed appends. 
                # In that case if we don't break here, following conditions are met and
                # some of remaining transitions are tried to append to full transition queue.
                # Eventually exception is raised.
                break

            # Episode finished.
            if terminated or truncated:
                
                current_queue = self.transition_queue_manager.get_current_trajectory()

                trajectory_type = TrajectoryType.LAST
                
                # Handle static transitions.
                self.static_sequence_handler.process(current_queue, trajectory_type, self.death_penalty)        
                
                self.transition_queue_manager.end_trajectory()

                # Reset environment.
                observation, info = self.environment.reset()
                self.current_lives = info["lives"] 

                observation = cut_and_transpose_frame(observation)
                self.sequence_buffer.clear()

                for _ in range(self.config.sequence_length):
                    self.sequence_buffer.append(observation)

                break

        # Decay epsilon
        self.agent.decay_epsilon()
        
        self.episode += 1

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
    
                    self.agent.encoder.eval() # ! Look at end of the file, why we use eval() here.
                    with torch.no_grad(): 
                        # Only purpose is getting the representation again.
                        # As we do at _environment_step(). Hence, no gradients.

                        encoder_output = self.agent.encode(
                            frames=(
                                torch.from_numpy(state)
                                .unsqueeze(0)
                                .unsqueeze(2)
                                .to(self.device)),
                            random_sampling=False,
                        )
                        transition.representation = encoder_output.representation

                    self.agent.encoder.train()
                    

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

                    with torch.no_grad(): 
                        # ! We don't want to gather key gradients here during lookup.
                        # It should be done only _network_optimization_step().

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
                self.replay_memory.append(transition, q_target=q_target, warmup=False)

        # Apply all memory updates simultaneously.
        insert_update_counts = self.agent.apply_memory_updates(updates_to_be_applied)
        
        return insert_update_counts

    def _network_optimization_step(self):
        """
        Optimizes the encoder (and optionally the DND keys) using
        mini-batches sampled from replay memory.
        """
        if not self.replay_memory.can_sample(self.config.batch_size):
            return
        
        losses = []

        # Network will be optimized every 'network_optimization_period' transitions.
        steps = int(len(self.transition_queue_manager) / self.config.network_optimization_period) + 1
        
        # Last place it is used in an episode, we clear the transition queue to gain space.
        self.transition_queue_manager.clear() 

        # Stratified replay memory spesifics.
        if isinstance(self.replay_memory, StratifiedReplayMemory): 

            # Stratified replay memory needs to mark death windows before any optimization step. 
            # Those are going to moved to death bucket.
            self.replay_memory.mark_death_windows() 
            self.replay_memory.lock_new_bucket_size()

            if self.replay_memory.first_turn:
                all_batches = []
                all_td_errors_abs = [] 

                # WHY?:
                # We don't need any TD stats at the beginning.
                # Thus, we can't move anything between buckets 
                # without complete all first _network_optimization_step() run through.
                # Eventually, we gather all batches and TD errors;
                # At the end we calculate TD stats, and move transitions to proeper buckets.

        # print(f"GPU memory usage, before network optimization steps: \n {torch.cuda.memory_summary(device=None, abbreviated=False)}")


        for _ in range(steps):

            print_gpu_usage("before sample()")
    
            # Sample a mini-batch.
            indices, batch = self.replay_memory.sample(
                self.config.batch_size, 
                self.config.network_optimization_period
            )

            
            print_gpu_usage("after sample()")

            states, actions, q_targets = self.replay_memory.extract_batch(batch, device=self.device)


            
            print_gpu_usage("before encode()")

            # Encode state sequences.
            encoder_output = self.agent.encode(states, random_sampling=False) # We use 'posterior_mean's as representations for stability

            print_gpu_usage("after encode()")
            

            # Estimate Q-values from the episodic memories.
            predicted_q_values = self.agent.lookup_batch(
                representations=encoder_output.representation,
                actions=actions,
                track_key_updates=self.config.key_updates,
            )

            print_gpu_usage("after lookup_batch()")

            # print("After lookup_batch()", torch.cuda.memory_allocated() / 1024**3)

            with torch.no_grad(): 
            # To make sure gradients are not affected by calculations of priorities.

                td_errors_abs = (predicted_q_values - q_targets).detach().abs().cpu().tolist()
                
                if isinstance(self.replay_memory, ReplayMemory):
                    # Prioritized replay memory needs to update priorities.
                    if self.config["normal_memory_kwargs"]["prioritized"]:
                        self.replay_memory.update_priorities(indices, td_errors_abs)

                # Stratified replay memory specifics.
                elif isinstance(self.replay_memory, StratifiedReplayMemory):
                    # Stratified replay memory needs to move transitions between buckets,
                    # and update TD error statistics.
                    
                    if self.replay_memory.first_turn:
                        all_batches.extend(batch)
                        all_td_errors_abs.extend(td_errors_abs)

                        # print("After first turn extend()s", torch.cuda.memory_allocated() / 1024**3)

                        self.replay_memory.register_td_errors(td_errors_abs)
                        # We register all td errors in the first turn.

                    else: # Normal turns.
                        # print("Stratified normal turn before move_between_buckets()")

                        self.replay_memory.move_between_buckets(transitions=batch, td_errors_abs=td_errors_abs)

                        # print("After normal turn move_between_buckets()", torch.cuda.memory_allocated() / 1024**3)

                        self.replay_memory.register_td_errors(td_errors_abs[:indices])
                        # We return 'new_bucket' indices to use them for TD stats.
                        # !! It is the 1+ of last index of transitions to be used for TD stats.

            print_gpu_usage("after torch.no_grad()")


            loss = compute_network_loss(
                predicted_q_values=predicted_q_values,
                q_targets=q_targets,
                encoder_output=encoder_output,
                kl_loss_weight=self.config.kl_loss_weight,
            )

            print_gpu_usage("after compute_network_loss()")


            # Optimize encoder.
            self.encoder_optimizer.zero_grad()
            self.agent.zero_key_gradients()

            print_gpu_usage("after zero_grad()")

            loss['total_loss'].backward()

            print_gpu_usage("after loss.backward()")

            # For sanity check
            # self.agent.check_dnd_key_gradients()

            self.encoder_optimizer.step()

            print_gpu_usage("after encoder_optimizer.step()")

            # Optionally optimize DND keys.
            if self.config.key_updates:
                self.agent.step_key_optimizers()

                print_gpu_usage("after key_optimizer.step()")

            # Store loss to be logged.
            loss['optimization_step'] = self.optimization_step
            losses.append(loss)
            self.optimization_step += 1


        if isinstance(self.replay_memory, StratifiedReplayMemory):
            # Update TD statistics.
            self.replay_memory.update_td_statistics()
            # Update TD stats first turn case handled inside the function update_td_statistics(), no worries.

            # print("After update_td_statistics()", torch.cuda.memory_allocated() / 1024**3)

            if self.replay_memory.first_turn: 
                # We do all transfer operations at once at the beginning for the first turn. 
                self.replay_memory.move_between_buckets(transitions=all_batches, td_errors_abs=all_td_errors_abs)
                self.replay_memory.first_turn = False

                # print("After first turn move_between_buckets()", torch.cuda.memory_allocated() / 1024**3)


                all_batches.clear()
                all_td_errors_abs.clear()

            self.replay_memory.reset_new_bucket() # We don't want to carry over it to the following turns.

            self.replay_memory.report() # Print current circumstances.

            print_gpu_usage("after report()")


        # print(f"GPU memory usage, after network optimization steps: \n {torch.cuda.memory_summary(device=None, abbreviated=False)}")
        

        return indices, losses

    
    ##=========LOGGING_AND_EVALUATION===========

    def _should_checkpoint(self):
        return self.episode % self.config.checkpoint_period == 0
    
    def _should_evaluate(self):
        return self.episode % self.config.evaluation_period == 0
      
    def _logging_step(self, indices: list[int], losses: dict | None, insert_update_counts: dict | None):
        """
        Records training metrics.
        """
        # Multiple optimization steps
        if losses is None or insert_update_counts is None:
            return
        
        total_inserts = 0
        total_updates = 0
        for action_counts in insert_update_counts:
            total_inserts += action_counts["insert"]
            total_updates += action_counts["update"] 
        
        for l in losses:
            self.logger.log(
                optimization_step=l['optimization_step'],
                global_step=self.global_step,
                episode=self.episode,
                total_reward=self.episode_reward,
                total_loss=l["total_loss"].item(),
                td_loss=l["td_loss"].item(),
                kl_loss=l["kl_loss"].item(),
                total_dnd_inserts=total_inserts,
                total_dnd_updates=total_updates
            )
        

        print(f"Episode {self.episode}, optimization step {self.optimization_step}, is fnished.")
        print(self.logger.last())
        print()

        # 'episode_reward' is logged, reset.
        self.episode_reward = 0

        if self.config.get("log_batch_index", False):
            self.batch_index_logger.log(
                optimization_step=self.optimization_step,
                global_step=self.global_step,
                episode=self.episode,
                batch_index=indices
            )

        if self._should_checkpoint(): # Saving logs and checkpoint simultaneously.
            self.logger.save(
                start_step=self.checkpoint_start,
                end_step=self.optimization_step,
                step_name=f"ep_{self.episode}_opt_step",
                clear=True
            )
            
            if self.batch_index_logger:
                self.batch_index_logger.save(
                    start_step=self.checkpoint_start,
                    end_step=self.optimization_step,
                    step_name=f"batch_indices_ep_{self.episode}__opt_step",
                    clear=True
                )

    def _checkpoint_step(self):
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
    
        # dnd_lengths = []
        # dnd_sizes = []
        # for i, dnd in enumerate(self.agent.dnds):
        #     dnd_lengths.append(len(dnd.keys))
        #     dnd_sizes.append(dnd.keys.numel() * dnd.keys.element_size() / 1024**2)

        # print(f"DND lengths: {dnd_lengths} | DND sizes: {dnd_sizes} | total: {np.sum(dnd_sizes)} MB")

        self.checkpoint_manager.save(
            model_checkpoint,
            filename=f"model_ep_{self.episode}_step_{self.optimization_step}",
            colab_execution=self.config.colab_execution
        )

        print(f"Replay memory states total size: {self.replay_memory.get_states_total_size()} MB")
        
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

                insert_update_counts = self._memory_optimization_step()

                indices, losses = self._network_optimization_step()

                self._logging_step(indices, losses, insert_update_counts)    

                if self._should_checkpoint():
                    self._checkpoint_step()

                if self._should_evaluate():
                    print("Evaluating...")
                    
                    evaluation_summary = self.evaluator.evaluate(
                        render_mode='rgb_array',
                        log_file_custom=f"ep_{self.episode}",
                        video_file_custom=f"ep_{self.episode}"
                    )
                    print(evaluation_summary)

                    self._optuna_step(evaluation_summary)


            return self.logger.last()

        finally:

            self.environment.close()


    def load_checkpoint(self):
        """
        Restores a previous training checkpoint.
        """

        print("Loading checkpoint...")

        # -------------------------------------------------
        # Model checkpoint
        # -------------------------------------------------
        model_checkpoint = self.checkpoint_manager.load(
            self.config.resume_checkpoint,
            map_location=self.device,
        )

        self.agent.load_checkpoint_state(model_checkpoint["model"])

        self.encoder_optimizer.load_state_dict(model_checkpoint["optimizer"])

        training_state = model_checkpoint["training_state"]

        self.optimization_step = training_state["optimization_step"]
        self.global_step = training_state["environment_step"]
        self.episode = training_state["episode"]

        self.checkpoint_start = self.optimization_step

        #-------------------------------------------------
        # Manual Optimizer Learning Rate Adjustment
        #-------------------------------------------------
        if self.config.get("manual_lr_adjustment", None) is not None:
            for param_group in self.encoder_optimizer.param_groups:
                param_group['lr'] = self.config.manual_lr_adjustment

        # ! Key optimizer's state is not tracked. 
        # Hence, it can be adjusted from its main config parameter, which is 'dnd_learning_rate'.

        # -------------------------------------------------
        # Replay memory (optional)
        # -------------------------------------------------
        if self.config.load_replay_memory:
            try:
                replay_filename = (
                    self.config.resume_checkpoint
                    .replace("model_", "rep_memo_")
                )

                replay_checkpoint = self.checkpoint_manager.load(
                    replay_filename,
                    map_location="cpu",
                )

                self.replay_memory.load_state_dict(
                    replay_checkpoint["replay_memory"]
                )
            except:
                print("Replay memory checkpoint not found." \
                      "Continuing with fresh replay memory...")

        print(
            f"Checkpoint restored "
            f"(episode={self.episode}, "
            f"optimization_step={self.optimization_step}, "
            f"environment_step={self.global_step})"
            f"epsilon={self.agent.current_epsilon}"
        )


## EXPLANATIONS

# ------------------------------------------------------------------
                    # self.agent.encoder.eval() 
                    # with torch.no_grad(): 
                    #     # Only purpose is getting the representation again.
                    #     # As we do at _environment_step(). Hence, no gradients.

                    #     encoder_output = self.agent.encode(
                    #         frames=(
                    #             torch.from_numpy(state)
                    #             .unsqueeze(0)
                    #             .unsqueeze(2)
                    #             .to(self.device)),
                    #         random_sampling=False,
                    #     )
                    #     transition.representation = encoder_output.representation

                    # self.agent.encoder.train()

## We get into model.eval() mode here due to prevent encoder to 
# track batch norm mean and variance. Because we pass states one by one to encoder.
# If we don't turn off batch norm parameters tracking, they would be very noisy.
# ------------------------------------------------------------------

###===================================================================
###===================================================================

###===========================OLD_FUNCTIONS===========================
