from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, RecordEpisodeStatistics, RecordVideo
import ale_py
gym.register_envs(ale_py) # Explicitly register the Atari games to gym


from utils.gym_wrappers import RestrictedActionWrapper, RewardWrapper
from utils.metrics_logger import MetricsLogger
from utils.misc import ensure_directory
from utils.frame_processing import cut_and_transpose_frame, convert_and_norm_sequence
from models.data_buffers import FrameSequenceBuffer
from utils.training_config import TrainingConfig

from models.nec import NECAgent


class Evaluator:
    """
    Evaluates a trained NEC agent.

    Supports quantitative evaluation, human rendering and optional
    video recording.
    """

    def __init__(
        self,
        agent: NECAgent,
        config: TrainingConfig,
        experiment_dir: str | Path,
        device: str | torch.device = "cpu",
    ):
        self.agent = agent
        self.config = config

        self.device = torch.device(device)
        self.agent.to(self.device)

        self.videos_dir = ensure_directory(Path(experiment_dir) / "videos")

        self.environment = None

        self.sequence_buffer = FrameSequenceBuffer(
            sequence_length=config.sequence_length,
        )
        
        # Metric Loggers
        self.episode_logger = MetricsLogger(experiment_dir)
        self.summary_logger = MetricsLogger(experiment_dir)
        
        self.global_step = 0
        self.episode = 0
        self.evaluation_episodes = self.config.evaluation_episodes

    def _init_environment(
            self,
            render_mode: str | None = None,
            video_file_custom : str | None = None
    ):
        """
        Creates the training environment.
        """
        environment = gym.make(self.config.environment_name, render_mode=render_mode)

        environment = RewardWrapper(environment, strategy='identity')
        # ! For benchmarking, we need to use the original rewards.

        if self.config.action_mapping: # In case of mapping is changed.
            environment = RestrictedActionWrapper(
                environment,
                action_mapping=self.config.action_mapping,
            )

        print("Env metadata: ", environment.metadata)

        if self.config.record_video:
            environment = RecordVideo(
                environment,
                video_folder=self.videos_dir,
                name_prefix=f"evaluation_{video_file_custom}",
                episode_trigger=lambda _: True,
            )
        
        if self.config.grayscale:
            environment = GrayscaleObservation(environment)
        
        # Keep this wrapper at the end.
        environment = RecordEpisodeStatistics(environment)

        return environment

    def _setup(
        self,
        render_mode: str | None = None,
        video_file_custom : str | None = None
    ):
        """
        Creates the evaluation environment.
        """
        if render_mode is None:
            render_mode = "rgb_array" if self.config.record_video else None

        self.environment = self._init_environment(render_mode, video_file_custom)

        observation, _ = self.environment.reset()

        observation = cut_and_transpose_frame(observation)

        self.sequence_buffer.clear()

        for _ in range(self.config.sequence_length):
            self.sequence_buffer.append(observation)


    def _finished(self, episode: int) -> bool:
        """
        Returns whether evaluation has finished.
        """
        return episode >= self.evaluation_episodes


    def _evaluation_episode(self) -> dict:
        """
        Runs one complete evaluation episode.
        """
        
        episode_reward = 0.0
        episode_length = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            state = self.sequence_buffer.get_sequence() # Retrieve preprocessed frame sequence.
        
            encoder_output = self.agent.encode(
                frames=(
                        torch.from_numpy(state)
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .to(self.device)),
                random_sampling=False)

            action, _ = self.agent.choose_action(
                encoder_output=encoder_output,
                exploration=False,
            )

            observation, reward, terminated, truncated, _ = self.environment.step(action)

            observation = cut_and_transpose_frame(observation)
            self.sequence_buffer.append(observation)

            episode_reward += reward
            episode_length += 1
            self.global_step += 1


        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
        }

    def _summarize(self) -> dict:
        """
        Computes aggregate evaluation metrics.
        """

        rewards = np.asarray(self.environment.return_queue, dtype=np.float32)
        lengths = np.asarray(self.environment.length_queue, dtype=np.float32)

        summary = {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "max_reward": float(rewards.max()),
            "min_reward": float(rewards.min()),
            "mean_episode_length": float(lengths.mean()),
            "std_episode_length": float(lengths.std()),
            "num_episodes": len(rewards),
        }

        self.summary_logger.log(
            environment_step=self.global_step, 
            **summary)

        return summary

    def evaluate(
        self,
        render_mode: str | None = None,
        log_file_custom: str | None = None,
        video_file_custom : str | None = None
    ) -> dict:
        """
        Evaluates the current agent.
        """

        self._setup(
            render_mode=render_mode,
            video_file_custom=video_file_custom
        )

        try:
            episode = 0

            self.agent.eval() 
            with torch.inference_mode():
                # Set agent to evaluation mode and getting inference mode.
                # We don't want to gather gradients, and its must for batch norm 
                # and dropout layers if they are used.

                while True:
                    
                    print(f"Evaluating episode {episode} ...")
                    logs = self._evaluation_episode()

                    self.episode_logger.log(
                        environment_step=self.global_step,
                        episode=episode,
                        **logs,
                    )

                    episode += 1

                    # We need to reset the environment if evaluation is not finished.
                    # Otherwise, an unnecessary new video recording will be created.
                    # That causes error on Kaggle platform on top of created unncessary empty video files.
                    
                    if not self._finished(episode):
                        print(f"Evaluatinon episode {episode}.")

                        observation, _ = self.environment.reset()

                        observation = cut_and_transpose_frame(observation)

                        self.sequence_buffer.clear()

                        for _ in range(self.config.sequence_length):
                            self.sequence_buffer.append(observation)

                    else:
                        # All evaluation episodes are done.
                        break



            # Summaries of each evaluation episodes
            summary = self._summarize()

            # Save metrics
            self.episode_logger.save(
                start_step=self.global_step,
                end_step=self.global_step,
                step_name=f"eval_episodes_{log_file_custom}",
            )
            self.summary_logger.save(
                start_step=self.global_step,
                end_step=self.global_step,
                step_name=f"eval_summary_{log_file_custom}",
            )

            return summary

        finally:

            self.environment.close()

            self.agent.train()


    