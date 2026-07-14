from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import ale_py
gym.register_envs(ale_py) # Explicitly register the Atari games to gym


from utils.action_wrapper import RestrictedActionWrapper
from utils.metrics_logger import MetricsLogger
from utils.misc import ensure_directory, preprocess_frame
from utils.data_buffers import FrameSequenceBuffer
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

    def _setup(
        self,
        render_mode: str | None = None,
        record_video: bool = False,
        video_file_custom : str | None = None
    ):
        """
        Creates the evaluation environment.
        """

        if render_mode is None:
            render_mode = "rgb_array" if record_video else None

        self.environment = gym.make(self.config.environment_name, render_mode=render_mode)

        if self.config.record_video:
            self.environment = RecordVideo(
                self.environment,
                video_folder=self.videos_dir,
                name_prefix=f"evaluation_{video_file_custom}",
                episode_trigger=lambda _: True,
            )
        
        if self.config.action_mapping: # In case of mapping is changed.
            self.environment = RestrictedActionWrapper(
                self.environment,
                action_mapping=self.config.action_mapping,
            )

        self.environment = RecordEpisodeStatistics(self.environment)

        observation, _ = self.environment.reset()

        observation = preprocess_frame(observation)

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

            frames = torch.from_numpy(
                self.sequence_buffer.get_sequence()
            ).unsqueeze(0).to(self.device)

            self.agent.eval()

            with torch.inference_mode():
                encoder_output = self.agent.encode(frames=frames, random_sampling=False)

                action, _ = self.agent.choose_action(
                    encoder_output=encoder_output,
                    exploration=False,
                )

            observation, reward, terminated, truncated, _ = self.environment.step(action)

            observation = preprocess_frame(observation)

            self.sequence_buffer.append(observation)

            episode_reward += reward
            episode_length += 1

        observation, _ = self.environment.reset()

        observation = preprocess_frame(observation)

        self.sequence_buffer.clear()

        for _ in range(self.config.sequence_length):
            self.sequence_buffer.append(observation)

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
        num_episodes: int,
        render_mode: str | None = None,
        log_file_custom: str | None = None,
        record_video: bool = False,
        video_file_custom : str | None = None
    ) -> dict:
        """
        Evaluates the current agent.
        """

        self._setup(
            render_mode=render_mode,
            record_video=record_video,
            video_file_custom=video_file_custom
        )

        try:
            episode = 0

            while not self._finished(episode):
                
                print(f"Evaluating episode {episode} ...")
                logs = self._evaluation_episode()

                self.episode_logger.log(
                    environment_step=self.global_step,
                    episode=episode,
                    **logs,
                )

                episode += 1


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


    