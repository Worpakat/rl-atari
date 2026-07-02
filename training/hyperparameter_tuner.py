from pathlib import Path

import optuna
import torch
import shutil

from utils.misc import ensure_directory


class HyperparameterTuner:
    """
    Generic Optuna wrapper for trainer-based experiments.

    The tuner is responsible for creating studies, managing trial
    directories, and launching trainer instances. It does not contain
    any model- or trainer-specific logic.
    """

    def __init__(
        self,
        study_name,
        experiment_root,
        trainer_class,
        model_builder,
        optimizer_builder,
        config_builder,
        objective_metric,
        device: str | torch.device = "cpu",
        direction="minimize",
        storage="sqlite:///study.db",
        keep_best_models=3,
    ):
        self.study_name = study_name
        self.experiment_root = ensure_directory(experiment_root)
        self.study_directory = ensure_directory(self.experiment_root / study_name)
        self.trials_directory = ensure_directory(self.study_directory / "trials")
        self.best_models_directory = ensure_directory(self.study_directory / "best_models")

        self.trainer_class = trainer_class
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        self.config_builder = config_builder
        self.device = torch.device(device)

        self.objective_metric = objective_metric
        self.keep_best_models = keep_best_models

        storage_path = (self.study_directory / Path(storage).name)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
        )

    def optimize(self, n_trials):
        """
        Starts or continues the Optuna study.
        """

        self.study.optimize(self._objective, n_trials=n_trials)

    def _objective(self, trial):
        """
        Executes one Optuna trial.
        """

        trial_directory = ensure_directory(self.trials_directory / f"trial_{trial.number:04d}")

        config = self.config_builder(trial)
        config.save(trial_directory / "config.json")
        
        model = self.model_builder(config)
        optimizer = self.optimizer_builder(model, config)

        trainer = self.trainer_class(
            model=model,
            optimizer=optimizer,
            config=config,
            experiment_dir=trial_directory,
            device=self.device,
            trial=trial,
        )

        last_metrics = trainer.train()

        self._handle_best_models(trial)

        if self.objective_metric not in last_metrics:
            raise KeyError(
                f"Metric '{self.objective_metric}' was not returned by the trainer."
            )

        return last_metrics[self.objective_metric]


    def _completed_trials(self):
        """
        Returns all completed trials sorted by objective value.
        """

        completed = [
            trial
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        reverse = (self.study.direction == optuna.study.StudyDirection.MAXIMIZE)

        completed.sort(key=lambda trial: trial.value, reverse=reverse)

        return completed

    def _handle_best_models(self, trial):
        """
        Keeps only the best completed trial models.
        """

        completed = self._completed_trials()

        best_trial_numbers = {
            completed_trial.number
            for completed_trial in completed[: self.keep_best_models]
        }

        source_directory = (self.trials_directory / f"trial_{trial.number:04d}" / "checkpoints")
        destination_directory = (self.best_models_directory / f"trial_{trial.number:04d}")

        if (trial.number in best_trial_numbers and source_directory.exists()):
            if destination_directory.exists():
                shutil.rmtree(destination_directory)

            shutil.copytree(source_directory, destination_directory)

        for directory in self.best_models_directory.iterdir():

            if not directory.is_dir():
                continue

            trial_number = int(directory.name.split("_")[1])

            if trial_number not in best_trial_numbers:
                shutil.rmtree(directory)