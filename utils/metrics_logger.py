from pathlib import Path

import pandas as pd

from utils.misc import ensure_directory


class MetricsLogger:
    """
    Generic experiment metrics logger.

    Stores arbitrary scalar metrics and periodically saves them into
    separate CSV files. Existing log files can later be loaded and
    concatenated for analysis.
    """

    def __init__(self, experiment_dir: str | Path):
        self.logs_dir = ensure_directory(Path(experiment_dir) / "logs")
        self.records = []

    def log(self, **metrics) -> None:
        """
        Stores one metrics record.
        """
        self.records.append(metrics)

    def save(
        self,
        start_step: int | None,
        end_step: int | None,
        step_name: str = "step",
        clear: bool = True,
    ) -> Path:
        """
        Saves currently stored metrics as a CSV file.

        Returns
        -------
        Path
            Path of the created CSV file.
        """

        filename = (f"metrics_{step_name}")
        if start_step or end_step:
            filename += f"_{start_step}_{end_step}"

        filename += ".csv"

        filepath = self.logs_dir / filename

        dataframe = pd.DataFrame(self.records)
        dataframe.to_csv(
            filepath,
            index=False,
        )

        if clear:
            self.clear()

        return filepath

    def load(self) -> pd.DataFrame:
        """
        Loads and concatenates all saved metric files.

        Returns
        -------
        pd.DataFrame
        """

        csv_files = sorted(self.logs_dir.glob("metrics_*.csv"))

        if len(csv_files) == 0:
            return pd.DataFrame()

        dataframes = [pd.read_csv(file) for file in csv_files]

        return pd.concat(dataframes, ignore_index=True)

    def clear(self) -> None:
        """
        Removes all in-memory metric records.
        """
        self.records.clear()

    def last(self) -> dict | None:
        """
        Returns the latest logged record.
        """

        if not self.records:
            return None

        return self.records[-1]

    def __len__(self) -> int:
        return len(self.records)