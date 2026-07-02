from pathlib import Path
import json

# Example usage:
# config = TrainingConfig(
#     batch_size=32,
#     learning_rate=2e-4,
#     sequence_length=4,
#     hidden_dim=512,
#     ...
# )

class TrainingConfig:
    """
    Generic experiment configuration.

    Stores arbitrary configuration parameters and provides
    convenient attribute access together with JSON
    serialization.
    """

    def __init__(self, **kwargs):
        self.update(kwargs)

    def update(self, values: dict) -> None:
        """
        Updates configuration parameters.
        """

        for key, value in values.items():
            setattr(self, key, value)

    def setdefault(self, key: str, value) -> None:
        """
        Sets a default value only if the parameter
        does not already exist.
        """

        if not hasattr(self, key):
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """
        Returns the configuration as a dictionary.
        """

        return vars(self).copy()

    def save(self, filepath: str | Path) -> None:
        """
        Saves the configuration as a JSON file.
        """

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load(cls, filepath: str | Path):
        """
        Loads a configuration from a JSON file.
        """

        with open(filepath, "r") as file:
            values = json.load(file)

        return cls(**values)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"