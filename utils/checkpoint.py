from pathlib import Path

import torch

from utils.misc import ensure_directory

## Example Checkpoint Structure
# checkpoint = {
#     "dsae": dsae.state_dict(),
#     "content_adapter": content_adapter.state_dict(),
#     "dynamics_adapter": dynamics_adapter.state_dict(),
#     "encoder_optimizer": encoder_optimizer.state_dict(),
#     "decoder_optimizer": decoder_optimizer.state_dict(),
#     "replay_buffer": replay_buffer.state_dict(),
#     "dnd": dnd.state_dict(),
#     "training_state": training_state,
# }

class CheckpointManager:
    """
    Generic checkpoint manager.

    Stores and loads arbitrary checkpoint dictionaries. The manager
    is intentionally architecture-agnostic and can therefore be used
    with any combination of models, optimizers, replay buffers,
    schedulers, memories or custom data structures.
    """

    def __init__(self, experiment_dir: str | Path):
        self.checkpoints_dir = ensure_directory(Path(experiment_dir) / "checkpoints")

    def save(
        self, 
        checkpoint: dict, 
        filename: str, 
        colab_execution: bool,
        kaggle_execution: bool,
        ) -> Path:
        """
        Saves a checkpoint dictionary.

        Parameters
        ----------
        checkpoint:
            Dictionary containing any serializable objects.

        filename:
            Checkpoint filename ('.pt' will be appended if omitted).

        Returns
        -------
        Path
            Path to the saved checkpoint.
        """

        if not filename.endswith(".pt"):
            filename += ".pt"

        if colab_execution: # Save to colab session local storage
            save_dir = Path("/content/checkpoints")
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / filename
            
        elif kaggle_execution: 
            # Save to kaggle session local storage with respect to how we save checkpoints as a dataset.
            save_dir = Path("/kaggle/working")
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / filename

        else:
            # Save to experiment's checkpoints directory
            filepath = self.checkpoints_dir / filename

        torch.save(checkpoint, filepath)

        return filepath

    def load(
        self, filename: str | Path, map_location=None, 
        colab_execution: bool = False,
        kaggle_execution: bool = False,
        ) -> dict:
        """
        Loads a checkpoint.

        Returns
        -------
        dict
            Loaded checkpoint dictionary.
        """
        if not filename.endswith(".pt"):
            filename += ".pt"
        
        if colab_execution: # Load from colab session local storage
            local_checkpoint_dir = Path("/content/checkpoints")
            self.checkpoints_dir = local_checkpoint_dir

        #THIS ONE IS TEMPORARY, NEED TO BE REMOVED LATER OR REPLACED WITH A BETTER SOLUTION
        if kaggle_execution: # Load from kaggle session local storage
            local_checkpoint_dir = Path("/kaggle/input/worpakat/lon-run-0-checkpoint-350")
            # local_checkpoint_dir = Path("/kaggle/working")
            filepath = local_checkpoint_dir / filename
        else:
            filepath = self.checkpoints_dir / filename

        return torch.load(filepath, map_location=map_location)

    def latest(self) -> Path | None:
        """
        Returns the latest checkpoint.
        """

        checkpoints = self.list()

        if not checkpoints:
            return None

        return checkpoints[-1]

    def list(self) -> list[Path]:
        """
        Returns all checkpoints sorted by filename.
        """

        return sorted(self.checkpoints_dir.glob("*.pt"))

    def cleanup(self, keep_last: int) -> None:
        """
        Deletes the oldest checkpoints until only
        'keep_last' checkpoints remain.
        """

        checkpoints = self.list()

        while len(checkpoints) > keep_last:
            checkpoints[0].unlink()
            checkpoints.pop(0)