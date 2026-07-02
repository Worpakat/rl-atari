from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.misc import ensure_directory


class ReconstructionRecorder:
    """
    Records and visualizes reconstructed frame sequences during training.
    """

    def __init__(self, experiment_dir: str | Path, save_every: int = 500):
        self.recordings_dir = ensure_directory(Path(experiment_dir) / "reconstructions")
        self.save_every = save_every

    def should_record(self, step: int) -> bool:
        """
        Returns whether a reconstruction should be recorded.
        """

        return step % self.save_every == 0

    def save(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
        name: str,
        save_plot: bool = True,
    ) -> Path:
        """
        Saves a reconstruction recording.

        Parameters
        ----------
        original:
            Tensor of shape (T, C, H, W).

        reconstruction:
            Tensor of shape (T, C, H, W).

        name:
            Recording identifier.

        save_plot:
            Whether to save a visualization image.

        Returns
        -------
        Path
            Directory containing the recording.
        """

        recording_dir = ensure_directory(self.recordings_dir / name)

        torch.save(
            {
                "original": original.detach().cpu(),
                "reconstruction": reconstruction.detach().cpu(),
            },
            recording_dir / "sequence.pt",
        )

        if save_plot:
            self.plot(original, reconstruction, save_path=recording_dir / "comparison.png")

        return recording_dir

    def load(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a saved reconstruction recording.
        """

        data = torch.load(self.recordings_dir / name / "sequence.pt")

        return (data["original"], data["reconstruction"])

    def plot(
        self,
        original: torch.Tensor | None = None,
        reconstruction: torch.Tensor | None = None,
        recording: str | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
    ):
        """
        Plots a reconstruction comparison.

        Either provide tensors directly or specify a saved recording.
        """

        if recording is not None:
            original, reconstruction = self.load(recording)

        if original is None or reconstruction is None:
            raise ValueError(
                "Either tensors or a recording name must be provided."
            )

        original = self._prepare(original)
        reconstruction = self._prepare(reconstruction)

        num_frames = original.shape[0]

        fig, axes = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6))

        if num_frames == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_frames):
            axes[0, i].imshow(original[i])
            axes[0, i].axis("off")

            axes[1, i].imshow(reconstruction[i])
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Reconstruction")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

    @staticmethod
    def _prepare(
        tensor: torch.Tensor,
    ):
        """
        Converts a tensor into a NumPy array suitable for visualization.
        """

        tensor = tensor.detach().cpu()

        if tensor.ndim != 4:
            raise ValueError(
                "Expected tensor of shape (T, C, H, W)."
            )

        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0.0, 1.0)

        return tensor.permute(0, 2, 3, 1).numpy()