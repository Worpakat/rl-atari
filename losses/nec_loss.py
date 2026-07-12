import torch
import torch.nn.functional as F

from losses.dsae import _apply_reduction, dynamics_kl_loss
from models.seq_enc import EncoderOutput

def compute_network_loss(
    predicted_q_values: torch.Tensor,
    q_targets: torch.Tensor,
    encoder_output: EncoderOutput,
    dynamics_kl_loss_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes the optimization loss for the encoder.
    """

    # TD loss
    td_loss = F.mse_loss(predicted_q_values, q_targets, reduction=reduction)

    # Temporal KL loss
    kl_loss = dynamics_kl_loss(
        encoder_output.posterior_mean,
        encoder_output.posterior_logvar,
        encoder_output.prior_mean,
        encoder_output.prior_logvar,
        reduction=reduction,
    )

    # Total loss
    total_loss = (
        (1 - dynamics_kl_loss_weight) * td_loss
        + dynamics_kl_loss_weight * kl_loss
    )

    return {
        "total_loss": total_loss,
        "td_loss": td_loss,
        "dynamics_kl_loss": kl_loss,
    }