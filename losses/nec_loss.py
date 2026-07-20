import torch
import torch.nn.functional as F

from losses.dsae import _apply_reduction, dynamics_kl_loss
from models.seq_enc import EncoderOutput

def compute_network_loss(
    predicted_q_values: torch.Tensor,
    q_targets: torch.Tensor,
    encoder_output: EncoderOutput,
    kl_loss_weight: float = 1.0,
    reduction: str = "mean",
) -> dict[str, torch.Tensor]:
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
    total_loss = td_loss + kl_loss_weight * kl_loss
    
    print(f"TD loss shape: {td_loss.shape} | KL loss: {kl_loss.shape} | Total loss: {total_loss.shape}")

    return {
        "total_loss": total_loss,
        "td_loss": td_loss,
        "kl_loss": kl_loss,
    }