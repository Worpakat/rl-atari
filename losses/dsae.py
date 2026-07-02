import torch
import torch.nn.functional as F

def _apply_reduction(loss, reduction):
    """
    Utility for reduction method.
    """
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def reconstruction_loss(
    input_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Reconstruction loss between the input and reconstructed frame sequences.
    """
    return F.mse_loss(
        input=reconstructed_frames,        
        target=input_frames,
        reduction=reduction
    )


def content_kl_loss(
    content_mean: torch.Tensor,
    content_logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL divergence between the content posterior and the unit Gaussian prior.
    """

    loss = -0.5 * (1 + content_logvar - content_mean.pow(2) - content_logvar.exp())
    loss = loss.sum(dim=1)

    return _apply_reduction(loss, reduction)


def dynamics_kl_loss(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL divergence between the dynamics posterior and learned dynamics prior.
    """

    posterior_var = posterior_logvar.exp()
    prior_var = prior_logvar.exp()

    loss = 0.5 * (
        prior_logvar
        - posterior_logvar
        + (
            posterior_var
            + (posterior_mean - prior_mean).pow(2)
        ) / prior_var
        - 1
    )
    loss = loss.sum(dim=(1, 2))
    
    return _apply_reduction(loss, reduction)


def dsae_loss(
    input_frames: torch.Tensor,
    reconstructed_frames: torch.Tensor,
    content_mean: torch.Tensor,
    content_logvar: torch.Tensor,
    dynamics_posterior_mean: torch.Tensor,
    dynamics_posterior_logvar: torch.Tensor,
    dynamics_prior_mean: torch.Tensor,
    dynamics_prior_logvar: torch.Tensor,
    reconstruction_weight: float = 1.0,
    content_kl_weight: float = 1.0,
    dynamics_kl_weight: float = 1.0,
):
    """
    Complete DSAE objective.
    """

    recon = reconstruction_loss(
        input_frames,
        reconstructed_frames,
        reduction="mean",
    )

    content_kl = content_kl_loss(
        content_mean,
        content_logvar,
        reduction="mean",
    )

    dynamics_kl = dynamics_kl_loss(
        dynamics_posterior_mean,
        dynamics_posterior_logvar,
        dynamics_prior_mean,
        dynamics_prior_logvar,
        reduction="mean",
    )

    total = (
        reconstruction_weight * recon
        + content_kl_weight * content_kl
        + dynamics_kl_weight * dynamics_kl
    )

    return {
        "total": total,
        "reconstruction": recon,
        "content_kl": content_kl,
        "dynamics_kl": dynamics_kl,
    }