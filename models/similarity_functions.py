import torch
import torch.nn.functional as F

def exponential_inverse(
        key: torch.Tensor,
        neighbor_keys: torch.Tensor,
        neighbor_auxiliary: torch.Tensor | None = None,
        **kwargs
    ):
    """
    Computes exponential inverse-distance similarities.

    Similarity is defined as

        exp(-||k - k_i||_2)

    for every neighboring key.

    Parameters
    ----------
    key:
        Query key of shape (feature_dim,).

    neighbor_keys:
        Neighbor keys of shape (num_neighbors, feature_dim).

    neighbor_auxiliary:
        Unused. Included for a consistent similarity function interface.

    Returns
    -------
    torch.Tensor
        Similarity tensor of shape (num_neighbors,).
    """


    distances = torch.norm(neighbor_keys - key, dim=1).unsqueeze(1)

    return torch.exp(-distances * kwargs.get("similarity_scale", 1.0))


def inverse_distance(
        key: torch.Tensor,
        neighbor_keys: torch.Tensor,
        neighbor_auxiliary: torch.Tensor | None = None,
    ):
    """
    Computes inverse distance similarities.

    Similarity is defined as

        1 / (1 + ||k - k_i||_2)

    for every neighboring key.

    Parameters
    ----------
    key:
        Query key of shape (feature_dim,).

    neighbor_keys:
        Neighbor keys of shape (num_neighbors, feature_dim).

    neighbor_auxiliary:
        Unused. Included for a consistent similarity function interface.

    Returns
    -------
    torch.Tensor
        Similarity tensor of shape (num_neighbors,).
    """
    distances = torch.norm(neighbor_keys - key, dim=1).unsqueeze(1)

    return 1.0 / (0.0001 + distances)
