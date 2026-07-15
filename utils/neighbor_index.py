from abc import ABC, abstractmethod

import torch
import faiss


class NeighborIndex(ABC):
    """
    Abstract nearest-neighbor index used by the DND.

    Concrete implementations may use FAISS, KD-Tree, HNSW,
    ScaNN or any other indexing backend.
    """

    @abstractmethod
    def build(
        self,
        keys: torch.Tensor,
    ) -> None:
        """
        Builds or rebuilds the searchable index from the given keys.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Returns the indices of the k nearest neighbors.
        """
        raise NotImplementedError

    @abstractmethod
    def find(
        self,
        query: torch.Tensor,
    ) -> int | None:
        """
        Returns the exact (or duplicate-threshold) index of the query if it
        already exists in memory. Otherwise returns None.
        """
        raise NotImplementedError


class FaissIndex(NeighborIndex):
    """
    FAISS-based exact nearest-neighbor index using Euclidean distance.
    """

    def __init__(
        self,
        feature_dim: int,
        duplicate_threshold: float = 1e-6,
    ):
        self.feature_dim = feature_dim
        self.duplicate_threshold = duplicate_threshold

        self.index = faiss.IndexFlatL2(feature_dim)
        self.keys: torch.Tensor | None = None

    def build(self, keys: torch.Tensor) -> None:
        """
        Builds (or rebuilds) the FAISS index.
        """
        self.keys = keys
        self.index.reset()

        if len(keys) == 0:
            return

        self.index.add(
            keys.detach().cpu().numpy()
        )

    def search(self, query: torch.Tensor, k: int) -> torch.Tensor:
        """
        Returns indices of the k nearest neighbors.
        """

        if self.keys is None or len(self.keys) == 0:
            return torch.empty(0, dtype=torch.long)

        k = min(k, len(self.keys))

        _, indices = self.index.search(
            query.detach().cpu().numpy(),
            k,
        )

        return torch.from_numpy(indices[0]).long()

    def find(self, query: torch.Tensor) -> int | None:
        """
        Returns the index of an existing key if one lies within the
        duplicate threshold.
        """

        if self.keys is None or len(self.keys) == 0:
            return None

        _, indices = self.index.search(
            query.detach().cpu().numpy(),
            1,
        )

        index = int(indices[0, 0])

        print(f"query device: {query.device}, keys device: {self.keys.device}")


        distance = torch.norm(
            self.keys[index] - query
        )



        if distance <= self.duplicate_threshold:
            return index

        return None