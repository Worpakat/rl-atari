from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from models.dnd import LookupResult, DND

@dataclass(slots=True)
class MemoryUpdateRequest:
    """
    Describes a single memory update operation.

    This object is produced by a memory update strategy after computing
    the desired modification for one DND entry. It contains all
    information required by the DND to safely apply the update while
    remaining independent of the particular update strategy.

    Attributes:
        action:
            Action whose DND should be updated.

        index:
            Index of the primary memory entry to update. ``None`` if a
            new entry should be inserted.

        generation:
            Generation number of the primary memory entry. Used to verify
            that the entry has not been overwritten before the update is
            applied.

        key:
            State representation associated with the memory entry.

        value:
            Updated value to assign to the memory entry.

        auxiliary:
            Optional auxiliary representation associated with the key.

        neighbor_indices:
            Optional neighboring entry indices affected by the update.

        neighbor_generations:
            Generation numbers corresponding to the neighboring entries.

        neighbor_values:
            Updated values for the neighboring entries.

        neighbor_weights:
            Optional weights used when updating neighboring entries.
        """

    action: int

    index: int | None
    generation: int | None

    key: torch.Tensor
    value: torch.Tensor
    auxiliary: torch.Tensor | None = None

    neighbor_indices: torch.Tensor | None = None
    neighbor_generations: torch.Tensor | None = None
    neighbor_values: torch.Tensor | None = None
    neighbor_weights: torch.Tensor | None = None


class MemoryUpdateStrategy(ABC):
    """
    Base class for DND memory update strategies.

    A memory update strategy defines how newly computed target values are
    incorporated into the differentiable neural dictionaries. Different
    strategies may update only the queried memory entry, propagate
    updates to neighboring entries, or implement any other memory update
    rule.

    Implementations are responsible for generating the required memory
    update requests from lookup results and target values, then applying
    those requests to the corresponding DNDs.
    """

    @abstractmethod
    def apply(
        self,
        dnds: list[DND],
        lookup_results: list[LookupResult],
        actions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
    ) -> None:
        """
        Applies memory updates to the provided DNDs.

        Args:
            dnds:
                Action-specific differentiable neural dictionaries.

            lookup_results:
                Lookup results corresponding to the queried states.

            actions:
                Action index for each sample.

            keys:
                State representations associated with the updates.

            values:
                Target values (typically N-step targets) to be written
                into memory.

            auxiliary:
                Optional auxiliary representations associated with the
                keys.
        """

        raise NotImplementedError