from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from models.dnd import LookupResult, DND, MemoryEntry
from utils.data_buffers import Transition

@dataclass(slots=True)
class MemoryUpdateRequest:
    """
    Describes a single memory update operation.

    This object is produced by a memory update strategy after computing
    the desired modification for one DND entry. It contains all
    information required by the DND to safely apply the update while
    remaining independent of the particular update strategy.

    Attributes:
        update_or_insert: 
            Whether purpose is an update or an insert of a new entry.
        
        action:
            Action whose DND should be updated.

        index:
            Index of the primary memory entry to update. ``None`` if a
            new entry should be inserted.

        key:
            State representation associated with the memory entry.
        
        is_change:
            Whether the update value is a change amount or a new value for assignment.

        update_value:
            Update value to be added to or assigned to the associated memory value.

        
        generation:
            Generation number of the primary memory entry. Used to verify
            that the entry has not been overwritten before the update is
            applied.

        auxiliary:
            Optional auxiliary representation associated with the key.

        """

    update_or_insert: str
    action: int
    key: torch.Tensor
    is_change: bool 
    index: int | None 
    update_value: torch.Tensor

    generation: int | None
    auxiliary: torch.Tensor | None = None



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

    def __init__(
            self, 
            learning_rate: float
            ) -> None:
        
        self.learning_rate = learning_rate

    def calculate_bellman_update_change(self, current_value: torch.Tensor, q_target: torch.Tensor):
        """
        """
        return self.learning_rate * (q_target - current_value)
    
    @abstractmethod
    def calculate_memory_update_request(
        self,
        dnd: DND,
        transition: Transition,
        q_target: torch.Tensor,
        lookup_result: LookupResult,
        exploration_update: bool = False
        ) -> list[MemoryUpdateRequest]:
        
        """
        Makes update calculations for DND memory state values according to implemented spesific strategy. 
        And returns to be updated informations as list of MemoryUpdateRequest.
        Does not applies yet.
        """

        raise NotImplementedError

    def apply(self, dnds: list[DND],update_requests: list[MemoryUpdateRequest]) -> None:
        """
        Applies all prepared memory updates.

        Existing entries are updated before any new entries are inserted so
        that pending insertions cannot invalidate existing memory indices.
        """

        requests_by_action: dict[int, list[MemoryUpdateRequest]] = {}

        for request in update_requests:
            requests_by_action.setdefault(request.action, []).append(request)

        for action, requests in requests_by_action.items():

            dnd = dnds[action]

            updates = [request for request in requests if request.update_or_insert == "update"]
            inserts = [request for request in requests if request.update_or_insert == "insert"]

            # Existing memory updates.
            update_changes = [request for request in updates if request.is_change]
            update_assigns = [request for request in updates if not request.is_change]

            if update_changes:

                indices = torch.tensor(
                    [request.index for request in update_changes],
                    dtype=torch.long,
                    device=dnd.device,
                )
                changes = torch.stack([request.update_value for request in update_changes])

                dnd.update(indices=indices, changes=changes)

            if update_assigns:

                indices = torch.tensor(
                    [request.index for request in update_assigns], 
                    dtype=torch.long, 
                    device=dnd.device)
                values = torch.stack([request.update_value for request in update_assigns])

                dnd.update(indices=indices, values=values)

            
            # New memory entries.
            if inserts:

                keys = [request.key for request in inserts]
                values = [request.update_value for request in inserts]

                dnd.insert(keys, values)

                dnd.commit()