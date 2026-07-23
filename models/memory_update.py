from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from models.dnd import LookupResult, DND
from models.transition_classes import Transition

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
    update_value: torch.Tensor
    index: int | None = None

    generation: int | None = None
    auxiliary: torch.Tensor | None = None

@dataclass(slots=True)
class LookupRequirements:
    """
    """
    return_indices: bool = False
    return_similarities: bool = False
    return_neighbors: bool = False



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

    def __init__(self,  learning_rate: float) -> None:
        
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

        insert_update_counts = [{"insert": 0, "update": 0} for _ in dnds]

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

            insert_update_counts[action]["insert"] += len(inserts)
            insert_update_counts[action]["update"] += len(updates)

        return insert_update_counts


class OriginalNECUpdateStrategy(MemoryUpdateStrategy):
    """
    Inserts new memory entries as original NEC does: Assigns N-step Q target directly.
    """
    def __init__(
        self,
        learning_rate: float
        ) -> None:
        super().__init__(learning_rate)
        
        self.lookup_requirements = LookupRequirements() # All False

    def calculate_memory_update_request(
        self,
        dnd: DND,
        transition: Transition,
        q_target: torch.Tensor,
        lookup_result: LookupResult | None = None,
        exploration_update: bool = False
        ) -> list[MemoryUpdateRequest]:
        
        return [MemoryUpdateRequest(
            update_or_insert='insert',
            action=transition.action,
            key=transition.representation,
            index=None,
            is_change=False,
            update_value=q_target,
        )]


class Option1UpdateStrategy(MemoryUpdateStrategy):
    """
    Theoretically first, assigns lookup estimate, 
    then applies Bellman update with N-step Q target.

    exploration_lr:
        To keep N-step Q target's effect high in case of action selected randomly.
        It should be a high rate, e.g. 0.8, 0.9 etc. 
        Hence it resembles original NEC insertion, assigning N-step Q target directly.
    """
    def __init__(self, learning_rate: float, exploration_lr: float) -> None:
        super().__init__(learning_rate)
        
        self.exploration_lr = exploration_lr

        self.lookup_requirements = LookupRequirements() # All False

    def calculate_exploration_update_change(self, current_value: torch.Tensor, q_target: torch.Tensor):
        return self.exploration_lr * (q_target - current_value)

    def calculate_memory_update_request(
        self,
        dnd: DND,
        transition: Transition,
        q_target: torch.Tensor,
        lookup_result: LookupResult,
        exploration_update: bool = False
        ) -> list[MemoryUpdateRequest]:

        # Exploitation action
        if not transition.is_exploration_action: 
            td_error = self.calculate_bellman_update_change(lookup_result.value, q_target)
            update_value = lookup_result.value + td_error

            return [MemoryUpdateRequest(
                update_or_insert='insert',
                action=transition.action,
                key=transition.representation,
                index=None,
                is_change=False,
                update_value=update_value,
            )]

        # Exploration action
        if exploration_update: # Exploration update mode is active
            td_error = self.calculate_exploration_update_change(lookup_result.value, q_target)
            update_value = lookup_result.value + td_error 

            return [MemoryUpdateRequest(
                update_or_insert='insert',
                action=transition.action,
                key=transition.representation,
                index=None,
                is_change=False,
                update_value=update_value,
            )]
        
        else: # Exploration update mode is inactive, drops to original update
            return [MemoryUpdateRequest(
                update_or_insert='insert',
                action=transition.action,
                key=transition.representation,
                index=None,
                is_change=False,
                update_value=q_target,
            )]


class Option2UpdateStrategy(Option1UpdateStrategy):
    """
    neighbor_shrink:
        Used to shrink neighbor updates more.
    """
    def __init__(self, learning_rate, exploration_lr, neighbor_shrink):
        super().__init__(learning_rate, exploration_lr)

        self.neighbor_shrink = neighbor_shrink

        self.lookup_requirements = LookupRequirements(
                                            return_similarities=True, # Similarity scores
                                            return_neighbors=True, # Keys and values
                                            return_indices=True, # Neighbor indices
                                            )

    def calculate_memory_update_request(
        self,
        dnd: DND,
        transition: Transition,
        q_target: torch.Tensor,
        lookup_result: LookupResult,
        exploration_update: bool = False
        ) -> list[MemoryUpdateRequest]:

        # Neighbor information retrieved from lookup
        neighbor_indices = lookup_result.neighbor_indices
        neighbor_keys = lookup_result.neighbor_keys
        neighbor_values = lookup_result.neighbor_values
        neighbor_similarities = lookup_result.similarities


        # Exploitation action
        if not transition.is_exploration_action: 
            td_error = self.calculate_bellman_update_change(lookup_result.value, q_target)
            state_update_value = (lookup_result.value + td_error).squeeze().to(dnd.device) 
            # Squeezed to make it scalar as 'q_target'. 
            # If 'exploration_update' is not active, it drops to original NEC update, which is assigning N-step 'q_target' directly.


            q_target_tensor = torch.full_like(neighbor_values, fill_value=q_target.item()).to(dnd.device)
            scalar_rates= self.learning_rate * self.neighbor_shrink * neighbor_similarities.unsqueeze(1)
            neighbor_update_values = neighbor_values + scalar_rates * (q_target_tensor - neighbor_values)

            requests = [MemoryUpdateRequest(
                update_or_insert='update',
                action=transition.action,
                key=key,
                index=index,
                is_change=False, # ! We've done summation with old values.
                update_value=update_value,
            ) for index, key, update_value in zip(neighbor_indices, neighbor_keys, neighbor_update_values)]
            
            requests.append(MemoryUpdateRequest(
                update_or_insert='insert',
                action=transition.action,
                key=transition.representation,
                index=None,
                is_change=False,
                update_value=state_update_value,
            ))

            return requests

        # Exploration action
        if exploration_update: # Exploration update mode is active
            td_error = self.calculate_exploration_update_change(lookup_result.value, q_target) # !!
            state_update_value = (lookup_result.value + td_error).squeeze().to(dnd.device)

            q_target_tensor = torch.full_like(neighbor_values, fill_value=q_target.item()).to(dnd.device)
            scalar_rates= self.learning_rate * self.neighbor_shrink * neighbor_similarities.unsqueeze(1) * (1 - self.exploration_lr)
            neighbor_update_values = neighbor_values + scalar_rates * (q_target_tensor - neighbor_values)
            ##!!! We shrink scalar_rates more `1 - exploration_lr`. By this way, updates' effect drops significantly.
            # Because eventually, lookup results are not actually used for decision of this transition. 

            requests = [MemoryUpdateRequest(
                update_or_insert='update',
                action=transition.action,
                key=key,
                index=index,
                is_change=False, # ! We've done summation with old values.
                update_value=update_value,
            ) for index, key, update_value in zip(neighbor_indices, neighbor_keys, neighbor_update_values)]
            

            requests.append(MemoryUpdateRequest(
                update_or_insert='insert',
                action=transition.action,
                key=transition.representation,
                index=None,
                is_change=False,
                update_value=state_update_value,
            ))

            return requests
        
        else: # Exploration update mode is inactive, drops to original update

            return [MemoryUpdateRequest(
                update_or_insert='insert',
                action=transition.action,
                key=transition.representation,
                index=None,
                is_change=False,
                update_value=q_target,
            )]
